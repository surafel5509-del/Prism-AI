import OpenAI from "openai";

export const runtime = "nodejs";
export const maxDuration = 60;

const MAX_INPUT_CHARS = 60_000;
const MAX_MESSAGES = 60;
const OUTPUT_LIMITS = new Set([4096, 8192, 12288, 16384]);
const DEFAULT_MODEL = "z-ai/glm-5.3-flash";

function client() {
  const apiKey = process.env.NVIDIA_API_KEY;
  if (!apiKey) {
    const error = new Error("Server setup is incomplete: NVIDIA_API_KEY is missing in Vercel Environment Variables.");
    error.status = 500;
    throw error;
  }
  return new OpenAI({
    baseURL: "https://integrate.api.nvidia.com/v1",
    apiKey,
    timeout: 48_000,
    maxRetries: 0,
  });
}

function cleanMessages(messages) {
  if (!Array.isArray(messages) || messages.length === 0 || messages.length > MAX_MESSAGES) {
    const error = new Error(`Please send between 1 and ${MAX_MESSAGES} messages. Start a new chat if this conversation is very long.`);
    error.status = 400;
    throw error;
  }
  const cleaned = messages.map((message) => {
    if (!message || !["user", "assistant"].includes(message.role) || typeof message.content !== "string") {
      const error = new Error("A message has an invalid format. Please start a new conversation and try again.");
      error.status = 400;
      throw error;
    }
    return { role: message.role, content: message.content.slice(0, MAX_INPUT_CHARS) };
  });
  const total = cleaned.reduce((sum, message) => sum + message.content.length, 0);
  if (total > MAX_INPUT_CHARS) {
    const error = new Error("This conversation is too long for one request. Start a new chat and try again.");
    error.status = 400;
    throw error;
  }
  if (cleaned.at(-1)?.role !== "user") {
    const error = new Error("The latest message must be from you. Please send your message again.");
    error.status = 400;
    throw error;
  }
  return cleaned;
}

function instructions({ mode, language, effort, codeMode }) {
  const languageName = language === "am" ? "Amharic (አማርኛ)" : "English";
  const depth = {
    standard: "Give a clear, useful answer with enough context and practical examples.",
    high: "Be rigorous: consider assumptions, alternatives, edge cases, and practical validation. Do not reveal private chain-of-thought; give conclusions and a brief rationale.",
    max: "Prioritize correctness and completeness. For complex tasks, organize the deliverable into phases and address edge cases, failure modes, tests, and trade-offs. Do not reveal private chain-of-thought; give a concise rationale and verifiable results.",
  }[effort] || "Give a clear, useful answer with enough context and practical examples.";
  const code = codeMode
    ? "For coding requests, produce complete, directly usable code. For multi-file projects, put every file in its own fenced code block and put the relative filename immediately after the opening backticks, e.g. ```tsx app/page.tsx. Include essential files, styles, configuration, and setup steps. Avoid placeholders and omitted sections. If too large for one response, provide a complete first slice and clearly state what remains."
    : "When code is useful, use fenced code blocks and identify the language. For large code deliverables, organize code into file-specific fenced blocks.";
  const agent = mode === "agent"
    ? "You are Prism AI Agent. In this single request, first plan internally, then produce the requested deliverable, check it for omissions, and include tests or verification steps and known limitations. Do not claim to have run code, accessed websites, modified files, or deployed anything unless a tool actually did so. Do not make a separate planning API call."
    : "You are Prism AI, a capable assistant. Be direct, accurate, friendly, and actionable.";
  return `${agent}\nRespond in ${languageName} unless the user clearly asks for another language.\n${depth}\n${code}\nNever invent tool execution, browsing, test results, or external actions. Distinguish suggestions from completed actions.`;
}

function publicError(error) {
  const status = error?.status || error?.statusCode;
  const message = String(error?.message || "");
  if (status === 401 || status === 403 || /invalid api key|unauthorized|authentication/i.test(message)) {
    return { status: 502, message: "NVIDIA rejected the API key. In Vercel, check NVIDIA_API_KEY, replace it with a valid NVIDIA API key, then redeploy." };
  }
  if (status === 404 || /model.*not found|unknown model|no such model/i.test(message)) {
    return { status: 502, message: `The NVIDIA model "${process.env.NVIDIA_MODEL || DEFAULT_MODEL}" was not found or is unavailable to this key. Check the model ID in NVIDIA API Catalog.` };
  }
  if (status === 429 || /rate limit|too many requests|quota/i.test(message)) {
    return { status: 429, message: "NVIDIA API rate limit or quota reached. Wait a little and try again." };
  }
  if (/timed out|timeout|abort/i.test(message)) {
    return { status: 504, message: "GLM 5.3 Flash took too long to respond. Try Balanced thinking, a shorter prompt, or a smaller output size." };
  }
  return { status: Number.isInteger(status) && status >= 400 && status < 600 ? status : 502, message: "The NVIDIA AI request failed. Please retry. If it continues, check the NVIDIA API key, model access, and NVIDIA service status." };
}

export async function POST(request) {
  try {
    const body = await request.json();
    const messages = cleanMessages(body.messages);
    const mode = body.mode === "agent" ? "agent" : "chat";
    const language = body.language === "am" ? "am" : "en";
    const effort = ["standard", "high", "max"].includes(body.effort) ? body.effort : "high";
    const codeMode = Boolean(body.codeMode);
    const requestedTokens = Number(body.maxOutputTokens);
    // Keep the default responsive on Vercel's 60-second function limit.
    const maxTokens = OUTPUT_LIMITS.has(requestedTokens) ? requestedTokens : 4096;
    const model = process.env.NVIDIA_MODEL || DEFAULT_MODEL;
    const ai = client();
    const upstream = await ai.chat.completions.create({
      model,
      messages: [{ role: "system", content: instructions({ mode, language, effort, codeMode }) }, ...messages],
      stream: true,
      max_tokens: maxTokens,
      temperature: effort === "standard" ? 0.65 : 0.35,
    });
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        try {
          for await (const chunk of upstream) {
            const token = chunk.choices?.[0]?.delta?.content;
            if (token) controller.enqueue(encoder.encode(token));
          }
          controller.close();
        } catch (error) {
          // Fail the reader so the browser shows a useful retry message instead of
          // rendering an upstream exception as if it were a successful AI answer.
          controller.error(error);
        }
      },
      cancel() { upstream.controller?.abort?.(); },
    });
    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch (error) {
    const mapped = publicError(error);
    return Response.json({ error: mapped.message }, { status: mapped.status });
  }
}
