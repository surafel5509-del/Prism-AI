import OpenAI from "openai";

export const runtime = "nodejs";
export const maxDuration = 60;

const MAX_INPUT_CHARS = 100_000;
const MAX_MESSAGES = 80;
const OUTPUT_LIMITS = new Set([4096, 8192, 12288, 16384]);

function client() {
  const apiKey = process.env.NVIDIA_API_KEY;
  if (!apiKey) throw new Error("NVIDIA_API_KEY is missing. Add it to your server environment variables.");
  return new OpenAI({ baseURL: "https://integrate.api.nvidia.com/v1", apiKey, timeout: 55_000, maxRetries: 1 });
}

function cleanMessages(messages) {
  if (!Array.isArray(messages) || messages.length === 0 || messages.length > MAX_MESSAGES) {
    throw new Error(`messages must contain between 1 and ${MAX_MESSAGES} items.`);
  }
  const cleaned = messages.map((message) => {
    if (!message || !["user", "assistant"].includes(message.role) || typeof message.content !== "string") {
      throw new Error("Each message must have a valid role and text content.");
    }
    return { role: message.role, content: message.content.slice(0, MAX_INPUT_CHARS) };
  });
  const total = cleaned.reduce((sum, message) => sum + message.content.length, 0);
  if (total > MAX_INPUT_CHARS) throw new Error("This conversation is too large. Start a new chat or shorten the context.");
  if (cleaned.at(-1)?.role !== "user") throw new Error("The latest message must be from the user.");
  return cleaned;
}

function instructions({ mode, language, effort, codeMode }) {
  const languageName = language === "am" ? "Amharic (አማርኛ)" : "English";
  const depth = {
    standard: "Give a clear, useful answer with enough context and practical examples.",
    high: "Use a rigorous problem-solving approach: identify assumptions, compare alternatives, handle edge cases, and explain the key reasoning in a concise, user-facing way. Do not reveal private chain-of-thought; provide conclusions and a brief rationale instead.",
    max: "Prioritize completeness and correctness. Break complex work into explicit phases, address edge cases, failure modes, validation, and trade-offs. For coding requests, provide production-oriented structure and thorough implementation details. Do not reveal private chain-of-thought; provide a concise rationale and verifiable results instead.",
  }[effort] || "Give a clear, useful answer with enough context and practical examples.";
  const code = codeMode
    ? "The user is asking for code or a software artifact. Produce complete, coherent, directly usable code. For multi-file projects, put each file in its own fenced block and put the relative filename immediately after the opening backticks, e.g. ```tsx app/page.tsx. Include all essential files, styles, configuration, and setup instructions. Avoid placeholders, ellipses, and 'rest of code here'. If the request is too large for one response, deliver a useful complete first slice and state exactly what remains."
    : "When code is useful, use fenced code blocks and identify the language. For large code deliverables, organize code into file-specific fenced blocks with the path after the opening backticks.";
  const agent = mode === "agent"
    ? "You are operating in Prism Agent mode. Work like a careful senior engineer: (1) restate the goal internally, (2) make a compact execution plan, (3) deliver the actual result rather than only advice, (4) check the result for omissions and consistency, and (5) include tests or verification steps and known limitations. For website/project requests, produce a structured multi-file implementation. Never claim to have run code, accessed websites, modified files, or deployed anything unless a tool actually did so."
    : "You are Prism AI, a capable assistant. Be direct, accurate, friendly, and actionable.";
  return `${agent}\nRespond in ${languageName} unless the user clearly asks for another language.\n${depth}\n${code}\nNever invent tool execution, browsing, test results, or external actions. Clearly distinguish suggestions from completed actions.`;
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
    const maxTokens = OUTPUT_LIMITS.has(requestedTokens) ? requestedTokens : 8192;
    const model = process.env.NVIDIA_MODEL || "z-ai/glm-5.3-flash";
    const ai = client();
    const system = instructions({ mode, language, effort, codeMode });
    let finalMessages = [{ role: "system", content: system }, ...messages];

    // Agent mode uses a separate planning pass before generating the deliverable.
    // This is a multi-pass LLM workflow, not a claim of autonomous filesystem access.
    if (mode === "agent") {
      const planning = await ai.chat.completions.create({
        model,
        messages: [
          { role: "system", content: "Create a compact execution plan for the user's latest request. Identify the deliverable, major components, risks, and validation checks. Be concise; do not answer the user yet." },
          ...messages,
        ],
        max_tokens: 1200,
        temperature: 0.2,
      });
      const plan = planning.choices?.[0]?.message?.content;
      if (plan) finalMessages = [
        { role: "system", content: `${system}\nInternal work plan to follow while preparing your answer (do not expose hidden reasoning):\n${plan}` },
        ...messages,
      ];
    }

    const upstream = await ai.chat.completions.create({
      model,
      messages: finalMessages,
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
          controller.enqueue(encoder.encode(`\n\n[Stream interrupted: ${error?.message || "upstream error"}]`));
          controller.close();
        }
      },
      cancel() { upstream.controller?.abort?.(); },
    });
    return new Response(stream, { headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "X-Content-Type-Options": "nosniff",
    } });
  } catch (error) {
    const message = error?.message || "Unexpected server error.";
    const status = message.includes("NVIDIA_API_KEY") ? 500 : /messages|conversation is too large|latest message|valid role|between 1 and/.test(message) ? 400 : 502;
    return Response.json({ error: message }, { status });
  }
}
