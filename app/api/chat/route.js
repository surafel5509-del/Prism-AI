import OpenAI from "openai";

export const runtime = "nodejs";
export const maxDuration = 60;

const MAX_INPUT_CHARS = 60_000;
const MAX_MESSAGES = 60;
const OUTPUT_LIMITS = new Set([1024, 2048, 4096]);
const DEFAULT_MODEL = "deepseek-ai/deepseek-v4.1-flash";

// NVIDIA-hosted API Catalog model IDs. Access can vary by account and models can
// be retired, so provider errors are returned clearly rather than silently hidden.
export const MODELS = [
  { id: "deepseek-ai/deepseek-v4.1-flash", name: "DeepSeek V4.1 Flash", tier: "Free endpoint", enabled: true },
  { id: "minimaxai/minimax-m3.1-flash", name: "MiniMax M3.1 Flash", tier: "Free endpoint", enabled: true },
  { id: "mistralai/mistral-nemotron", name: "Mistral Nemotron", tier: "Free endpoint", enabled: true },
  { id: "openai/gpt-oss-20b", name: "GPT-OSS 20B", tier: "Free endpoint", enabled: true },
  { id: "z-ai/glm-5.3-flash", name: "GLM 5.3 Flash", tier: "Free endpoint (may be retired)", enabled: true },
];
const MODEL_IDS = new Set(MODELS.filter((model) => model.enabled).map((model) => model.id));

function client() {
  const apiKey = process.env.NVIDIA_API_KEY;
  if (!apiKey) {
    const error = new Error("Server setup is incomplete: NVIDIA_API_KEY is missing in Vercel Environment Variables.");
    error.status = 500;
    throw error;
  }
  return new OpenAI({
    baseURL: process.env.NVIDIA_BASE_URL || "https://integrate.api.nvidia.com/v1",
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
    if (!message || !["user", "assistant"].includes(message.role)) {
      const error = new Error("A message has an invalid format. Please start a new conversation and try again.");
      error.status = 400;
      throw error;
    }
    if (typeof message.content === "string") {
      return { role: message.role, content: message.content.slice(0, MAX_INPUT_CHARS) };
    }
    // NVIDIA supports OpenAI-compatible image_url parts in user messages.
    if (message.role === "user" && Array.isArray(message.content)) {
      const parts = message.content.map((part) => {
        if (part?.type === "text" && typeof part.text === "string") {
          return { type: "text", text: part.text.slice(0, MAX_INPUT_CHARS) };
        }
        const imageUrl = part?.type === "image_url" ? part.image_url?.url : null;
        if (typeof imageUrl === "string" && (imageUrl.startsWith("https://") || /^data:image\/(png|jpeg|webp|gif);base64,/i.test(imageUrl))) {
          return { type: "image_url", image_url: { url: imageUrl } };
        }
        const error = new Error("Image or message format is unsupported. Use HTTPS image URLs or PNG, JPEG, WebP, or GIF data URLs.");
        error.status = 400;
        throw error;
      });
      return { role: message.role, content: parts };
    }
    const error = new Error("A message has an invalid format. Please start a new conversation and try again.");
    error.status = 400;
    throw error;
  });
  const total = cleaned.reduce((sum, message) => sum + (typeof message.content === "string" ? message.content.length : JSON.stringify(message.content).length), 0);
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

function publicError(error, model) {
  const status = error?.status || error?.statusCode;
  const message = String(error?.message || "");
  if (status === 401 || status === 403 || /invalid api key|unauthorized|authentication/i.test(message)) {
    return { status: 502, message: "NVIDIA rejected the API key. In Vercel, check NVIDIA_API_KEY, replace it with a valid NVIDIA API key, then redeploy." };
  }
  if (status === 410) {
    return { status: 502, message: `The selected model "${model}" is not available through this NVIDIA hosted endpoint. Choose a model marked available or configure a separate provider.` };
  }
  if (status === 404 || /model.*not found|unknown model|no such model/i.test(message)) {
    return { status: 502, message: `The selected model "${model}" is unavailable to this NVIDIA API key. Choose another model or check its access in NVIDIA API Catalog.` };
  }
  if (status === 429 || /rate limit|too many requests|quota/i.test(message)) {
    return { status: 429, message: "NVIDIA API rate limit or quota reached. Wait a little and try again, or select another available model." };
  }
  if (status === 408 || /timed out|timeout|abort/i.test(message)) {
    return { status: 504, message: "The model took too long to respond. Try a shorter prompt, Balanced thinking, or a smaller output size." };
  }
  return { status: Number.isInteger(status) && status >= 400 && status < 600 ? status : 502, message: "The NVIDIA AI request failed. Please retry. If it continues, check the NVIDIA API key, model access, and NVIDIA service status." };
}

export async function GET(request) {
  try {
    const ai = client();
    const url = new URL(request.url);
    if (url.searchParams.get("probe") === "1") {
      const model = url.searchParams.get("model") || DEFAULT_MODEL;
      if (!MODEL_IDS.has(model)) {
        return Response.json({ connection: "error", error: "Unsupported model." }, { status: 400 });
      }
      const completion = await ai.chat.completions.create({
        model,
        messages: [{ role: "user", content: "Reply with exactly: Prism AI API test passed." }],
        max_tokens: 24,
        stream: false,
        temperature: 0,
      });
      return Response.json({
        connection: "ok",
        model,
        answer: completion.choices?.[0]?.message?.content || "",
      }, { headers: { "Cache-Control": "no-store" } });
    }
    const catalog = await ai.models.list();
    return Response.json({
      provider: "NVIDIA NIM",
      connection: "ok",
      catalogCount: catalog.data.length,
      defaultModel: DEFAULT_MODEL,
      models: MODELS,
    }, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    const mapped = publicError(error, DEFAULT_MODEL);
    return Response.json({
      provider: "NVIDIA NIM",
      connection: "error",
      error: mapped.message,
      models: MODELS,
    }, {
      status: mapped.status,
      headers: { "Cache-Control": "no-store" },
    });
  }
}

export async function POST(request) {
  let model = DEFAULT_MODEL;
  try {
    const body = await request.json();
    const messages = cleanMessages(body.messages);
    const mode = body.mode === "agent" ? "agent" : "chat";
    const language = body.language === "am" ? "am" : "en";
    const effort = ["standard", "high", "max"].includes(body.effort) ? body.effort : "standard";
    const codeMode = Boolean(body.codeMode);
    const requestedTokens = Number(body.maxOutputTokens);
    const maxTokens = OUTPUT_LIMITS.has(requestedTokens) ? requestedTokens : 2048;
    const requestedModel = typeof body.model === "string" ? body.model : DEFAULT_MODEL;
    if (!MODEL_IDS.has(requestedModel)) {
      return Response.json({ error: "That model is not in the supported Prism AI model list. Refresh the page and choose a listed model." }, { status: 400 });
    }
    model = requestedModel;
    const ai = client();

    // Avoid model-specific reasoning parameters: NVIDIA's OpenAI-compatible models
    // do not all support them, and they can consume the whole token budget without
    // producing visible answer text. Stream output as soon as the provider emits it.
    const upstream = await ai.chat.completions.create({
      model,
      messages: [{ role: "system", content: instructions({ mode, language, effort, codeMode }) }, ...messages],
      stream: true,
      max_tokens: maxTokens,
      temperature: model === "deepseek-ai/deepseek-v4.1-flash" ? 1 : effort === "standard" ? 0.6 : 0.3,
      ...(model === "deepseek-ai/deepseek-v4.1-flash" ? { top_p: 0.95 } : {}),
    });

    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        let emittedText = false;
        try {
          for await (const chunk of upstream) {
            const delta = chunk.choices?.[0]?.delta;
            const token = typeof delta?.content === "string" ? delta.content : "";
            if (token) {
              emittedText = true;
              controller.enqueue(encoder.encode(token));
            }
          }
          if (!emittedText) {
            controller.enqueue(encoder.encode("The selected model returned an empty answer. Please retry or choose another model."));
          }
          controller.close();
        } catch (error) {
          const status = error?.status || error?.statusCode;
          const message = status === 401 || status === 403
            ? "NVIDIA rejected the API key. Check NVIDIA_API_KEY in Vercel."
            : status === 429
              ? "NVIDIA API rate limit or quota reached. Please retry later or choose another model."
              : "The NVIDIA model stream failed before completing. Please retry or choose another model.";
          try { controller.enqueue(encoder.encode(`\n\n[Prism AI error: ${message}]`)); } catch {}
          controller.close();
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
    const mapped = publicError(error, model);
    return Response.json({ error: mapped.message }, { status: mapped.status });
  }
}
