import OpenAI from "openai";

export const runtime = "nodejs";
export const maxDuration = 60;

const MAX_INPUT_CHARS = 60_000;
const MAX_MESSAGES = 60;
const OUTPUT_LIMITS = new Set([1024, 2048, 4096, 8192, 12000, 16000]);
const DEFAULT_MODEL = process.env.NVIDIA_MODEL || "z-ai/glm-5.3-flash";

// NVIDIA-hosted API Catalog model IDs. Access can vary by account and models can
// be retired, so provider errors are returned clearly rather than silently hidden.
export const MODELS = [
  { id: "z-ai/glm-5.3", name: "GLM 5.3", tier: "Free endpoint - powerful, slower", rank: 1, enabled: true },
  { id: "stepfun-ai/step-5-preview", name: "Step 5 Preview", tier: "Free endpoint - experimental", rank: 2, enabled: true },
  { id: "minimaxai/minimax-m3.1-flash", name: "MiniMax M3.1 Flash", tier: "Free endpoint - tested", rank: 3, enabled: true },
  { id: "z-ai/glm-5.3-flash", name: "GLM 5.3 Flash", tier: "Free endpoint - low-reasoning preset", rank: 4, enabled: true },
  { id: "nvidia/nemotron-3.5-super-vl-120b-a12b", name: "Nemotron 3.5 Super VL", tier: "Free endpoint - experimental", rank: 5, enabled: true },
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
    timeout: 45_000,
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

export async function GET() {
  try {
    const catalog = await client().models.list();
    const ids = new Set((catalog.data || []).map((item) => item.id));
    return Response.json({
      provider: "NVIDIA NIM",
      connection: "ok",
      catalogCount: catalog.data.length,
      defaultModel: DEFAULT_MODEL,
      models: MODELS.map((item) => ({ ...item, inCatalog: ids.has(item.id) })),
      agentTools: ["calculator", "current_datetime", "unit_conversion", "word_count"],
    }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    const mapped = publicError(error, DEFAULT_MODEL);
    return Response.json({
      provider: "NVIDIA NIM",
      connection: "error",
      error: mapped.message,
      models: MODELS,
    }, { status: mapped.status, headers: { "Cache-Control": "no-store" } });
  }
}

const AGENT_TOOLS = [
  { type: "function", function: { name: "calculator", description: "Calculate arithmetic with +, -, *, /, %, ^ and parentheses.", parameters: { type: "object", properties: { expression: { type: "string" } }, required: ["expression"], additionalProperties: false } } },
  { type: "function", function: { name: "current_datetime", description: "Get the current date and time in UTC.", parameters: { type: "object", properties: {}, required: [], additionalProperties: false } } },
  { type: "function", function: { name: "unit_conversion", description: "Convert common length, mass, volume, and temperature units.", parameters: { type: "object", properties: { value: { type: "number" }, from: { type: "string" }, to: { type: "string" } }, required: ["value", "from", "to"], additionalProperties: false } } },
  { type: "function", function: { name: "word_count", description: "Count words, characters, and lines in text.", parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"], additionalProperties: false } } },
];

function calculate(expression) {
  if (typeof expression !== "string" || expression.length > 200) throw new Error("Expression is too long.");
  const tokens = expression.match(/\d+(?:\.\d+)?|[()+\-*/%^]/g) || [];
  if (tokens.join("") !== expression.replace(/\s+/g, "")) throw new Error("Unsupported arithmetic characters.");
  let i = 0;
  function atom() {
    const t = tokens[i++];
    if (t === "+" || t === "-") { const n = atom(); return t === "-" ? -n : n; }
    if (t === "(") { const n = sum(); if (tokens[i++] !== ")") throw new Error("Missing closing parenthesis."); return n; }
    if (!/^\d+(?:\.\d+)?$/.test(t || "")) throw new Error("Expected a number.");
    return Number(t);
  }
  function power() { const n = atom(); if (tokens[i] === "^") { i++; return n ** power(); } return n; }
  function product() { let n = power(); while (["*", "/", "%"].includes(tokens[i])) { const op = tokens[i++], rhs = power(); if ((op === "/" || op === "%") && rhs === 0) throw new Error("Division by zero."); n = op === "*" ? n * rhs : op === "/" ? n / rhs : n % rhs; } return n; }
  function sum() { let n = product(); while (tokens[i] === "+" || tokens[i] === "-") { const op = tokens[i++], rhs = product(); n = op === "+" ? n + rhs : n - rhs; } return n; }
  const result = sum();
  if (i !== tokens.length || !Number.isFinite(result)) throw new Error("Invalid arithmetic expression.");
  return result;
}

function runAgentTool(name, args) {
  if (name === "calculator") return { result: calculate(args.expression) };
  if (name === "current_datetime") return { iso: new Date().toISOString(), timezone: "UTC" };
  if (name === "word_count") { const text = String(args.text || ""); return { words: (text.trim().match(/\S+/g) || []).length, characters: text.length, lines: text ? text.split(/\r?\n/).length : 0 }; }
  if (name === "unit_conversion") {
    const value = Number(args.value), from = String(args.from || "").toLowerCase().trim(), to = String(args.to || "").toLowerCase().trim();
    if (!Number.isFinite(value)) throw new Error("Value must be a finite number.");
    const groups = {
      length: { m: 1, meter: 1, meters: 1, km: 1000, cm: .01, mm: .001, mi: 1609.344, mile: 1609.344, ft: .3048, feet: .3048, inch: .0254, in: .0254 },
      mass: { kg: 1, kilogram: 1, g: .001, gram: .001, lb: .45359237, lbs: .45359237, oz: .028349523125 },
      volume: { l: 1, liter: 1, liters: 1, ml: .001, gal: 3.785411784, gallon: 3.785411784 },
    };
    for (const [category, units] of Object.entries(groups)) if (units[from] && units[to]) return { value, from, to, result: value * units[from] / units[to], category };
    const isC = ["c", "°c", "celsius"].includes(from), isF = ["f", "°f", "fahrenheit"].includes(from);
    const toC = ["c", "°c", "celsius"].includes(to), toF = ["f", "°f", "fahrenheit"].includes(to);
    if ((isC || isF) && (toC || toF)) { const c = isF ? (value - 32) * 5 / 9 : value; return { value, from, to, result: toF ? c * 9 / 5 + 32 : c, category: "temperature" }; }
    throw new Error("Unsupported unit pair. Try m/km/mi/ft, kg/g/lb, l/ml/gal, or C/F.");
  }
  throw new Error("Unknown tool.");
}

function skillInstructions(messages, codeMode) {
  const prompt = String(messages.at(-1)?.content || "").toLowerCase(), skills = [];
  if (codeMode || /\b(code|program|debug|bug|repo|website|api|typescript|javascript|python|deploy|build an app)\b/.test(prompt)) skills.push("Engineering skill: provide maintainable complete solutions, edge cases, and tests. Never claim execution without evidence.");
  if (/\b(analy[sz]e|compare|research|evaluate|evidence|source)\b/.test(prompt)) skills.push("Analysis skill: distinguish facts from assumptions, compare options using criteria, and state uncertainty.");
  if (/\b(write|draft|rewrite|email|proposal|story|post)\b/.test(prompt)) skills.push("Writing skill: match audience and purpose, and provide polished usable copy.");
  if (/\b(data|csv|spreadsheet|statistics|dataset|chart|metrics)\b/.test(prompt)) skills.push("Data skill: validate assumptions and calculations; never invent data.");
  return skills.length ? "\n\nActivated skills:\n" + skills.join("\n") : "";
}

function modelOptions() {
  // Keep provider-specific reasoning payloads off by default: catalog models differ
  // in supported parameters, and unsupported options can cause empty output or stalls.
  return {};
}

async function runAgent(ai, model, messages, system, maxTokens, effort) {
  const history = [{ role: "system", content: system + "\nYou can call real tools: calculator, current_datetime, unit_conversion, word_count. Use them when useful; only claim a tool ran if it did." }, ...messages];
  for (let round = 0; round < 3; round++) {
    const response = await ai.chat.completions.create({ model, messages: history, tools: AGENT_TOOLS, tool_choice: "auto", stream: false, max_tokens: Math.min(maxTokens, 4096), ...modelOptions(model, effort) });
    const message = response.choices?.[0]?.message;
    if (!message) throw new Error("The selected model returned an empty agent response.");
    if (!message.tool_calls?.length) return message.content || "";
    history.push({ role: "assistant", content: message.content || null, tool_calls: message.tool_calls });
    for (const call of message.tool_calls) {
      let result;
      try { result = runAgentTool(call.function.name, JSON.parse(call.function.arguments || "{}")); }
      catch (error) { result = { error: String(error?.message || "Tool execution failed.") }; }
      history.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify(result) });
    }
  }
  const final = await ai.chat.completions.create({ model, messages: history, stream: false, max_tokens: maxTokens, ...modelOptions(model, effort) });
  return final.choices?.[0]?.message?.content || "";
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
    const maxTokens = OUTPUT_LIMITS.has(requestedTokens) ? requestedTokens : 4096;
    const requestedModel = typeof body.model === "string" ? body.model : DEFAULT_MODEL;
    if (!MODEL_IDS.has(requestedModel)) {
      return Response.json({ error: "That model is not in the supported Prism AI model list. Refresh the page and choose a listed model." }, { status: 400 });
    }
    model = requestedModel;
    const ai = client();
    const system = instructions({ mode, language, effort, codeMode }) + skillInstructions(messages, codeMode);
    if (mode === "agent") {
      const answer = await runAgent(ai, model, messages, system, maxTokens, effort);
      if (!answer.trim()) throw new Error("The selected model returned an empty agent response.");
      return new Response(answer, { headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-cache, no-transform", "X-Prism-Agent-Tools": "calculator,current_datetime,unit_conversion,word_count" } });
    }

    // Avoid model-specific reasoning parameters: NVIDIA's OpenAI-compatible models
    // do not all support them, and they can consume the whole token budget without
    // producing visible answer text. Stream output as soon as the provider emits it.
    const upstream = await ai.chat.completions.create({
      model,
      messages: [{ role: "system", content: system }, ...messages],
      stream: true,
      max_tokens: maxTokens,
      ...modelOptions(model, effort),
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
