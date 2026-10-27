# Prism AI — Chat, Agent & Code Studio

Prism AI is a login-free, bilingual (English / Amharic) AI workspace built with Next.js and NVIDIA's OpenAI-compatible API. It combines streaming chat, a multi-pass agent workflow, configurable response depth, and a separate editable code workspace for large generated artifacts.

## Features

- **Chat mode** for focused conversations with streaming responses.
- **Agent mode** that runs a planning pass before generating the final answer. This is an LLM workflow, not an autonomous computer or filesystem agent.
- **Thinking depth** setting (Balanced, High, Deep) translated into prompting strategies. These options do not guarantee hidden model reasoning or change a provider's internal reasoning budget.
- **Large output choices** of 4K, 8K, 12K, and 16K requested output tokens. The model/provider may impose lower limits or reject unsupported values.
- **Code Studio** extracts fenced code blocks into a separate workspace, lets you edit code, copy it, download individual files, and export all detected blocks as a JSON project bundle.
- **Browser-local chat history** without an account or database; up to 30 recent conversations are kept in local storage on the current browser.
- **English and Amharic UI**, responsive desktop/mobile layout, starter prompts, stop generation, and code-priority mode.
- NVIDIA credentials are used server-side only.

## Requirements

- Node.js 20.9 or newer
- An NVIDIA API key: https://build.nvidia.com/settings
- A model ID available to your NVIDIA API account

## Local development

```bash
npm install
cp .env.example .env.local
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env.local
```

Edit `.env.local`:

```env
NVIDIA_API_KEY=your_real_nvidia_api_key
NVIDIA_MODEL=deepseek-ai/deepseek-v4.1-flash
```

Confirm the exact model ID is available in your NVIDIA API catalog. Start the app with `npm run dev`, then open http://localhost:3000.

## API test

```bash
curl -i http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello from Prism AI"}],"mode":"chat","language":"en","effort":"high","maxOutputTokens":8192}'
```

The response streams as plain text. Missing keys return HTTP 500, invalid input returns HTTP 400, and provider errors normally return HTTP 502.

## Quality checks

```bash
npm run lint
npm run build
npm start
```

The app's `/api/chat` GET endpoint checks NVIDIA API authentication and model-catalog connectivity. A real chat POST should also be tested after deployment; model access and free-endpoint availability can change over time.

## Deploy to Vercel

1. Push this repository to GitHub.
2. Visit https://vercel.com/new and import `surafel5509-del/Prism-AI`.
3. In **Project Settings → Environment Variables**, add `NVIDIA_API_KEY` and `NVIDIA_MODEL`.
4. Deploy and test both Chat and Agent modes.

Never commit `.env.local` or expose the NVIDIA key with a `NEXT_PUBLIC_` prefix.

## Architecture

```text
app/
  api/chat/route.js  # NVIDIA chat + agent planning pass; streams final output
  ui/chat.js         # Chat UI, local history, settings, artifact extraction and code editor
  globals.css        # Responsive dark workspace design
  layout.js          # Root layout and metadata
  page.js            # Workspace entry point
.env.example
```

## Important limitations and security notes

- No login is required, as requested. Chat history lives only in the current browser and is not synced between devices.
- There is no server-side database, multi-user isolation, billing, rate limiting, abuse prevention, or quota enforcement. Do not expose this publicly at high traffic without suitable controls.
- Agent mode performs a planning model call followed by a final model call. It cannot execute terminal commands, browse the web, edit remote repositories, or autonomously deploy applications.
- Code Studio shows and edits code blocks detected in model output. Export produces a JSON bundle, not a ZIP archive, and the current editor is a plain-text editor without syntax highlighting or live preview.
- Token controls request an output limit; the model may impose a smaller context/output limit. Very large websites may require multiple prompts.
- Long generations can be interrupted by provider or serverless runtime limits. Verify current Vercel function limits before production use.
- This app has not been authenticated against NVIDIA from this repository; test it with a real API key.
