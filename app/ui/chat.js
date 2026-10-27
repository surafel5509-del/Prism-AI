"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const TEXT = {
  en: {
    newChat: "New conversation", history: "WORKSPACE", chats: "Recent chats", welcome: "Make something remarkable.",
    intro: "Explore ideas, solve hard problems, or build a complete website. Prism adapts to the way you work.",
    placeholder: "Message Prism…", send: "Send message", stop: "Stop generating", chat: "Chat", agent: "Agent mode",
    standard: "Balanced", high: "High thinking", max: "Deep thinking", output: "Output size", code: "Code Studio",
    noCode: "Your code workspace is ready", noCodeDesc: "Ask Prism to build a website, app, script, or full project. Generated code will appear here, separate from the conversation.",
    copy: "Copy code", download: "Download file", downloadAll: "Export project", copied: "Copied", file: "File", lines: "lines", chars: "characters",
    welcomeTag: "YOUR IDEAS, AMPLIFIED", prompt1: "Build a complete modern website", prompt2: "Explain a complex topic deeply", prompt3: "Plan and implement an app", prompt4: "Review and improve my code",
    agentHint: "Agent plans first, then creates a detailed answer", chatHint: "Fast, focused conversation", codeMode: "Prioritize code", language: "Language", model: "AI model", settings: "Generation", emptyHistory: "Your chats stay in this browser.",
    error: "Something went wrong. Check the API key and model configuration, then try again.",
    stopNote: "Generation stopped.", ready: "Ready", working: "Working…", disclaimer: "AI can make mistakes. Review important information and generated code before using it.",
    menu: "Workspace", artifact: "Generated artifacts", noArtifacts: "No code artifacts yet", artifactsFound: "code blocks", editHint: "Edit code directly in this workspace. Your edits are local until you download them.",
    saveHint: "No account required · Conversations are saved in this browser",
  },
  am: {
    newChat: "አዲስ ውይይት", history: "የሥራ ቦታ", chats: "የቅርብ ውይይቶች", welcome: "ታላቅ ነገር ይፍጠሩ።",
    intro: "ሀሳቦችን ያስፋፉ፣ ከባድ ችግሮችን ይፍቱ ወይም ሙሉ ድረ-ገጽ ይገንቡ። Prism ከሥራ መንገድዎ ጋር ይስማማል።",
    placeholder: "Prismን ይጠይቁ…", send: "መልዕክት ላክ", stop: "ማመንጨት አቁም", chat: "ውይይት", agent: "Agent ሁነታ",
    standard: "መደበኛ", high: "ከፍተኛ አስተሳሰብ", max: "ጥልቅ አስተሳሰብ", output: "የውጤት መጠን", code: "የኮድ ስቱዲዮ",
    noCode: "የኮድ ቦታዎ ዝግጁ ነው", noCodeDesc: "Prismን ድረ-ገጽ፣ መተግበሪያ፣ ስክሪፕት ወይም ሙሉ ፕሮጀክት እንዲገነባ ይጠይቁ። ኮዱ ከውይይቱ ተለይቶ እዚህ ይታያል።",
    copy: "ኮድ ቅዳ", download: "ፋይል አውርድ", downloadAll: "ፕሮጀክት አውጣ", copied: "ተቀድቷል", file: "ፋይል", lines: "መስመሮች", chars: "ቁምፊዎች",
    welcomeTag: "ሀሳብዎን ወደ ትልቅ ውጤት", prompt1: "ዘመናዊ ድረ-ገጽ ገንባ", prompt2: "ከባድ ርዕስን በጥልቀት አስረዳ", prompt3: "መተግበሪያ እቅድ አውጣና ገንባ", prompt4: "ኮዴን ገምግምና አሻሽል",
    agentHint: "Agent መጀመሪያ እቅድ ያወጣል፣ ከዚያ ዝርዝር ውጤት ይሰጣል", chatHint: "ፈጣንና ቀጥተኛ ውይይት", codeMode: "ኮድን ቅድሚያ ስጥ", language: "ቋንቋ", model: "የAI ሞዴል", settings: "ማመንጨት", emptyHistory: "ውይይቶችዎ በዚህ አሳሽ ውስጥ ይቀመጣሉ።",
    error: "ችግር ተፈጥሯል። API ቁልፉንና የሞዴል ቅንብሩን ያረጋግጡና እንደገና ይሞክሩ።", stopNote: "ማመንጨት ቆሟል።", ready: "ዝግጁ", working: "በሥራ ላይ…", disclaimer: "AI ስህተት ሊሠራ ይችላል። አስፈላጊ መረጃንና ኮድን ከመጠቀምዎ በፊት ያረጋግጡ።",
    menu: "የሥራ ቦታ", artifact: "የተፈጠሩ ኮዶች", noArtifacts: "ገና ኮድ አልተፈጠረም", artifactsFound: "የኮድ ብሎኮች", editHint: "ኮዱን እዚህ በቀጥታ ያርትዑ። እስኪያወርዱት ድረስ ለእርስዎ ብቻ ነው።", saveHint: "መለያ አያስፈልግም · ውይይቶች በዚህ አሳሽ ይቀመጣሉ",
  },
};

const MODELS = [
  { id: "deepseek-ai/deepseek-v4.1-flash", name: "DeepSeek V4.1 Flash" },
  { id: "mistralai/mistral-small-4-119b-2603", name: "Mistral Small 4" },
  { id: "minimaxai/minimax-m3.1-flash", name: "MiniMax M3.1 Flash" },
  { id: "z-ai/glm-5.3-flash", name: "GLM 5.3 Flash" },
  { id: "openai/gpt-oss-20b", name: "GPT OSS 20B" },
  { id: "google/gemma-4-31b-it", name: "Gemma 4 31B" },
  { id: "nvidia/nemotron-3.5-lightning-30b-a3b", name: "Nemotron 3.5 Lightning" },
];

const STARTERS = [
  "Build a polished multi-page SaaS website with HTML, CSS, and JavaScript",
  "Create a complete React dashboard with responsive layout",
  "Explain an advanced topic step by step with examples",
  "Design and implement a small business website",
];

function Icon({ name, size = 18 }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true };
  const paths = {
    spark: <><path d="m12 3 1.9 5.8L20 11l-6.1 2.2L12 19l-2-5.8L4 11l6-2.2L12 3Z"/><path d="m19 14 1.1 2.9L23 18l-2.9 1.1L19 22l-1.1-2.9L15 18l2.9-1.1L19 14Z"/></>,
    plus: <path d="M12 5v14M5 12h14"/>,
    message: <><path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8A8.5 8.5 0 0 1 8.7 3.9a8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8Z"/></>,
    code: <><path d="m8 17-5-5 5-5M16 7l5 5-5 5M14 4l-4 16"/></>,
    send: <><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></>,
    copy: <><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></>,
    download: <><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5M12 15V3"/></>,
    panel: <><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/></>,
    menu: <path d="M4 6h16M4 12h16M4 18h16"/>,
    close: <path d="m18 6-12 12M6 6l12 12"/>,
    stop: <rect x="6" y="6" width="12" height="12" rx="2"/>,
    agent: <><circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/></>,
  };
  return <svg {...common}>{paths[name] || paths.spark}</svg>;
}

function slugify(value) {
  return value.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 70) || "generated-code";
}

function extractArtifacts(text) {
  const found = [];
  const fence = String.fromCharCode(96).repeat(3);
  const regex = new RegExp(fence + "([^\\n]*)\\n([\\s\\S]*?)(?:" + fence + "|$)", "g");
  let match;
  while ((match = regex.exec(text))) {
    const label = match[1].trim();
    const content = match[2].replace(/\n$/, "");
    if (!content.trim()) continue;
    const bits = label.split(/\s+/);
    let filename = bits.find((bit) => /\.[a-z0-9]{1,8}$/i.test(bit) || bit.includes("/"));
    const language = bits[0] || "text";
    if (!filename) {
      const ext = ({ html: "html", css: "css", javascript: "js", js: "js", jsx: "jsx", tsx: "tsx", typescript: "ts", ts: "ts", json: "json", python: "py", py: "py", bash: "sh", sql: "sql", markdown: "md", md: "md", yaml: "yml", xml: "xml", svg: "svg" })[language.toLowerCase()] || "txt";
      filename = found.length === 0 && ["html", "css", "js", "jsx", "tsx", "ts", "json", "py", "sql", "md", "svg"].includes(ext) ? "index." + ext : "file-" + (found.length + 1) + "." + ext;
    }
    found.push({ id: found.length + "-" + slugify(filename), filename, language, content });
  }
  return found;
}


function displayMessage(text) {
  const fence = String.fromCharCode(96).repeat(3);
  const regex = new RegExp(fence + "[^\\n]*\\n[\\s\\S]*?(?:" + fence + "|$)", "g");
  return text.replace(regex, "[Generated code is available in Code Studio]");
}

function saveDownload(filename, content, type = "text/plain;charset=utf-8") {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = filename; link.click();
  URL.revokeObjectURL(url);
}

export default function PrismWorkspace() {
  const [lang, setLang] = useState("en");
  const [mode, setMode] = useState("chat");
  const [effort, setEffort] = useState("standard");
  const [model, setModel] = useState("deepseek-ai/deepseek-v4.1-flash");
  const [maxOutputTokens, setMaxOutputTokens] = useState(4096);
  const [codeMode, setCodeMode] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [codeOpen, setCodeOpen] = useState(true);
  const [selectedArtifact, setSelectedArtifact] = useState(0);
  const [artifacts, setArtifacts] = useState([]);
  const [copied, setCopied] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [savedChats, setSavedChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const t = TEXT[lang];
  const activeArtifact = artifacts[selectedArtifact] || null;
  const lines = activeArtifact ? activeArtifact.content.split("\n").length : 0;

  useEffect(() => {
    try {
      const stored = JSON.parse(localStorage.getItem("prism-ai-chats") || "[]");
      if (Array.isArray(stored)) setSavedChats(stored.slice(0, 30));
      const preferred = localStorage.getItem("prism-ai-language");
      if (preferred === "am" || preferred === "en") setLang(preferred);
    } catch { /* Local history is optional. */ }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try { localStorage.setItem("prism-ai-chats", JSON.stringify(savedChats.slice(0, 30))); } catch { /* Storage may be full or disabled. */ }
  }, [savedChats, hydrated]);

  useEffect(() => { if (hydrated) localStorage.setItem("prism-ai-language", lang); }, [lang, hydrated]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages, busy]);
  useEffect(() => () => abortRef.current?.abort(), []);

  const persistConversation = useCallback((nextMessages, id = activeChatId) => {
    if (!nextMessages.length) return;
    const firstUser = nextMessages.find((m) => m.role === "user")?.content || "New conversation";
    const chat = { id: id || `chat-${Date.now()}`, title: firstUser.slice(0, 64), updatedAt: Date.now(), messages: nextMessages };
    setActiveChatId(chat.id);
    setSavedChats((previous) => [chat, ...previous.filter((item) => item.id !== chat.id)].slice(0, 30));
  }, [activeChatId]);

  function newChat() {
    abortRef.current?.abort();
    setMessages([]); setInput(""); setError(""); setBusy(false); setArtifacts([]); setSelectedArtifact(0); setActiveChatId(null); setSidebarOpen(false);
    inputRef.current?.focus();
  }

  function openChat(chat) {
    abortRef.current?.abort();
    setMessages(chat.messages || []); setActiveChatId(chat.id); setInput(""); setError(""); setBusy(false); setSidebarOpen(false);
    const nextArtifacts = (chat.messages || []).filter((m) => m.role === "assistant").flatMap((m) => extractArtifacts(m.content || ""));
    setArtifacts(nextArtifacts); setSelectedArtifact(0);
  }

  async function sendMessage(value = input) {
    const text = value.trim();
    if (!text || busy) return;
    const updated = [...messages, { role: "user", content: text }];
    const placeholder = [...updated, { role: "assistant", content: "" }];
    setMessages(placeholder); setInput(""); setError(""); setBusy(true); setSidebarOpen(false);
    const controller = new AbortController(); abortRef.current = controller;
    try {
      const response = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: updated, mode, language: lang, effort, maxOutputTokens, codeMode, model }), signal: controller.signal,
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || t.error);
      }
      if (!response.body) throw new Error("Streaming is not supported by this browser.");
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let answer = "";
      while (true) {
        const { value: chunk, done } = await reader.read();
        if (done) break;
        answer += decoder.decode(chunk, { stream: true });
        setMessages([...updated, { role: "assistant", content: answer }]);
        const nextArtifacts = extractArtifacts(answer);
        if (nextArtifacts.length) { setArtifacts(nextArtifacts); setSelectedArtifact(nextArtifacts.length - 1); setCodeOpen(true); }
      }
      answer += decoder.decode();
      const streamError = answer.match(/\[Prism AI error: ([^\]]+)\]/);
      if (streamError) throw new Error(streamError[1]);
      const completed = [...updated, { role: "assistant", content: answer || t.error }];
      setMessages(completed); persistConversation(completed);
      const finalArtifacts = extractArtifacts(answer);
      if (finalArtifacts.length) { setArtifacts(finalArtifacts); setSelectedArtifact(0); setCodeOpen(true); }
    } catch (e) {
      if (e.name !== "AbortError") {
        setError(e.message || t.error);
        setMessages((prev) => prev.at(-1)?.role === "assistant" && (!prev.at(-1)?.content || prev.at(-1)?.content.includes("[Prism AI error:")) ? prev.slice(0, -1) : prev);
      } else {
        setMessages((previous) => previous.map((m, i) => i === previous.length - 1 && m.role === "assistant" ? { ...m, content: `${m.content}${m.content ? "\n\n" : ""}${t.stopNote}` } : m));
      }
    } finally { setBusy(false); abortRef.current = null; inputRef.current?.focus(); }
  }

  async function copyText(value) {
    try { await navigator.clipboard.writeText(value); setCopied(true); window.setTimeout(() => setCopied(false), 1400); }
    catch { setError("Clipboard access was blocked by your browser."); }
  }

  function updateActiveArtifact(value) {
    setArtifacts((previous) => previous.map((item, index) => index === selectedArtifact ? { ...item, content: value } : item));
  }

  const visibleChats = useMemo(() => savedChats.slice(0, 12), [savedChats]);

  return (
    <main className={`prism-shell ${codeOpen ? "with-code" : "without-code"}`}>
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand-row"><div className="brand-mark"><Icon name="spark" size={21} /></div><span className="brand-name">prism<span>ai</span></span><span className="brand-pill">BETA</span><button className="icon-button mobile-close" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar"><Icon name="close" /></button></div>
        <button className="new-chat-button" onClick={newChat}><Icon name="plus" size={17} /><span>{t.newChat}</span><kbd>⌘ K</kbd></button>
        <div className="sidebar-section-label">{t.chats}</div>
        <div className="history-list">{visibleChats.length ? visibleChats.map((chat) => <button key={chat.id} className={`history-item ${activeChatId === chat.id ? "history-active" : ""}`} onClick={() => openChat(chat)}><Icon name="message" size={15} /><span>{chat.title}</span></button>) : <div className="empty-history">{t.emptyHistory}</div>}</div>
        <div className="sidebar-spacer" />
        <div className="sidebar-bottom"><div className="provider-dot" /><div className="provider-copy"><div className="provider-title">{t.ready}</div><div className="provider-subtitle">NVIDIA API · OpenAI-compatible</div></div><span className="status-dot" /></div>
        <div className="sidebar-foot">PRISM AI WORKSPACE <span>·</span> 2.0</div>
      </aside>
      {sidebarOpen && <button className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar" />}

      <section className="main-panel">
        <header className="topbar">
          <div className="topbar-left"><button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar"><Icon name="menu" /></button><div className="topbar-title"><span className="topbar-heading">{t.menu}</span><span className="topbar-subheading">{mode === "agent" ? t.agent : t.chat}</span></div></div>
          <div className="topbar-actions"><div className="mode-switch" role="group" aria-label="Assistant mode"><button className={mode === "chat" ? "mode-active" : ""} onClick={() => setMode("chat")}><Icon name="message" size={14}/>{t.chat}</button><button className={mode === "agent" ? "mode-active agent-active" : ""} onClick={() => setMode("agent")}><Icon name="agent" size={14}/>{t.agent}</button></div><label className="language-control"><span className="sr-only">{t.language}</span><select value={lang} onChange={(e) => setLang(e.target.value)} aria-label={t.language}><option value="en">EN</option><option value="am">አማ</option></select></label><button className={`icon-button workspace-toggle ${codeOpen ? "toggle-on" : ""}`} onClick={() => setCodeOpen((v) => !v)} title={t.code} aria-label={t.code}><Icon name="panel" size={18}/></button></div>
        </header>

        <div className={`conversation ${messages.length === 0 ? "conversation-empty" : ""}`}>
          {messages.length === 0 ? <div className="welcome-panel">
            <div className="welcome-orb"><div className="orb-core"><Icon name="spark" size={30}/></div><span className="orb-ring orb-ring-one"/><span className="orb-ring orb-ring-two"/></div>
            <div className="eyebrow"><span className="eyebrow-line"/>{t.welcomeTag}<span className="eyebrow-line"/></div>
            <h1>{t.welcome}</h1><p className="welcome-copy">{t.intro}</p>
            <div className="starter-grid">{(lang === "en" ? STARTERS : [t.prompt1, t.prompt2, t.prompt3, t.prompt4]).map((starter, i) => <button className="starter-card" key={starter} onClick={() => sendMessage(starter)} disabled={busy}><span className="starter-icon">{[<Icon key="code" name="code" size={17}/>, <Icon key="spark" name="spark" size={17}/>, <Icon key="agent" name="agent" size={17}/>, <Icon key="message" name="message" size={17}/>][i]}</span><span className="starter-text">{starter}</span><span className="starter-arrow">↗</span></button>)}</div>
            <div className="feature-strip"><span><i/> {t.chat}</span><span><i/> {t.agent}</span><span><i/> {t.code}</span></div>
          </div> : <div className="message-list">{messages.map((message, index) => <article className={`message-row ${message.role === "user" ? "user-row" : "assistant-row"}`} key={`${index}-${message.role}`}>
            {message.role === "assistant" && <div className="assistant-avatar"><Icon name="spark" size={16}/></div>}
            <div className="message-content"><div className="message-meta">{message.role === "user" ? (lang === "am" ? "እርስዎ" : "You") : <><span>Prism AI</span>{mode === "agent" && <span className="agent-tag">AGENT</span>}</>}</div>
              {message.role === "user" ? <div className="user-bubble"><div className="message-text">{message.content}</div></div> : <><div className="assistant-text">{message.content ? displayMessage(message.content).split("\n").map((line, i) => <span key={i}>{line || "\u00a0"}{i < displayMessage(message.content).split("\n").length - 1 && <br/>}</span>) : busy ? <div className="thinking"><span/><span/><span/><em>{t.working}</em></div> : null}</div>{message.content && <div className="message-tools"><button onClick={() => copyText(message.content)}><Icon name="copy" size={13}/>{copied ? t.copied : t.copy}</button>{extractArtifacts(message.content).length > 0 && <button onClick={() => setCodeOpen(true)}><Icon name="code" size={13}/>{t.code}</button>}</div>}</>}
            </div>
          </article>)}<div ref={bottomRef}/></div>}
          {error && <div className="error-banner"><span>!</span><div>{error}</div><button onClick={() => setError("")} aria-label="Dismiss error"><Icon name="close" size={14}/></button></div>}
        </div>

        <div className="composer-area"><div className="generation-toolbar"><div className="setting-group"><span className="setting-label">{t.model}</span><select className="model-select" value={model} onChange={(e) => setModel(e.target.value)} aria-label={t.model}>{MODELS.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select><span className="setting-label">{t.settings}</span><select value={effort} onChange={(e) => setEffort(e.target.value)} aria-label="Thinking level"><option value="standard">{t.standard}</option><option value="high">{t.high}</option><option value="max">{t.max}</option></select><select value={maxOutputTokens} onChange={(e) => setMaxOutputTokens(Number(e.target.value))} aria-label={t.output}><option value={4096}>4K {t.output}</option><option value={8192}>8K {t.output}</option><option value={12288}>12K {t.output}</option><option value={16384}>16K {t.output}</option></select></div><label className={`code-mode-toggle ${codeMode ? "code-mode-on" : ""}`}><input type="checkbox" checked={codeMode} onChange={(e) => setCodeMode(e.target.checked)}/><Icon name="code" size={14}/><span>{t.codeMode}</span></label></div>
          <form className="composer" onSubmit={(e) => { e.preventDefault(); sendMessage(); }}><textarea ref={inputRef} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }} placeholder={t.placeholder} rows={3} maxLength={50000} aria-label={t.placeholder}/><div className="composer-bottom"><span className="composer-hint">{mode === "agent" ? t.agentHint : t.chatHint} <span className="hint-divider">·</span> Shift + Enter</span><div className="composer-actions">{busy && <button type="button" className="stop-button" onClick={() => abortRef.current?.abort()} aria-label={t.stop}><Icon name="stop" size={14}/>{t.stop}</button>}<button className="send-button" type="submit" disabled={!input.trim() || busy} aria-label={t.send}><Icon name="send" size={17}/></button></div></div></form><div className="footer-note">{t.disclaimer}</div>
        </div>
      </section>

      {codeOpen && <aside className="code-panel"><div className="code-panel-header"><div className="code-title-wrap"><div className="code-title-icon"><Icon name="code" size={16}/></div><div><div className="code-panel-title">{t.code}</div><div className="code-panel-subtitle">{t.artifact}</div></div></div><div className="code-panel-actions"><button className="icon-button" onClick={() => setCodeOpen(false)} aria-label="Close code workspace" title="Close workspace"><Icon name="close" size={17}/></button></div></div>
        {artifacts.length > 0 ? <><div className="artifact-tabs"><span className="artifact-label">{t.file}</span><select value={selectedArtifact} onChange={(e) => setSelectedArtifact(Number(e.target.value))} aria-label={t.file}>{artifacts.map((artifact, index) => <option key={artifact.id} value={index}>{artifact.filename}</option>)}</select><span className="artifact-count">{artifacts.length} {t.artifactsFound}</span></div><div className="editor-toolbar"><div className="editor-file"><span className="file-dot"/><span>{activeArtifact?.filename}</span><span className="language-badge">{activeArtifact?.language}</span></div><div className="editor-actions"><button onClick={() => copyText(activeArtifact?.content || "")}><Icon name="copy" size={13}/>{copied ? t.copied : t.copy}</button><button onClick={() => activeArtifact && saveDownload(activeArtifact.filename.replaceAll("/", "-"), activeArtifact.content)}><Icon name="download" size={13}/>{t.download}</button></div></div><div className="editor-wrap"><div className="line-numbers" aria-hidden="true">{Array.from({ length: Math.max(lines, 1) }, (_, i) => <span key={i}>{i + 1}</span>)}</div><textarea className="code-editor" spellCheck="false" value={activeArtifact?.content || ""} onChange={(e) => updateActiveArtifact(e.target.value)} aria-label="Editable code workspace"/></div><div className="editor-status"><span><i/> {t.ready}</span><span>{lines} {t.lines}</span><span>{(activeArtifact?.content || "").length.toLocaleString()} {t.chars}</span></div><div className="project-export"><div><div className="export-title">{t.artifact}</div><div className="export-subtitle">{t.editHint}</div></div><button onClick={() => saveDownload("prism-ai-project.json", JSON.stringify({ name: "Prism AI generated project", files: artifacts.map(({ filename, language, content }) => ({ filename, language, content })) }, null, 2), "application/json;charset=utf-8")}><Icon name="download" size={14}/>{t.downloadAll}</button></div></> : <div className="code-empty"><div className="empty-code-illustration"><div className="mini-window"><div className="mini-window-dots"><i/><i/><i/></div><div className="mini-code-lines"><span/><span/><span/><span/><span/></div></div><div className="floating-code-mark"><Icon name="code" size={19}/></div></div><h3>{t.noCode}</h3><p>{t.noCodeDesc}</p><div className="empty-code-tip"><Icon name="spark" size={15}/><span>{t.editHint}</span></div></div>}
        <div className="code-panel-footer"><span><span className="footer-green-dot"/> NVIDIA model · Prism AI</span><span>{t.saveHint}</span></div>
      </aside>}
    </main>
  );
}
