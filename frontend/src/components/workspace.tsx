"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { citationLinks, VerifiedUI } from "@/lib/verified-ui";
import type { VerifiedSource } from "@/lib/verified-ui-contract";

type Stage = { name: string; state: "active" | "complete" };
type FinalEvent = {
  request_id?: string;
  status: "answered" | "needs_clarification" | "insufficient_evidence" | string;
  answer?: string;
  sources?: VerifiedSource[];
  openui?: string;
  metrics?: Record<string, string | number | boolean | null>;
};
type Result = FinalEvent & { question: string; runId: number; genUiValid?: boolean };
type ServiceStatus = { api: string; presentation_status: string };

function validSource(value: unknown): value is VerifiedSource {
  if (!value || typeof value !== "object") return false;
  const source = value as Record<string, unknown>;
  return typeof source.id === "string" && typeof source.title === "string" && typeof source.text === "string" && (source.url === undefined || source.url === null || typeof source.url === "string");
}

function label(value: string) {
  return value.replace(/[_-]+/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function statusLabel(status: string) {
  if (["direct_groq"].includes(status.toLowerCase())) return "Groq configured · OpenUI";
  if (["verified_text_fallback", "text_fallback"].includes(status.toLowerCase())) return "Verified text only";
  if (["ready", "available", "configured"].includes(status.toLowerCase())) return "OpenUI available";
  if (["missing_key", "needs_key", "not_configured", "disabled", "setup_pending", "api_key_missing", "missing_api_key"].includes(status.toLowerCase())) return "Generated layouts unavailable";
  if (status === "unknown") return "OpenUI status unknown";
  return `OpenUI ${label(status).toLowerCase()}`;
}

function safeUrl(value?: string | null) {
  try {
    const url = new URL(value ?? "");
    return ["https:", "http:"].includes(url.protocol) ? url.href : undefined;
  } catch {
    return undefined;
  }
}

async function* readEvents(response: Response) {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("The research service returned an empty response.");
  const decoder = new TextDecoder();
  let buffer = "";
  const parseBlock = (block: string) => {
    let event = "";
    const data: string[] = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
    }
    if (!data.length) return null;
    try { return { event, data: JSON.parse(data.join("\n")) as Record<string, unknown> }; }
    catch { throw new Error("The research service sent an invalid event."); }
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let boundary: number;
      while ((boundary = buffer.search(/\r?\n\r?\n/)) >= 0) {
        const block = buffer.slice(0, boundary);
        const sep = buffer.slice(boundary).match(/^\r?\n\r?\n/)?.[0].length ?? 2;
        buffer = buffer.slice(boundary + sep);
        const parsed = parseBlock(block);
        if (parsed) yield parsed;
      }
      if (done) break;
    }
    if (buffer.trim()) {
      const parsed = parseBlock(buffer);
      if (parsed) yield parsed;
    }
  } finally {
    reader.releaseLock();
  }
}

export default function Workspace() {
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [stages, setStages] = useState<Stage[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus | null>(null);
  const [controller, setController] = useState<AbortController | null>(null);
  const [answerProgress, setAnswerProgress] = useState<{ runId: number; text: string }>({ runId: -1, text: "" });
  const [openUiMessage, setOpenUiMessage] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const runCounter = useRef(0);

  useEffect(() => {
    fetch("/api/status", { cache: "no-store" }).then((response) => response.json()).then(setServiceStatus).catch(() => setServiceStatus({ api: "unavailable", presentation_status: "unknown" }));
  }, []);

  useEffect(() => {
    const endSessionOnClose = () => {
      navigator.sendBeacon("/api/session/end", new Blob([], { type: "application/json" }));
    };
    const resetAfterRestore = (event: PageTransitionEvent) => {
      if (!event.persisted) return;
      runCounter.current += 1;
      setQuestion("");
      setResult(null);
      setStages([]);
      setError("");
      setOpenUiMessage("");
      setAnswerProgress({ runId: -1, text: "" });
    };
    window.addEventListener("pagehide", endSessionOnClose);
    window.addEventListener("pageshow", resetAfterRestore);
    return () => {
      window.removeEventListener("pagehide", endSessionOnClose);
      window.removeEventListener("pageshow", resetAfterRestore);
    };
  }, []);

  useEffect(() => {
    if (!result?.answer || result.status !== "answered") return;
    const runId = result.runId;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const frame = window.requestAnimationFrame(() => setAnswerProgress({ runId, text: result.answer! }));
      return () => window.cancelAnimationFrame(frame);
    }
    let index = 0;
    const step = Math.max(10, Math.ceil(result.answer.length / 80));
    const timer = window.setInterval(() => {
      index = Math.min(result.answer!.length, index + step);
      setAnswerProgress({ runId, text: result.answer!.slice(0, index) });
      if (index === result.answer!.length) window.clearInterval(timer);
    }, 16);
    return () => window.clearInterval(timer);
  }, [result?.runId, result?.answer, result?.status]);

  async function ask(text: string, regenerate = false) {
    const submitted = text;
    if (!submitted.trim() || busy) return;
    const aborter = new AbortController();
    const runId = ++runCounter.current;
    setController(aborter);
    setBusy(true);
    setError("");
    setOpenUiMessage("");
    setResult(null);
    setStages([]);
    setQuestion(submitted);
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        credentials: "include",
        headers: { "content-type": "application/json", accept: "text/event-stream" },
        body: JSON.stringify({ question: submitted, ...(regenerate ? { regenerate: true } : {}) }),
        signal: aborter.signal,
      });
      if (!response.ok) {
        let message = "The research service could not answer right now.";
        try {
          const body = await response.json();
          message = body.message || body.detail || message;
        } catch { /* keep the useful fallback */ }
        throw new Error(message);
      }
      let final: FinalEvent | null = null;
      for await (const event of readEvents(response)) {
        const type = event.event || String(event.data.type || "");
        if (type === "stage" || typeof event.data.stage === "string") {
          const stage = String(event.data.stage || "working");
          setStages((current) => [...current.map((item) => ({ ...item, state: "complete" as const })), { name: stage, state: "active" }]);
        } else if (type === "error") {
          throw new Error(String(event.data.message || "The request failed."));
        } else if (type === "final") {
          final = event.data as FinalEvent;
        }
      }
      if (!final) throw new Error("The request ended before a final answer arrived.");
      if (!["answered", "needs_clarification", "insufficient_evidence"].includes(final.status)) throw new Error("The research service returned an unknown final status.");
      if (final.status === "answered" && (typeof final.answer !== "string" || !Array.isArray(final.sources) || !final.sources.every(validSource))) {
        throw new Error("The research service returned an incomplete verified answer.");
      }
      setStages((current) => current.map((item) => ({ ...item, state: "complete" })));
      const completed: Result = { ...final, sources: Array.isArray(final.sources) && final.sources.every(validSource) ? final.sources : [], question: submitted, runId };
      setResult(completed);

      if (final.status === "answered" && typeof final.openui === "string" && final.openui.trim()) {
        try {
          const validation = await fetch("/api/ui-validate", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ openui: final.openui, sources: final.sources || [] }),
          });
          const checked = await validation.json();
          if (validation.ok && checked.valid) setResult((current) => current ? { ...current, genUiValid: true } : current);
          else setOpenUiMessage("The generated layout did not pass validation. Showing the verified text answer.");
        } catch {
          setOpenUiMessage("Generated layout validation is unavailable. Showing the verified text answer.");
        }
      }
    } catch (caught) {
      if (aborter.signal.aborted) setError("Request stopped.");
      else setError(caught instanceof Error ? caught.message : "Something went wrong.");
    } finally {
      setBusy(false);
      setController(null);
    }
  }

  async function newSession() {
    if (busy) return;
    setError("");
    try {
      const response = await fetch("/api/session/end", { method: "POST", credentials: "include" });
      if (!response.ok) {
        let message = "Could not end this session.";
        try { const body = await response.json(); message = body.message || body.detail || message; } catch { /* keep fallback */ }
        throw new Error(message);
      }
      runCounter.current += 1;
      setQuestion("");
      setResult(null);
      setStages([]);
      setError("");
      setOpenUiMessage("");
      setAnswerProgress({ runId: -1, text: "" });
      inputRef.current?.focus();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not end this session.");
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void ask(question);
  }

  function submitOnEnter(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void ask(question);
    }
  }

  const sources = result?.sources || [];
  const showGeneratedUi = result?.status === "answered" && result.genUiValid && result.openui;

  return (
    <main className="workspace-shell">
      <aside className="sidebar" aria-label="Workspace navigation">
        <a href="#main" className="brand" aria-label="Fieldnote home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>fieldnote</span>
        </a>
        <div className="workspace-switcher">
          <div className="workspace-avatar" aria-hidden="true">T</div>
          <div><strong>IBM TechQA</strong><span>28,482 articles</span></div>
          <span className="switcher-chevron" aria-hidden="true">⌄</span>
        </div>
        <nav className="primary-nav">
          <span className="nav-section">Workspace</span>
          <a className="nav-link active" href="#main" aria-current="page"><span className="nav-glyph search-glyph" aria-hidden="true" />Ask a question</a>
          <a className="nav-link" href="#sources"><span className="nav-glyph docs-glyph" aria-hidden="true" />Sources <span className="nav-count">{sources.length || "—"}</span></a>
        </nav>
        <div className="sidebar-bottom">
          <div className={`service-dot ${serviceStatus?.api === "ready" ? "is-ready" : ""}`} />
          <div><strong>{serviceStatus?.api === "ready" ? "Research API connected" : "Research API offline"}</strong><span>Local processing</span></div>
        </div>
      </aside>

      <section className="main-column" id="main">
        <header className="topbar">
          <div className="breadcrumb"><span>Knowledge base</span><span className="crumb-sep">/</span><strong>Ask</strong></div>
          <div className="topbar-actions">
            <div className={`genui-status ${["ready", "direct_groq", "available"].includes(serviceStatus?.presentation_status || "") ? "is-ready" : ""}`} title={serviceStatus?.presentation_status === "verified_text_fallback" ? "Generated layout unavailable; verified canonical text remains available." : "Presentation status from the local API"}>
              <span className="status-dot" />{serviceStatus ? statusLabel(serviceStatus.presentation_status) : "Checking OpenUI"}
            </div>
            <button type="button" className="new-session-button" disabled={busy} onClick={() => void newSession()}>New session</button>
          </div>
        </header>

        <div className="content-layout">
          <div className="answer-column">
            <div className="page-heading">
              <div>
                <h1>Research workspace</h1>
                <p>Search IBM TechQA articles. Answers appear after verification.</p>
              </div>
              <span className="private-mark"><span aria-hidden="true">⌑</span> Session only</span>
            </div>

            {!result && !busy && !error && (
              <section className="empty-state" aria-label="Start a research question">
                <div className="empty-orbit" aria-hidden="true"><span /><span /><span /></div>
                <h2>What do you need to find?</h2>
                <p>Search your documents for a precise, source-backed answer.</p>
              </section>
            )}

            {busy && (
              <section className="progress-panel" aria-live="polite" aria-busy="true">
                <div className="progress-heading"><span className="progress-spinner" aria-hidden="true" /><div><strong>Working through your sources</strong><span>{stages.at(-1) ? label(stages.at(-1)!.name) : "Starting the research run"}</span></div></div>
                {stages.length > 0 && <ol className="stage-list">{stages.map((stage, index) => <li key={`${stage.name}-${index}`} className={stage.state}><span className="stage-mark">{stage.state === "complete" ? "✓" : "·"}</span>{label(stage.name)}</li>)}</ol>}
                <button className="text-button stop-button" type="button" onClick={() => controller?.abort()}>Stop request</button>
              </section>
            )}

            {error && <div className="error-panel" role="alert"><span className="error-mark" aria-hidden="true">!</span><div><strong>Couldn’t complete the request</strong><p>{error}</p></div></div>}

            {result && (
              <article className="result-view">
                <div className="question-row"><span className="question-mark" aria-hidden="true">Q</span><h2>{result.question}</h2></div>
                {result.status === "answered" ? (
                  <>
                    <div className="answer-meta"><span className="verified-badge" role="status" aria-live="polite"><span aria-hidden="true">✓</span> Verified answer</span>{result.request_id && <span className="request-id">Run {result.request_id.slice(0, 10)}</span>}</div>
                    {showGeneratedUi && <section className="genui-panel" aria-label="Generated answer presentation"><VerifiedUI key={result.runId} response={result.openui!} sources={sources} onError={() => {
                      setResult((current) => current ? { ...current, genUiValid: false } : current);
                      setOpenUiMessage("The generated layout could not render. Showing the verified text answer.");
                    }} /></section>}
                    {!showGeneratedUi && <div className="answer-markdown canonical-answer">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm, citationLinks(sources)]}
                        components={{
                          a: ({ href, children }) => {
                            const url = safeUrl(href);
                            if (url) return <a href={url} target="_blank" rel="noreferrer">{children}</a>;
                            const sourceLink = sources.some((source) => href === `#source-${encodeURIComponent(source.id)}`);
                            return sourceLink ? <a href={href}>{children}</a> : <>{children}</>;
                          },
                          img: () => null,
                        }}
                      >{answerProgress.runId === result.runId ? answerProgress.text : ""}</ReactMarkdown>
                    </div>}
                    {openUiMessage && <p className="presentation-fallback" role="status">{openUiMessage}</p>}
                    <div className="result-actions"><button type="button" className="secondary-button" onClick={() => void ask(result.question, true)} disabled={busy}><span aria-hidden="true">↻</span> Regenerate answer</button><span>Verified against {sources.length} {sources.length === 1 ? "source" : "sources"}</span></div>
                  </>
                ) : (
                  <div className="nonanswer-panel" role="status">
                    <strong>{result.status === "needs_clarification" ? "A detail is needed before I can answer." : "I couldn’t verify an answer from the available sources."}</strong>
                    <p>{result.status === "needs_clarification" ? "Refine your question with the missing system, version, or context and ask again." : "Try a narrower question or check that the relevant documents are in this knowledge base."}</p>
                  </div>
                )}
              </article>
            )}

            <form className="composer" onSubmit={submit}>
              <label className="sr-only" htmlFor="question">Ask a question about your knowledge base</label>
              <textarea ref={inputRef} id="question" value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={submitOnEnter} placeholder="Ask a question about your documents…" rows={2} maxLength={8000} disabled={busy} />
              <div className="composer-footer"><span><kbd>Enter</kbd> to ask <span className="composer-separator">·</span> <kbd>Shift + Enter</kbd> for a new line</span><button className="send-button" type="submit" disabled={busy || !question.trim()} aria-label="Send question">{busy ? <span className="send-loading" aria-hidden="true" /> : <span aria-hidden="true">↑</span>}</button></div>
            </form>
            <p className="privacy-note">Questions and retrieved evidence are sent to Groq for answer generation. Conversation memory lasts for this session.</p>
          </div>

          <aside className="sources-panel" id="sources" aria-label="Sources used in the answer">
            <div className="sources-heading"><div><h2>Sources</h2><p>{result?.status === "answered" ? "Evidence behind this answer" : "Evidence will appear here"}</p></div><span className="source-count">{sources.length}</span></div>
            {sources.length ? (
              <ol className="source-list">{sources.map((source, index) => {
                const href = safeUrl(source.url);
                return <li key={source.id} id={`source-${encodeURIComponent(source.id)}`}><span className="source-index">{String(index + 1).padStart(2, "0")}</span><div className="source-details"><div className="source-title-row">{href ? <a href={href} target="_blank" rel="noreferrer">{source.title}</a> : <strong>{source.title}</strong>}{href && <span className="external-mark" aria-hidden="true">↗</span>}</div><p className="source-location">{source.document_id ? `Document ${source.document_id}` : "Knowledge base excerpt"}{source.chunk_id ? ` · ${source.chunk_id}` : ""}</p><details><summary>View excerpt</summary><blockquote>{source.text}</blockquote></details></div></li>;
              })}</ol>
            ) : <div className="sources-empty"><span aria-hidden="true">⌕</span><p>Sources are attached to verified answers so you can inspect the evidence.</p></div>}
            {result?.metrics && <details className="run-metrics"><summary>Run details</summary><dl>{Object.entries(result.metrics).map(([key, value]) => <div key={key}><dt>{label(key)}</dt><dd>{String(value ?? "—")}</dd></div>)}</dl></details>}
          </aside>
        </div>
      </section>
    </main>
  );
}
