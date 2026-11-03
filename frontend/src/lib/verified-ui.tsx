"use client";

import { Component, createContext, useContext, useId, useState, type KeyboardEvent, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { createLibrary, defineComponent, Renderer, useRenderNode, type ComponentRenderProps } from "@openuidev/react-lang";
import { uiDescriptions, uiProps, type VerifiedSource } from "@/lib/verified-ui-contract";

const Sources = createContext<VerifiedSource[]>([]);

function sourceFor(id: string, sources: VerifiedSource[]) {
  return sources.find((source) => source.id === id);
}

function safeUrl(value?: string | null) {
  try {
    const url = new URL(value ?? "");
    return ["https:", "http:"].includes(url.protocol) ? url.href : undefined;
  } catch {
    return undefined;
  }
}

export function citationLinks(sources: VerifiedSource[]) {
  return () => (tree: MarkNode) => {
    const visit = (parent: MarkNode) => {
      if (!["link", "code", "inlineCode"].includes(parent.type ?? "") && parent.children) {
        parent.children = parent.children.flatMap((child) => {
          if (child.type !== "text" || !child.value) { visit(child); return [child]; }
          const nodes: MarkNode[] = [];
          let offset = 0;
          for (const match of child.value.matchAll(/\[(S\d+|\d+)\]/g)) {
            const citation = match[1];
            const source = /^\d+$/.test(citation) ? sources[Number(citation) - 1] : sources.find((item) => item.id === citation);
            if (!source || match.index === undefined) continue;
            if (match.index > offset) nodes.push({ type: "text", value: child.value.slice(offset, match.index) });
            nodes.push({ type: "link", url: `#source-${encodeURIComponent(source.id)}`, children: [{ type: "text", value: match[0] }] });
            offset = match.index + match[0].length;
          }
          if (!offset) return [child];
          if (offset < child.value.length) nodes.push({ type: "text", value: child.value.slice(offset) });
          return nodes;
        });
      }
    };
    visit(tree);
  };
}

type MarkNode = { type?: string; value?: string; url?: string; children?: MarkNode[] };

const Stack = defineComponent({
  name: "Stack",
  description: uiDescriptions.Stack,
  props: uiProps.Stack,
  component: ({ props, renderNode }) => (
    <div className={`gen-stack gen-stack-${props.direction ?? "column"} gen-gap-${props.gap ?? "m"}`}>
      {renderNode(props.children)}
    </div>
  ),
});

const AnswerText = defineComponent({
  name: "AnswerText",
  description: uiDescriptions.AnswerText,
  props: uiProps.AnswerText,
  component: AnswerTextView,
});

function AnswerTextView({ props }: ComponentRenderProps<{ text: string }>) {
  const sources = useContext(Sources);
  return (
    <div className="answer-markdown">
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
          code: ({ className, children, ...rest }) => <code className={className} {...rest}>{children}</code>,
        }}
      >{props.text}</ReactMarkdown>
    </div>
  );
}

const CodeBlock = defineComponent({
  name: "CodeBlock",
  description: uiDescriptions.CodeBlock,
  props: uiProps.CodeBlock,
  component: CodeBlockView,
});

function CopyButton({ value }: { value: string }) {
  const [status, setStatus] = useState<"idle" | "copied" | "error">("idle");
  async function copy() {
    try {
      await navigator.clipboard.writeText(value);
      setStatus("copied");
    } catch {
      setStatus("error");
    }
  }
  return <button className="gen-copy" type="button" onClick={copy} aria-live="polite">{status === "copied" ? "Copied" : status === "error" ? "Copy failed" : "Copy"}</button>;
}

function CodeBlockView({ props }: ComponentRenderProps<{ language: string; code: string }>) {
  return <figure className="gen-code"><figcaption><span>{props.language || "Code"}</span><CopyButton value={props.code} /></figcaption><pre><code>{props.code}</code></pre></figure>;
}

function SourceMarkers({ sourceIds }: { sourceIds: string[] }) {
  const sources = useContext(Sources);
  const ids = [...new Set(sourceIds)].filter((id) => sourceFor(id, sources));
  if (!ids.length) return null;
  return <span className="gen-source-markers" aria-label="Sources">{ids.map((id) => {
    return <a key={id} href={`#source-${encodeURIComponent(id)}`} aria-label={`Source ${id}`}>[{id}]</a>;
  })}</span>;
}

const AnswerCard = defineComponent({
  name: "AnswerCard", description: uiDescriptions.AnswerCard, props: uiProps.AnswerCard,
  component: ({ props }) => <section className="gen-answer-card"><h3>{props.title}</h3><p>{props.summary}</p><SourceMarkers sourceIds={props.sourceIds} /></section>,
});

function ProcedureView({ props }: ComponentRenderProps<{ title: string; steps: { title: string; detail: string; code?: string; language?: string; sourceIds: string[] }[] }>) {
  const [reviewed, setReviewed] = useState<boolean[]>([]);
  const count = props.steps.length;
  const reviewedCount = reviewed.filter(Boolean).length;
  return <section className="gen-procedure"><header><h3>{props.title}</h3><span className="gen-reviewed" aria-live="polite">Reviewed {reviewedCount} of {count}</span></header><ol>{props.steps.map((step, index) => <li key={`${step.title}-${index}`}><div className="gen-step-number" aria-hidden="true">{index + 1}</div><div className="gen-step-content"><div className="gen-step-title"><h4>{step.title}</h4><SourceMarkers sourceIds={step.sourceIds} /></div><p>{step.detail}</p>{step.code && <figure className="gen-code"><figcaption><span>{step.language || "Code"}</span><CopyButton value={step.code} /></figcaption><pre><code>{step.code}</code></pre></figure>}<label className="gen-review"><input type="checkbox" checked={Boolean(reviewed[index])} onChange={(event) => setReviewed((current) => { const next = [...current]; next[index] = event.target.checked; return next; })} />Reviewed</label></div></li>)}</ol></section>;
}

const Procedure = defineComponent({ name: "Procedure", description: uiDescriptions.Procedure, props: uiProps.Procedure, component: ProcedureView });

const Configuration = defineComponent({
  name: "Configuration", description: uiDescriptions.Configuration, props: uiProps.Configuration,
  component: ({ props }) => <section className="gen-configuration"><h3>{props.title}</h3><dl>{props.entries.map((entry, index) => <div className="gen-config-row" key={`${entry.key}-${index}`}><dt>{entry.key}<SourceMarkers sourceIds={entry.sourceIds} />{entry.description && <small>{entry.description}</small>}</dt><dd><code>{entry.value}</code><CopyButton value={entry.value} /></dd></div>)}</dl></section>,
});

const Notice = defineComponent({
  name: "Notice", description: uiDescriptions.Notice, props: uiProps.Notice,
  component: ({ props }) => <aside className={`gen-notice gen-notice-${props.kind}`}><div><h3>{props.title}</h3><p>{props.body}</p><SourceMarkers sourceIds={props.sourceIds} /></div></aside>,
});

function TabsView({ props }: ComponentRenderProps<{ items: { label: string; children: unknown[] }[] }>) {
  const renderNode = useRenderNode();
  const id = useId();
  const [active, setActive] = useState(0);
  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    let next = active;
    if (event.key === "ArrowRight") next = (active + 1) % props.items.length;
    else if (event.key === "ArrowLeft") next = (active - 1 + props.items.length) % props.items.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = props.items.length - 1;
    else return;
    event.preventDefault(); setActive(next);
    event.currentTarget.querySelectorAll<HTMLButtonElement>("[role=tab]")[next]?.focus();
  }
  if (!props.items.length) return null;
  return <div className="gen-tabs"><div className="gen-tab-list" role="tablist" aria-label="Answer sections" onKeyDown={onKeyDown}>{props.items.map((item, index) => <button key={`${item.label}-${index}`} id={`${id}-tab-${index}`} type="button" role="tab" aria-selected={active === index} aria-controls={`${id}-panel-${index}`} tabIndex={active === index ? 0 : -1} onClick={() => setActive(index)}>{item.label}</button>)}</div>{props.items.map((item, index) => <div key={`${item.label}-${index}`} id={`${id}-panel-${index}`} role="tabpanel" aria-labelledby={`${id}-tab-${index}`} hidden={active !== index}><div className="gen-stack">{renderNode(item.children)}</div></div>)}</div>;
}

const Tabs = defineComponent({ name: "Tabs", description: uiDescriptions.Tabs, props: uiProps.Tabs, component: TabsView });

const DataTable = defineComponent({
  name: "DataTable",
  description: uiDescriptions.DataTable,
  props: uiProps.DataTable,
  component: ({ props }) => (
    <div className="gen-table-wrap">
      <table className="gen-table">
        {props.caption && <caption>{props.caption}</caption>}
        <thead><tr>{props.headers.map((header, i) => <th key={i} scope="col">{header}</th>)}</tr></thead>
        <tbody>{props.rows.map((row, i) => <tr key={i}>{props.headers.map((_, j) => <td key={j}>{row[j] ?? "—"}</td>)}</tr>)}</tbody>
      </table>
    </div>
  ),
});

const Citation = defineComponent({
  name: "Citation",
  description: uiDescriptions.Citation,
  props: uiProps.Citation,
  component: CitationView,
});

function CitationView({ props }: ComponentRenderProps<{ sourceId: string }>) {
  const source = sourceFor(props.sourceId, useContext(Sources));
  if (!source) return <span className="source-missing">Source unavailable</span>;
  const url = safeUrl(source.url);
  return <a className="gen-citation" href={url || `#source-${encodeURIComponent(source.id)}`} target={url ? "_blank" : undefined} rel={url ? "noreferrer" : undefined}>{source.title}</a>;
}

const EvidenceExcerpt = defineComponent({
  name: "EvidenceExcerpt",
  description: uiDescriptions.EvidenceExcerpt,
  props: uiProps.EvidenceExcerpt,
  component: EvidenceExcerptView,
});

function EvidenceExcerptView({ props }: ComponentRenderProps<{ sourceId: string }>) {
  const source = sourceFor(props.sourceId, useContext(Sources));
  return source ? <blockquote className="gen-excerpt"><p>{source.text}</p><cite>{source.title}</cite></blockquote> : <p className="source-missing">Source unavailable</p>;
}

export const verifiedUiLibrary = createLibrary({
  id: "rag-verified-answer",
  root: "Stack",
  components: [Stack, AnswerText, CodeBlock, DataTable, Citation, EvidenceExcerpt, AnswerCard, Procedure, Configuration, Notice, Tabs],
});

export function VerifiedUI({ response, sources, onError }: { response: string; sources: VerifiedSource[]; onError: () => void }) {
  return (
    <Sources.Provider value={sources}>
      <RendererBoundary onError={onError}>
        <Renderer response={response} library={verifiedUiLibrary} isStreaming={false} />
      </RendererBoundary>
    </Sources.Provider>
  );
}

class RendererBoundary extends Component<{ children: ReactNode; onError: () => void }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch() {
    this.props.onError();
  }

  render() {
    return this.state.failed
      ? <p className="genui-error" role="status">The generated layout could not render. Showing the verified text answer below.</p>
      : this.props.children;
  }
}

