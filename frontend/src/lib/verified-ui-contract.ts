import { createLibrary, createParser, defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";

export type VerifiedSource = { id: string; document_id?: string; chunk_id?: string; title: string; url?: string | null; text: string };
const sourceIds = z.array(z.string().regex(/^S[1-9]\d*$/)).min(1).max(32);

export const uiProps = {
  Stack: z.object({ children: z.array(z.any()).max(64), direction: z.enum(["column", "row"]).optional(), gap: z.enum(["none", "xs", "s", "m", "l", "xl"]).optional() }).strict(),
  AnswerText: z.object({ text: z.string() }).strict(),
  CodeBlock: z.object({ language: z.string(), code: z.string() }).strict(),
  DataTable: z.object({ caption: z.string().optional(), headers: z.array(z.string()).min(1).max(12), rows: z.array(z.array(z.string())).max(50) }).strict(),
  Citation: z.object({ sourceId: z.string() }).strict(),
  EvidenceExcerpt: z.object({ sourceId: z.string() }).strict(),
  AnswerCard: z.object({ title: z.string(), summary: z.string(), sourceIds }).strict(),
  Procedure: z.object({ title: z.string(), steps: z.array(z.object({ title: z.string(), detail: z.string(), code: z.string().optional(), language: z.string().optional(), sourceIds }).strict()).min(1).max(16) }).strict(),
  Configuration: z.object({ title: z.string(), entries: z.array(z.object({ key: z.string(), value: z.string(), description: z.string().optional(), sourceIds }).strict()).min(1).max(20) }).strict(),
  Notice: z.object({ title: z.string(), body: z.string(), kind: z.enum(["note", "warning"]), sourceIds }).strict(),
  Tabs: z.object({ items: z.array(z.object({ label: z.string(), children: z.array(z.any()).min(1).max(32) }).strict()).min(2).max(6) }).strict(),
};

export const uiDescriptions: Record<keyof typeof uiProps, string> = {
  Stack: "Compose answer components vertically, or side by side when space allows.",
  AnswerText: "Short supporting Markdown text. Prefer the task-specific components for the main answer.",
  CodeBlock: "A documented command or code example with a copy button. Display only; never executes code.",
  DataTable: "Compare documented versions or alternatives in a table. Preserve units; never invent values or scores.",
  Citation: "Reference a trusted source by its sourceId. The application supplies its real title and link.",
  EvidenceExcerpt: "Expand an exact source excerpt supplied by the application; never invent or paraphrase it.",
  AnswerCard: "Lead with a concise finding: title, direct answer summary, and sourceIds. Do not invent confidence scores.",
  Procedure: "Turn documented instructions into ordered steps with optional exact commands and sourceIds per step. Users can mark steps reviewed locally; this is not execution or success verification.",
  Configuration: "Display documented file names, setting names and example values as key/value rows with explanations and sourceIds. Never guess the user's current configuration.",
  Notice: "An explicitly supported warning, prerequisite or important note. All factual content must already occur in the approved answer.",
  Tabs: "Separate documented alternatives, platforms or versions into switchable panels. Keep required steps, warnings and prerequisites outside tabs. Each panel label and all hidden content are verified.",
};

export const componentNames = Object.keys(uiProps) as (keyof typeof uiProps)[];
export const verifiedUiSchemaLibrary = createLibrary({
  root: "Stack",
  components: componentNames.map((name) => defineComponent({ name, props: uiProps[name], description: uiDescriptions[name], component: () => null })),
});
const parser = createParser(verifiedUiSchemaLibrary.toJSONSchema(), "Stack");

export function getUiSystemPrompt() {
  return verifiedUiSchemaLibrary.prompt({
    toolCalls: false,
    bindings: false,
    preamble: "Compose an evidence-backed technical answer interface from the approved answer. Choose components to match the user's task. Do not add facts, instructions, configuration values, scores or claims of successful execution. Cite only supplied source IDs.",
    additionalRules: [
      "Return one Stack root. Use only the declared components; no Query, Mutation, state, functions, actions, raw HTML or generated URLs.",
      "Start with AnswerCard for the main finding. For a brief factual question this card plus citations is enough; do not fabricate extra sections.",
      "For troubleshooting or how-to answers, use Procedure rather than burying steps in a single AnswerText. Put exact documented commands in the step's code field or CodeBlock.",
      "For file paths, setting names and documented values use Configuration. For documented comparisons use DataTable; Tabs are useful only for real alternatives, not decorative duplicate summaries.",
      "Use Notice only for actual prerequisites, caveats or warnings present in the approved answer. Never add generic warnings to fill space.",
      "Preserve all meaning and qualifications. Required steps, prerequisites and safety warnings must not be hidden inside tabs or collapsed evidence.",
      "AnswerCard, Procedure steps, Configuration entries and Notice require sourceIds such as [\"S1\"]. Do not invent IDs. Other factual blocks need inline [S1] markers or an adjacent Citation.",
      "Keep titles short and specific. Reorganize content without repeating the whole answer across components. No charts, probabilities, severity scores, timing estimates or invented operational status.",
      "Use plain text in AnswerCard, Procedure, Configuration and Notice fields; their sourceIds provide citations. Reserve Markdown formatting for AnswerText and raw code for code fields.",
    ],
  });
}

export function inspectUi(openui: string, sources: VerifiedSource[]) {
  const parsed = parser.parse(openui);
  const citations = new Set<string>();
  const visible: string[] = [];
  const sourceMap = new Map(sources.map((source) => [source.id, source]));
  const path = new Set<object>();
  let count = 0;
  let valid = Boolean(parsed.root && parsed.root.typeName === "Stack") && !parsed.rootExpression && !parsed.meta.incomplete && parsed.meta.errors.length === 0 && parsed.meta.unresolved.length === 0 && parsed.meta.orphaned.length === 0 && parsed.meta.statementCount <= 128 && parsed.queryStatements.length === 0 && parsed.mutationStatements.length === 0 && Object.keys(parsed.stateDeclarations).length === 0;

  const cite = (id: string) => {
    const source = /^\d+$/.test(id) ? sources[Number(id) - 1] : sourceMap.get(id);
    if (!source) { valid = false; return; }
    citations.add(source.id);
  };
  const text = (value?: string) => {
    if (!value) return;
    visible.push(value);
    for (const match of value.matchAll(/\[(S\d+|\d+)\]/g)) cite(match[1]);
  };
  const ids = (values: string[]) => values.forEach(cite);

  const visit = (node: unknown, depth = 0): void => {
    if (Array.isArray(node)) { node.forEach((child) => visit(child, depth)); return; }
    if (!node || typeof node !== "object" || depth > 24 || ++count > 256 || path.has(node)) { valid = false; return; }
    path.add(node);
    const element = node as { type?: string; typeName?: string; props?: unknown };
    const name = element.typeName as keyof typeof uiProps;
    if (element.type !== "element" || !Object.hasOwn(uiProps, name)) { valid = false; path.delete(node); return; }
    const checked = uiProps[name].safeParse(element.props);
    if (!checked.success) { valid = false; path.delete(node); return; }
    // Parse each known type again to retain precise TypeScript narrowing below.
    switch (name) {
      case "Stack": visit(uiProps.Stack.parse(element.props).children, depth + 1); break;
      case "AnswerText": text(uiProps.AnswerText.parse(element.props).text); break;
      case "CodeBlock": {
        const p = uiProps.CodeBlock.parse(element.props); text(p.language); text(p.code); break;
      }
      case "DataTable": {
        const p = uiProps.DataTable.parse(element.props); text(p.caption); p.headers.forEach(text);
        p.rows.forEach((row) => { if (row.length !== p.headers.length) valid = false; row.forEach(text); }); break;
      }
      case "Citation": case "EvidenceExcerpt": {
        const p = uiProps[name].parse(element.props); cite(p.sourceId);
        const source = sourceMap.get(p.sourceId);
        if (source) { text(source.title); if (name === "EvidenceExcerpt") text(source.text); }
        break;
      }
      case "AnswerCard": {
        const p = uiProps.AnswerCard.parse(element.props); text(p.title); text(p.summary); ids(p.sourceIds); break;
      }
      case "Procedure": {
        const p = uiProps.Procedure.parse(element.props); text(p.title);
        p.steps.forEach((step) => { text(step.title); text(step.detail); text(step.code); text(step.language); ids(step.sourceIds); }); break;
      }
      case "Configuration": {
        const p = uiProps.Configuration.parse(element.props); text(p.title);
        p.entries.forEach((entry) => { text(entry.key); text(entry.value); text(entry.description); ids(entry.sourceIds); }); break;
      }
      case "Notice": {
        const p = uiProps.Notice.parse(element.props); text(p.title); text(p.body); ids(p.sourceIds); break;
      }
      case "Tabs": {
        const p = uiProps.Tabs.parse(element.props);
        p.items.forEach((item) => { text(item.label); visit(item.children, depth + 1); }); break;
      }
    }
    path.delete(node);
  };
  visit(parsed.root);
  const visibleText = visible.join("\n\n");
  if (!visibleText.trim() || visibleText.length > 60000) valid = false;
  return { valid, visible_text: visibleText, citation_source_ids: [...citations] };
}
