import { useState, ReactNode } from "react";

// Minimal markdown renderer. Avoids pulling in react-markdown +
// rehype-* (~50 kB gzipped) for the small subset of markdown the
// agent actually emits: code fences with optional lang, inline
// `code`, **bold**, _italic_/*italic*, [link](url), blockquotes,
// hr (---), bullet/ordered lists, and headings (# / ## / ###).
//
// What it intentionally skips:
//   • HTML embedding (agent output is plain markdown)
//   • Tables (the agent rarely emits them; if it does, the raw
//     pipe-separated layout still reads fine in monospace)
//   • Nested lists past depth 2 (not worth the parser complexity
//     for our use)
//
// The reader-friendly anchor is the code-fence block: every fenced
// block gets a copy button, a language pill, and proper monospace
// styling — that was the biggest visual gap in the old plain-text
// rendering.

type Token =
  | { kind: "code"; lang: string; body: string }
  | { kind: "para"; body: string }
  | { kind: "h1" | "h2" | "h3"; body: string }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] }
  | { kind: "quote"; body: string }
  | { kind: "hr" }
  | { kind: "blank" };

function tokenize(src: string): Token[] {
  const tokens: Token[] = [];
  const lines = src.replace(/\r\n/g, "\n").split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    // Fenced code block.
    const fence = /^```(\s*\w*)\s*$/.exec(line);
    if (fence) {
      const lang = fence[1].trim();
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) {
        body.push(lines[i]);
        i++;
      }
      // Skip the closing fence (or EOF).
      if (i < lines.length) i++;
      tokens.push({ kind: "code", lang, body: body.join("\n") });
      continue;
    }
    // Horizontal rule.
    if (/^---+\s*$/.test(line)) {
      tokens.push({ kind: "hr" });
      i++;
      continue;
    }
    // Headings.
    const h1 = /^#\s+(.+)$/.exec(line);
    if (h1) { tokens.push({ kind: "h1", body: h1[1] }); i++; continue; }
    const h2 = /^##\s+(.+)$/.exec(line);
    if (h2) { tokens.push({ kind: "h2", body: h2[1] }); i++; continue; }
    const h3 = /^###\s+(.+)$/.exec(line);
    if (h3) { tokens.push({ kind: "h3", body: h3[1] }); i++; continue; }
    // Blockquote.
    if (/^>\s?/.test(line)) {
      const body: string[] = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) {
        body.push(lines[i].replace(/^>\s?/, ""));
        i++;
      }
      tokens.push({ kind: "quote", body: body.join("\n") });
      continue;
    }
    // Bullet list.
    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ""));
        i++;
      }
      tokens.push({ kind: "ul", items });
      continue;
    }
    // Ordered list.
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i++;
      }
      tokens.push({ kind: "ol", items });
      continue;
    }
    // Blank line.
    if (line.trim() === "") {
      tokens.push({ kind: "blank" });
      i++;
      continue;
    }
    // Paragraph — accumulate consecutive non-empty non-special lines.
    const body: string[] = [line];
    i++;
    while (
      i < lines.length
      && lines[i].trim() !== ""
      && !/^```/.test(lines[i])
      && !/^#{1,3}\s/.test(lines[i])
      && !/^---+\s*$/.test(lines[i])
      && !/^>\s?/.test(lines[i])
      && !/^\s*[-*]\s+/.test(lines[i])
      && !/^\s*\d+\.\s+/.test(lines[i])
    ) {
      body.push(lines[i]);
      i++;
    }
    tokens.push({ kind: "para", body: body.join("\n") });
  }
  return tokens;
}

// Inline formatter — handles `code`, **bold**, _italic_/*italic*,
// [text](url). Returns React fragments so multiple format runs in
// a single line render cleanly. Order matters: we tokenize by
// scanning left-to-right with a non-overlapping regex alternation.
function renderInline(text: string): ReactNode {
  if (!text) return null;
  const out: ReactNode[] = [];
  // Use a single regex that captures the FIRST match of any inline
  // pattern. We re-scan from after each match.
  const pattern = /(`([^`]+)`)|(\*\*([^*]+)\*\*)|(__([^_]+)__)|(\*([^*\n]+)\*)|(_([^_\n]+)_)|(\[([^\]]+)\]\(([^)\s]+)\))/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = pattern.exec(text)) !== null) {
    if (m.index > last) {
      out.push(text.slice(last, m.index));
    }
    if (m[2] !== undefined) {
      out.push(
        <code key={key++} className="bg-slate-800/80 text-amber-200 rounded px-1 py-0.5 text-[0.875em] font-mono">
          {m[2]}
        </code>,
      );
    } else if (m[4] !== undefined || m[6] !== undefined) {
      const bold = m[4] ?? m[6];
      out.push(<strong key={key++} className="font-semibold text-slate-50">{bold}</strong>);
    } else if (m[8] !== undefined || m[10] !== undefined) {
      const italic = m[8] ?? m[10];
      out.push(<em key={key++} className="italic">{italic}</em>);
    } else if (m[12] !== undefined && m[13] !== undefined) {
      out.push(
        <a key={key++} href={m[13]} target="_blank" rel="noreferrer"
           className="text-sky-400 underline hover:text-sky-300">
          {m[12]}
        </a>,
      );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return <>{out}</>;
}

function CodeFence({ lang, body }: { lang: string; body: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(body);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // navigator.clipboard may be unavailable on http:// hosts; the
      // fence still renders, copy button just stops working — that's
      // a soft fallback we can live with.
    }
  };
  return (
    <div className="my-2 rounded-md overflow-hidden border border-slate-700/40 bg-slate-950/70">
      <div className="flex items-center px-2.5 py-1 border-b border-slate-700/40 text-[10px]">
        <span className="font-mono text-slate-400 uppercase tracking-wide">
          {lang || "code"}
        </span>
        <button
          type="button"
          onClick={onCopy}
          className={`ml-auto select-none px-2 py-0.5 rounded transition-colors ${
            copied
              ? "text-emerald-400 bg-emerald-900/30"
              : "text-slate-400 hover:text-slate-200 hover:bg-slate-700/40"
          }`}
        >
          {copied ? "✓ copied" : "copy"}
        </button>
      </div>
      <pre className="p-3 overflow-x-auto text-[12px] leading-snug font-mono text-slate-200">
        <code>{body}</code>
      </pre>
    </div>
  );
}

export function Markdown({ source }: { source: string }) {
  if (!source) return null;
  const tokens = tokenize(source);
  const out: ReactNode[] = [];
  let key = 0;
  for (const t of tokens) {
    switch (t.kind) {
      case "code":
        out.push(<CodeFence key={key++} lang={t.lang} body={t.body} />);
        break;
      case "h1":
        out.push(<h1 key={key++} className="text-lg font-bold mt-3 mb-1 text-slate-50">{renderInline(t.body)}</h1>);
        break;
      case "h2":
        out.push(<h2 key={key++} className="text-base font-bold mt-3 mb-1 text-slate-50">{renderInline(t.body)}</h2>);
        break;
      case "h3":
        out.push(<h3 key={key++} className="text-sm font-semibold mt-2 mb-1 text-slate-100">{renderInline(t.body)}</h3>);
        break;
      case "ul":
        out.push(
          <ul key={key++} className="list-disc pl-5 my-1 space-y-0.5 text-slate-100">
            {t.items.map((item, j) => <li key={j}>{renderInline(item)}</li>)}
          </ul>,
        );
        break;
      case "ol":
        out.push(
          <ol key={key++} className="list-decimal pl-5 my-1 space-y-0.5 text-slate-100">
            {t.items.map((item, j) => <li key={j}>{renderInline(item)}</li>)}
          </ol>,
        );
        break;
      case "quote":
        out.push(
          <blockquote key={key++} className="border-l-2 border-slate-600 pl-3 my-2 text-slate-300 italic">
            {renderInline(t.body)}
          </blockquote>,
        );
        break;
      case "hr":
        out.push(<hr key={key++} className="my-3 border-slate-700/60" />);
        break;
      case "para":
        out.push(
          <p key={key++} className="my-1 leading-relaxed whitespace-pre-wrap break-words text-slate-100">
            {renderInline(t.body)}
          </p>,
        );
        break;
      case "blank":
        // Collapse: paragraphs get their own margin; an extra blank
        // doesn't need to render anything visible. Skip.
        break;
    }
  }
  return <div className="markdown-body">{out}</div>;
}
