import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Renders markdown-formatted text with Tailwind-styled defaults that
 * match the surrounding UI density. Used for assistant chat replies and
 * for ToltIQ answer bodies (both preset and follow-up), where models
 * commonly emit `**bold**`, `## headings`, `- bullets`, etc.
 *
 * Styling intent: tight spacing, slate palette, sized to fit inside a
 * narrow chat sidebar and a wider data-room accordion. Headings stay
 * close to body size — these are inline answers, not document headings.
 */
export default function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => (
          <h1 className="text-base font-bold mt-3 mb-1 first:mt-0">{children}</h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-base font-semibold mt-3 mb-1 first:mt-0">{children}</h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-sm font-semibold mt-2 mb-1 first:mt-0">{children}</h3>
        ),
        h4: ({ children }) => (
          <h4 className="text-sm font-semibold mt-2 mb-0.5 first:mt-0">{children}</h4>
        ),
        p: ({ children }) => (
          <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>
        ),
        ul: ({ children }) => (
          <ul className="list-disc list-outside pl-5 mb-2 last:mb-0 space-y-0.5">
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal list-outside pl-5 mb-2 last:mb-0 space-y-0.5">
            {children}
          </ol>
        ),
        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
        strong: ({ children }) => (
          <strong className="font-semibold">{children}</strong>
        ),
        em: ({ children }) => <em className="italic">{children}</em>,
        a: ({ href, children }) => (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline"
          >
            {children}
          </a>
        ),
        code: ({ className, children, ...props }) => {
          // react-markdown gives fenced code a `language-xxx` className;
          // inline code has none. We render both with the same monospace
          // chip — fenced blocks get the `pre` wrapper below.
          const isInline = !className;
          if (isInline) {
            return (
              <code
                className="bg-slate-100 px-1 py-0.5 rounded text-[0.85em] font-mono"
                {...props}
              >
                {children}
              </code>
            );
          }
          return (
            <code className="font-mono text-xs" {...props}>
              {children}
            </code>
          );
        },
        pre: ({ children }) => (
          <pre className="bg-slate-100 p-2 rounded overflow-x-auto my-2 text-xs">
            {children}
          </pre>
        ),
        blockquote: ({ children }) => (
          <blockquote className="border-l-2 border-slate-300 pl-3 my-2 text-slate-600 italic">
            {children}
          </blockquote>
        ),
        hr: () => <hr className="my-3 border-slate-200" />,
        table: ({ children }) => (
          <div className="overflow-x-auto my-2">
            <table className="text-xs border-collapse">{children}</table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border border-slate-300 px-2 py-1 bg-slate-50 font-semibold text-left">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border border-slate-300 px-2 py-1 align-top">
            {children}
          </td>
        ),
      }}
    >
      {children}
    </ReactMarkdown>
  );
}
