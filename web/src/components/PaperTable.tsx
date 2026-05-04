import type { PaperSummary } from "../types";

function formatAuthors(authors: string[]): string {
  if (authors.length === 0) return "";
  if (authors.length <= 3) return authors.join(", ");
  return authors.slice(0, 3).join(", ") + " et al.";
}

interface Props {
  papers: PaperSummary[];
  onDelete?: (paperId: string) => void;
}

export default function PaperTable({ papers, onDelete }: Props) {
  if (papers.length === 0) {
    return (
      <div className="card text-sm text-slate-500 text-center">
        No papers indexed yet. Open the <strong>Ingest</strong> tab to add some.
      </div>
    );
  }
  return (
    <>
      {/* Mobile card view */}
      <div className="grid sm:hidden gap-3">
        {papers.map((p) => (
          <div key={p.paper_id} className="card space-y-1">
            <div className="text-sm font-medium text-slate-900">
              {p.title ?? "(untitled)"}
            </div>
            {p.authors.length > 0 && (
              <div className="text-xs text-slate-600">{formatAuthors(p.authors)}</div>
            )}
            <div className="text-xs text-slate-500 flex flex-wrap gap-2">
              {p.journal && <span>{p.journal}</span>}
              {p.publication_date && <span>{p.publication_date}</span>}
              <span className="pill bg-slate-100 text-slate-700">{p.source}</span>
              <span>{p.chunk_count} chunks</span>
            </div>
            <div className="flex items-center justify-between gap-2">
              {p.citation_token && (
                <span className="text-xs font-mono text-slate-500">{p.citation_token}</span>
              )}
              <div className="flex gap-3">
                {p.url && (
                  <a
                    href={p.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-brand-600 hover:underline"
                  >
                    open ↗
                  </a>
                )}
                {onDelete && (
                  <button
                    className="text-xs text-red-600 hover:underline"
                    onClick={() => onDelete(p.paper_id)}
                  >
                    delete
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Desktop table view */}
      <div className="hidden sm:block card overflow-x-auto p-0">
        <table className="min-w-full text-sm">
          <thead className="bg-slate-50 text-slate-600 text-xs uppercase tracking-wide">
            <tr>
              <th className="px-3 py-2 text-left">Title</th>
              <th className="px-3 py-2 text-left">Authors</th>
              <th className="px-3 py-2 text-left">Journal</th>
              <th className="px-3 py-2 text-left">Date</th>
              <th className="px-3 py-2 text-left">Source</th>
              <th className="px-3 py-2 text-left">Citation</th>
              <th className="px-3 py-2 text-right">Chunks</th>
              <th className="px-3 py-2 text-left">Link</th>
              {onDelete && <th className="px-3 py-2"></th>}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {papers.map((p) => (
              <tr key={p.paper_id} className="hover:bg-slate-50">
                <td className="px-3 py-2 text-slate-900 max-w-md">
                  <div className="line-clamp-2">{p.title ?? "(untitled)"}</div>
                </td>
                <td className="px-3 py-2 text-slate-700 max-w-[200px] truncate">
                  {formatAuthors(p.authors)}
                </td>
                <td className="px-3 py-2 text-slate-700">{p.journal ?? "—"}</td>
                <td className="px-3 py-2 text-slate-700 whitespace-nowrap">
                  {p.publication_date ?? "—"}
                </td>
                <td className="px-3 py-2">
                  <span className="pill bg-slate-100 text-slate-700">{p.source}</span>
                </td>
                <td className="px-3 py-2 font-mono text-xs text-slate-600">
                  {p.citation_token ?? "—"}
                </td>
                <td className="px-3 py-2 text-right text-slate-700">{p.chunk_count}</td>
                <td className="px-3 py-2">
                  {p.url ? (
                    <a
                      href={p.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-brand-600 hover:underline"
                    >
                      open ↗
                    </a>
                  ) : (
                    <span className="text-slate-400">—</span>
                  )}
                </td>
                {onDelete && (
                  <td className="px-3 py-2 text-right">
                    <button
                      className="text-xs text-red-600 hover:underline"
                      onClick={() => onDelete(p.paper_id)}
                    >
                      delete
                    </button>
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
