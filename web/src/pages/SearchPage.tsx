import { useState } from "react";
import { search } from "../api/search";
import type { RetrievalResult } from "../types";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [useMmr, setUseMmr] = useState(false);
  const [journals, setJournals] = useState("");
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function go() {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const journalList = journals.split(",").map((s) => s.trim()).filter(Boolean);
      const res = await search({
        query,
        top_k: topK,
        use_mmr: useMmr,
        filter: journalList.length ? { journals: journalList } : undefined,
      });
      setResults(res);
    } catch (e) {
      setError((e as Error).message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      <div className="card space-y-3">
        <div>
          <label className="label">Query</label>
          <input
            type="text"
            className="input"
            placeholder="off-target effects of base editors"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") go();
            }}
          />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          <div>
            <label className="label">Top-k</label>
            <input
              type="number"
              className="input"
              min={1}
              max={50}
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value) || 10)}
            />
          </div>
          <div className="flex items-end">
            <label className="inline-flex items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                className="rounded border-slate-300"
                checked={useMmr}
                onChange={(e) => setUseMmr(e.target.checked)}
              />
              MMR (diverse)
            </label>
          </div>
          <div className="col-span-2 sm:col-span-1">
            <label className="label">Journals (comma-sep)</label>
            <input
              type="text"
              className="input"
              placeholder="Nature, Cell"
              value={journals}
              onChange={(e) => setJournals(e.target.value)}
            />
          </div>
        </div>
        <div className="flex justify-end">
          <button className="btn-primary" onClick={go} disabled={loading || !query.trim()}>
            {loading ? "Searching…" : "Search"}
          </button>
        </div>
        {error && (
          <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
            {error}
          </div>
        )}
      </div>

      {results.length > 0 && (
        <div className="space-y-3">
          {results.map((r, i) => (
            <details key={r.chunk.id || i} className="card">
              <summary className="cursor-pointer flex flex-wrap items-center gap-2 text-sm">
                <span className="pill bg-slate-100 text-slate-700">
                  #{i + 1} · {r.score.toFixed(3)}
                </span>
                <span className="pill bg-brand-50 text-brand-700">
                  {r.chunk.citation_token ?? "—"}
                </span>
                {r.chunk.section && (
                  <span className="pill bg-slate-100 text-slate-600">§ {r.chunk.section}</span>
                )}
                <span className="font-medium text-slate-900 truncate">
                  {r.chunk.title ?? "(untitled)"}
                </span>
              </summary>
              <div className="mt-3 text-sm text-slate-800 whitespace-pre-wrap leading-relaxed">
                {r.chunk.text}
              </div>
              <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-500">
                {r.chunk.journal && <span>{r.chunk.journal}</span>}
                {r.chunk.publication_date && <span>{r.chunk.publication_date}</span>}
                {r.chunk.url && (
                  <a
                    className="text-brand-600 hover:underline"
                    href={r.chunk.url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    open source ↗
                  </a>
                )}
              </div>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}
