import { useCallback, useEffect, useMemo, useState } from "react";
import PaperTable from "../components/PaperTable";
import { deletePaper, listPapers } from "../api/library";
import type { PaperSummary } from "../types";

export default function LibraryPage() {
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setPapers(await listPapers());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function onDelete(paperId: string) {
    if (!confirm("Delete this paper and all its chunks from the index?")) return;
    try {
      await deletePaper(paperId);
      setPapers((prev) => prev.filter((p) => p.paper_id !== paperId));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const filtered = useMemo(() => {
    if (!filter.trim()) return papers;
    const q = filter.toLowerCase();
    return papers.filter(
      (p) =>
        (p.title ?? "").toLowerCase().includes(q) ||
        (p.journal ?? "").toLowerCase().includes(q) ||
        (p.citation_token ?? "").toLowerCase().includes(q) ||
        p.authors.some((a) => a.toLowerCase().includes(q))
    );
  }, [filter, papers]);

  return (
    <div className="space-y-4">
      <div className="card flex flex-col sm:flex-row gap-3 sm:items-end">
        <div className="flex-1">
          <label className="label">Filter (title / journal / author / citation)</label>
          <input
            type="text"
            className="input"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="CRISPR, Nature, Liu, PMID:..."
          />
        </div>
        <button className="btn-secondary" onClick={refresh}>
          🔄 Refresh
        </button>
      </div>

      <div className="text-xs text-slate-500">
        {loading ? "Loading…" : `${filtered.length} paper(s)`}
      </div>

      {error && (
        <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
          {error}
        </div>
      )}

      {!loading && <PaperTable papers={filtered} onDelete={onDelete} />}
    </div>
  );
}
