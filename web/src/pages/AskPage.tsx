import { useRef, useState } from "react";
import { askStream } from "../api/ask";
import SourceCard from "../components/SourceCard";
import type { AskSource } from "../types";

export default function AskPage() {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(8);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [sources, setSources] = useState<AskSource[]>([]);
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function ask() {
    if (!question.trim()) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setLoading(true);
    setError(null);
    setSources([]);
    setAnswer("");

    await askStream(
      {
        question,
        top_k: topK,
        filter:
          dateFrom || dateTo
            ? { date_from: dateFrom || undefined, date_to: dateTo || undefined }
            : undefined,
      },
      {
        onSources: setSources,
        onToken: (t) => setAnswer((prev) => prev + t),
        onDone: () => setLoading(false),
        onError: (e) => {
          setError((e as Error).message);
          setLoading(false);
        },
        signal: ac.signal,
      }
    );
  }

  function cancel() {
    abortRef.current?.abort();
    setLoading(false);
  }

  return (
    <div className="space-y-4">
      <div className="card space-y-3">
        <div>
          <label className="label">Question</label>
          <textarea
            className="input min-h-[100px] resize-y"
            placeholder="What are the main delivery methods for CRISPR therapeutics in vivo?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
                e.preventDefault();
                ask();
              }
            }}
          />
        </div>
        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="label">Top-k chunks</label>
            <input
              type="number"
              className="input"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value) || 8)}
            />
          </div>
          <div>
            <label className="label">From</label>
            <input
              type="date"
              className="input"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
            />
          </div>
          <div>
            <label className="label">To</label>
            <input
              type="date"
              className="input"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
            />
          </div>
        </div>
        <div className="flex justify-end gap-2">
          {loading && (
            <button className="btn-secondary" onClick={cancel}>
              Stop
            </button>
          )}
          <button className="btn-primary" onClick={ask} disabled={loading || !question.trim()}>
            {loading ? "Asking…" : "Ask"}
          </button>
        </div>
        {error && (
          <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
            {error}
          </div>
        )}
      </div>

      {sources.length > 0 && (
        <div>
          <h2 className="text-sm font-semibold text-slate-700 mb-2">
            Sources ({sources.length})
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {sources.map((s) => (
              <SourceCard key={s.rank} source={s} />
            ))}
          </div>
        </div>
      )}

      {(answer || loading) && (
        <div className="card">
          <h2 className="text-sm font-semibold text-slate-700 mb-2">Answer</h2>
          <div className="prose prose-sm max-w-none whitespace-pre-wrap text-slate-900">
            {answer}
            {loading && <span className="inline-block w-2 h-4 bg-brand-500 animate-pulse ml-1 align-text-bottom" />}
          </div>
        </div>
      )}
    </div>
  );
}
