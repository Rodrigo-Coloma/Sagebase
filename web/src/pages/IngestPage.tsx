import { useEffect, useRef, useState } from "react";
import {
  ingestArxiv,
  ingestBiorxiv,
  ingestDoi,
  ingestPubmed,
  pollJob,
  uploadFiles,
} from "../api/ingest";
import type { IngestionJob } from "../types";

type Mode = "pubmed" | "biorxiv" | "arxiv" | "doi" | "upload";

const MODES: { id: Mode; label: string }[] = [
  { id: "pubmed", label: "PubMed" },
  { id: "biorxiv", label: "bioRxiv / medRxiv" },
  { id: "arxiv", label: "arXiv" },
  { id: "doi", label: "DOI" },
  { id: "upload", label: "Upload" },
];

export default function IngestPage() {
  const [mode, setMode] = useState<Mode>("pubmed");
  const [job, setJob] = useState<IngestionJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Cancel any active polling when leaving the page or switching modes.
  useEffect(() => () => abortRef.current?.abort(), []);

  async function startJob(creator: () => Promise<{ job: IngestionJob }>) {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setError(null);
    setJob(null);
    try {
      const { job: started } = await creator();
      setJob(started);
      pollJob(started.id, setJob, { signal: ac.signal }).catch((e) => {
        if ((e as Error).message !== "aborted") setError((e as Error).message);
      });
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <div className="space-y-4">
      <div className="card">
        <div className="flex flex-wrap gap-2">
          {MODES.map((m) => (
            <button
              key={m.id}
              className={`btn ${
                mode === m.id
                  ? "bg-brand-500 text-white"
                  : "bg-white border border-slate-300 text-slate-700 hover:bg-slate-50"
              }`}
              onClick={() => setMode(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {mode === "pubmed" && <PubmedForm onSubmit={startJob} />}
      {mode === "biorxiv" && <BiorxivForm onSubmit={startJob} />}
      {mode === "arxiv" && <ArxivForm onSubmit={startJob} />}
      {mode === "doi" && <DoiForm onSubmit={startJob} />}
      {mode === "upload" && <UploadForm onSubmit={startJob} />}

      {error && (
        <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-2">
          {error}
        </div>
      )}

      {job && <JobStatus job={job} />}
    </div>
  );
}

// ---------- forms ----------
type SubmitFn = (creator: () => Promise<{ job: IngestionJob }>) => void;

function PubmedForm({ onSubmit }: { onSubmit: SubmitFn }) {
  const [query, setQuery] = useState("");
  const [maxResults, setMaxResults] = useState(25);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [fullText, setFullText] = useState(true);

  return (
    <div className="card space-y-3">
      <div>
        <label className="label">Query</label>
        <input
          className="input"
          placeholder="CRISPR base editing"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>
      <div className="grid grid-cols-3 gap-3">
        <div>
          <label className="label">Max results</label>
          <input
            type="number"
            className="input"
            min={1}
            max={2000}
            value={maxResults}
            onChange={(e) => setMaxResults(parseInt(e.target.value) || 25)}
          />
        </div>
        <div>
          <label className="label">From</label>
          <input
            type="date"
            className="input"
            value={from}
            onChange={(e) => setFrom(e.target.value)}
          />
        </div>
        <div>
          <label className="label">To</label>
          <input type="date" className="input" value={to} onChange={(e) => setTo(e.target.value)} />
        </div>
      </div>
      <label className="inline-flex items-center gap-2 text-sm text-slate-700">
        <input
          type="checkbox"
          className="rounded border-slate-300"
          checked={fullText}
          onChange={(e) => setFullText(e.target.checked)}
        />
        Fetch open-access full text when available
      </label>
      <div className="flex justify-end">
        <button
          className="btn-primary"
          disabled={!query.trim()}
          onClick={() =>
            onSubmit(() =>
              ingestPubmed({
                query,
                max_results: maxResults,
                date_from: from || undefined,
                date_to: to || undefined,
                with_full_text: fullText,
              })
            )
          }
        >
          Ingest
        </button>
      </div>
    </div>
  );
}

function BiorxivForm({ onSubmit }: { onSubmit: SubmitFn }) {
  const [query, setQuery] = useState("");
  const [server, setServer] = useState<"biorxiv" | "medrxiv">("biorxiv");
  const [maxResults, setMaxResults] = useState(25);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  return (
    <div className="card space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="label">Server</label>
          <select
            className="input"
            value={server}
            onChange={(e) => setServer(e.target.value as "biorxiv" | "medrxiv")}
          >
            <option value="biorxiv">biorxiv</option>
            <option value="medrxiv">medrxiv</option>
          </select>
        </div>
        <div>
          <label className="label">Max results</label>
          <input
            type="number"
            className="input"
            min={1}
            max={500}
            value={maxResults}
            onChange={(e) => setMaxResults(parseInt(e.target.value) || 25)}
          />
        </div>
      </div>
      <div>
        <label className="label">Substring filter (optional)</label>
        <input
          className="input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="base editing"
        />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="label">From</label>
          <input
            type="date"
            className="input"
            value={from}
            onChange={(e) => setFrom(e.target.value)}
          />
        </div>
        <div>
          <label className="label">To</label>
          <input type="date" className="input" value={to} onChange={(e) => setTo(e.target.value)} />
        </div>
      </div>
      <div className="flex justify-end">
        <button
          className="btn-primary"
          onClick={() =>
            onSubmit(() =>
              ingestBiorxiv({
                query: query || undefined,
                server,
                max_results: maxResults,
                date_from: from || undefined,
                date_to: to || undefined,
              })
            )
          }
        >
          Ingest
        </button>
      </div>
    </div>
  );
}

function ArxivForm({ onSubmit }: { onSubmit: SubmitFn }) {
  const [query, setQuery] = useState("");
  const [maxResults, setMaxResults] = useState(25);
  return (
    <div className="card space-y-3">
      <div>
        <label className="label">Query</label>
        <input
          className="input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder='cat:q-bio.QM AND "deep learning"'
        />
      </div>
      <div>
        <label className="label">Max results</label>
        <input
          type="number"
          className="input"
          min={1}
          max={500}
          value={maxResults}
          onChange={(e) => setMaxResults(parseInt(e.target.value) || 25)}
        />
      </div>
      <div className="flex justify-end">
        <button
          className="btn-primary"
          disabled={!query.trim()}
          onClick={() => onSubmit(() => ingestArxiv({ query, max_results: maxResults }))}
        >
          Ingest
        </button>
      </div>
    </div>
  );
}

function DoiForm({ onSubmit }: { onSubmit: SubmitFn }) {
  const [doi, setDoi] = useState("");
  return (
    <div className="card space-y-3">
      <div>
        <label className="label">DOI</label>
        <input
          className="input"
          value={doi}
          onChange={(e) => setDoi(e.target.value)}
          placeholder="10.1038/s41586-023-06547-x"
        />
      </div>
      <div className="flex justify-end">
        <button
          className="btn-primary"
          disabled={!doi.trim()}
          onClick={() => onSubmit(() => ingestDoi({ doi }))}
        >
          Ingest
        </button>
      </div>
    </div>
  );
}

function UploadForm({ onSubmit }: { onSubmit: SubmitFn }) {
  const [files, setFiles] = useState<File[]>([]);
  return (
    <div className="card space-y-3">
      <div>
        <label className="label">Files (PDF, XML, TXT, MD)</label>
        <input
          type="file"
          multiple
          accept=".pdf,.xml,.txt,.md,application/pdf,text/xml,text/plain,text/markdown"
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          className="block w-full text-sm text-slate-700 file:mr-3 file:rounded-md file:border-0 file:bg-brand-50 file:px-3 file:py-2 file:text-brand-700 hover:file:bg-brand-100"
        />
        {files.length > 0 && (
          <div className="text-xs text-slate-600 mt-1">
            {files.length} file(s) selected.
          </div>
        )}
      </div>
      <div className="flex justify-end">
        <button
          className="btn-primary"
          disabled={files.length === 0}
          onClick={() => onSubmit(() => uploadFiles(files))}
        >
          Upload &amp; ingest
        </button>
      </div>
    </div>
  );
}

// ---------- job status ----------
function JobStatus({ job }: { job: IngestionJob }) {
  const pct =
    job.total > 0 ? Math.min(100, Math.round((job.processed / job.total) * 100)) : null;
  const colour =
    job.status === "completed"
      ? "bg-emerald-500"
      : job.status === "failed"
      ? "bg-red-500"
      : "bg-brand-500";
  return (
    <div className="card space-y-2">
      <div className="flex justify-between text-sm">
        <span className="font-medium text-slate-800">Job {job.id.slice(0, 8)}…</span>
        <span className="text-slate-600">
          {job.source} · <span className="font-mono">{job.status}</span>
        </span>
      </div>
      <div className="h-2 bg-slate-100 rounded overflow-hidden">
        <div
          className={`${colour} h-full transition-[width]`}
          style={{ width: pct !== null ? `${pct}%` : job.status === "running" ? "30%" : "0%" }}
        />
      </div>
      <div className="text-xs text-slate-600">
        Processed {job.processed}
        {job.total > 0 && ` / ${job.total}`}
        {job.failed > 0 && ` · ${job.failed} failed`}
      </div>
      {job.error && (
        <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded p-2">
          {job.error}
        </div>
      )}
    </div>
  );
}
