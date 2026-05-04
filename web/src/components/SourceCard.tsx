import type { AskSource } from "../types";

export default function SourceCard({ source }: { source: AskSource }) {
  return (
    <div className="card flex flex-col gap-1">
      <div className="flex items-center justify-between gap-2">
        <span className="pill bg-brand-50 text-brand-700">
          #{source.rank} · {source.citation ?? "—"}
        </span>
        <span className="text-xs text-slate-500">score {source.score.toFixed(3)}</span>
      </div>
      <div className="text-sm font-medium text-slate-900 leading-snug">
        {source.title ?? "(untitled)"}
      </div>
      {source.section && (
        <div className="text-xs text-slate-500">§ {source.section}</div>
      )}
      <div className="text-xs text-slate-700 line-clamp-3">{source.snippet}…</div>
      {source.url && (
        <a
          className="text-xs text-brand-600 hover:underline"
          href={source.url}
          target="_blank"
          rel="noreferrer"
        >
          open source ↗
        </a>
      )}
    </div>
  );
}
