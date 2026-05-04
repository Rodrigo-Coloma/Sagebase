import { API_BASE, apiFetch } from "./client";
import type { IngestionJob } from "../types";

export interface PubMedIngestRequest {
  query: string;
  max_results?: number;
  date_from?: string;
  date_to?: string;
  with_full_text?: boolean;
}

export interface BiorxivIngestRequest {
  query?: string;
  max_results?: number;
  server?: "biorxiv" | "medrxiv";
  date_from?: string;
  date_to?: string;
}

export interface ArxivIngestRequest {
  query: string;
  max_results?: number;
}

export interface DoiIngestRequest {
  doi: string;
}

const post = <Req, Resp>(path: string, body: Req): Promise<Resp> =>
  apiFetch<Resp>(path, { method: "POST", body: JSON.stringify(body) });

export const ingestPubmed = (req: PubMedIngestRequest) =>
  post<PubMedIngestRequest, { job: IngestionJob }>("/ingest/pubmed", req);

export const ingestBiorxiv = (req: BiorxivIngestRequest) =>
  post<BiorxivIngestRequest, { job: IngestionJob }>("/ingest/biorxiv", req);

export const ingestArxiv = (req: ArxivIngestRequest) =>
  post<ArxivIngestRequest, { job: IngestionJob }>("/ingest/arxiv", req);

export const ingestDoi = (req: DoiIngestRequest) =>
  post<DoiIngestRequest, { job: IngestionJob }>("/ingest/doi", req);

export const getJob = (id: string): Promise<IngestionJob> =>
  apiFetch(`/ingest/jobs/${encodeURIComponent(id)}`);

export async function uploadFiles(files: File[]): Promise<{ job: IngestionJob }> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  // FormData -> let the browser set Content-Type with boundary
  const resp = await fetch(`${API_BASE}/ingest/upload`, {
    method: "POST",
    body: fd,
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

/** Poll until the job is no longer pending/running. */
export async function pollJob(
  id: string,
  onTick: (job: IngestionJob) => void,
  opts: { intervalMs?: number; signal?: AbortSignal } = {}
): Promise<IngestionJob> {
  const interval = opts.intervalMs ?? 1500;
  while (true) {
    if (opts.signal?.aborted) throw new Error("aborted");
    const job = await getJob(id);
    onTick(job);
    if (job.status === "completed" || job.status === "failed") return job;
    await new Promise((r) => setTimeout(r, interval));
  }
}
