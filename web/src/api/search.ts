import { apiFetch } from "./client";
import type { RetrievalResult, SearchFilterPayload } from "../types";

export interface SearchRequest {
  query: string;
  top_k?: number;
  use_mmr?: boolean;
  rerank?: boolean;
  filter?: SearchFilterPayload;
}

export async function search(req: SearchRequest): Promise<RetrievalResult[]> {
  const r = await apiFetch<{ results: RetrievalResult[] }>("/search", {
    method: "POST",
    body: JSON.stringify(req),
  });
  return r.results;
}
