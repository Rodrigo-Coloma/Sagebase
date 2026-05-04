import { apiFetch } from "./client";
import type { Chunk, PaperSummary } from "../types";

export const listPapers = (): Promise<PaperSummary[]> => apiFetch("/papers");

export const getPaperChunks = (id: string): Promise<Chunk[]> =>
  apiFetch(`/papers/${encodeURIComponent(id)}`);

export const deletePaper = (id: string): Promise<{ deleted: number }> =>
  apiFetch(`/papers/${encodeURIComponent(id)}`, { method: "DELETE" });

export const getStats = (): Promise<{
  store: Record<string, unknown>;
  embedding_model: string;
  generation_model: string;
}> => apiFetch("/stats");
