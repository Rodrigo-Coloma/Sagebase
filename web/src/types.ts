// TypeScript mirrors of the FastAPI / pydantic schemas.

export interface Chunk {
  id: string;
  paper_id: string;
  text: string;
  chunk_index: number;
  section?: string | null;
  page_number?: number | null;
  token_count?: number | null;
  title?: string | null;
  authors: string[];
  journal?: string | null;
  publication_date?: string | null;
  doi?: string | null;
  pmid?: string | null;
  arxiv_id?: string | null;
  mesh_terms: string[];
  publication_types: string[];
  source: string;
  url?: string | null;
  citation_token?: string | null;
}

export interface RetrievalResult {
  chunk: Chunk;
  score: number;
  dense_score?: number | null;
  sparse_score?: number | null;
  rerank_score?: number | null;
}

export interface PaperSummary {
  paper_id: string;
  title?: string | null;
  authors: string[];
  journal?: string | null;
  publication_date?: string | null;
  source: string;
  doi?: string | null;
  pmid?: string | null;
  arxiv_id?: string | null;
  url?: string | null;
  citation_token?: string | null;
  chunk_count: number;
}

export interface IngestionJob {
  id: string;
  status: "pending" | "running" | "completed" | "failed";
  source: string;
  query?: string | null;
  total: number;
  processed: number;
  failed: number;
  created_at: string;
  updated_at: string;
  error?: string | null;
}

export interface AskSource {
  rank: number;
  score: number;
  citation?: string | null;
  title?: string | null;
  url?: string | null;
  section?: string | null;
  doi?: string | null;
  pmid?: string | null;
  snippet: string;
}

export interface SearchFilterPayload {
  date_from?: string;
  date_to?: string;
  journals?: string[];
  authors?: string[];
  mesh_terms?: string[];
  publication_types?: string[];
  sources?: string[];
}
