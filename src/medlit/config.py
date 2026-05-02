"""Application settings, layered: defaults <- config/default.yaml <- env vars."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


class EmbeddingConfig(BaseModel):
    backend: Literal["sentence_transformers", "openai", "voyage"] = "sentence_transformers"
    model: str = "BAAI/bge-large-en-v1.5"
    batch_size: int = 32
    cache_dir: str = "./data/cache/embeddings"


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str | None = None


class ChromaConfig(BaseModel):
    persist_dir: str = "./data/chroma"


class PgVectorConfig(BaseModel):
    dsn: str = "postgresql://medlit:medlit@localhost:5432/medlit"


class VectorStoreConfig(BaseModel):
    backend: Literal["qdrant", "chroma", "pgvector"] = "qdrant"
    collection: str = "medlit"
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    pgvector: PgVectorConfig = Field(default_factory=PgVectorConfig)


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    respect_sections: bool = True
    keep_tables_intact: bool = True


class RetrievalConfig(BaseModel):
    top_k_dense: int = 30
    top_k_sparse: int = 30
    rrf_k: int = 60
    rerank_top_k: int = 10
    use_mmr: bool = False
    mmr_lambda: float = 0.5


class RerankerConfig(BaseModel):
    model: str = "BAAI/bge-reranker-large"
    enabled: bool = True


class GenerationConfig(BaseModel):
    backend: Literal["anthropic", "openai", "ollama"] = "anthropic"
    model: str = "claude-sonnet-4-5"
    max_tokens: int = 1024
    temperature: float = 0.2
    stream: bool = True


class PubMedIngestConfig(BaseModel):
    rate_limit_rps: int = 10
    batch_size: int = 100


class IngestionConfig(BaseModel):
    pubmed: PubMedIngestConfig = Field(default_factory=PubMedIngestConfig)
    open_access_only_fulltext: bool = True
    unpaywall_enabled: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
        "{name}:{function}:{line} - {message}"
    )


class Settings(BaseSettings):
    """Top-level settings. Env vars (MEDLIT_*, NCBI_*, etc.) override YAML."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    # Direct env vars (flat, easy override)
    ncbi_api_key: str | None = None
    ncbi_tool: str = "medlit"
    ncbi_email: str = "you@example.com"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    voyage_api_key: str | None = None
    unpaywall_email: str = "you@example.com"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "medlit"

    medlit_vector_backend: str = "qdrant"
    medlit_embedding_backend: str = "sentence_transformers"
    medlit_embedding_model: str = "BAAI/bge-large-en-v1.5"
    medlit_reranker_model: str = "BAAI/bge-reranker-large"
    medlit_generation_backend: str = "anthropic"
    medlit_generation_model: str = "claude-sonnet-4-5"
    medlit_data_dir: str = "./data"
    medlit_cache_dir: str = "./data/cache"
    medlit_api_host: str = "0.0.0.0"
    medlit_api_port: int = 8000
    medlit_log_level: str = "INFO"

    # Composed configs (loaded from YAML)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def merged(self) -> "Settings":
        """Apply env overrides on top of nested configs."""
        self.embedding.backend = self.medlit_embedding_backend  # type: ignore[assignment]
        self.embedding.model = self.medlit_embedding_model
        self.vector_store.backend = self.medlit_vector_backend  # type: ignore[assignment]
        self.vector_store.collection = self.qdrant_collection
        self.vector_store.qdrant.url = self.qdrant_url
        self.vector_store.qdrant.api_key = self.qdrant_api_key
        self.reranker.model = self.medlit_reranker_model
        self.generation.backend = self.medlit_generation_backend  # type: ignore[assignment]
        self.generation.model = self.medlit_generation_model
        self.logging.level = self.medlit_log_level
        return self


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_settings(config_path: Path | None = None) -> Settings:
    """Load settings, layering YAML defaults under environment variables."""
    yaml_data = _load_yaml(config_path or DEFAULT_CONFIG_PATH)
    settings = Settings(**yaml_data)
    return settings.merged()
