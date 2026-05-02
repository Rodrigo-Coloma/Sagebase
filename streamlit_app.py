"""medlit — Streamlit Community Cloud entry point.

Tuned for the free tier (~1 GB RAM):
  * Embeddings  : OpenAI text-embedding-3-small (no torch download)
  * Vector store: ChromaDB (in-process, persists to ./data/chroma)
  * Reranker    : disabled by default (cross-encoder is too heavy)
  * Generator   : Anthropic Claude (streamed)

Drop secrets in `.streamlit/secrets.toml` (or the Cloud "Secrets" UI).
See `.streamlit/secrets.toml.example`.
"""

from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
from datetime import date
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Push Streamlit secrets into env BEFORE importing medlit (settings cache env).
# ---------------------------------------------------------------------------
_SECRETS_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "VOYAGE_API_KEY",
    "NCBI_API_KEY",
    "NCBI_EMAIL",
    "UNPAYWALL_EMAIL",
    "QDRANT_URL",
    "QDRANT_API_KEY",
)
for key in _SECRETS_KEYS:
    try:
        val = st.secrets.get(key)
    except (FileNotFoundError, KeyError, AttributeError):
        val = None
    if val:
        os.environ.setdefault(key, str(val))

# Cloud-friendly defaults (overridden by env if user set them).
os.environ.setdefault("MEDLIT_EMBEDDING_BACKEND", "openai")
os.environ.setdefault("MEDLIT_EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("MEDLIT_VECTOR_BACKEND", "chroma")
os.environ.setdefault("MEDLIT_GENERATION_BACKEND", "anthropic")
os.environ.setdefault("MEDLIT_GENERATION_MODEL", "claude-sonnet-4-5")
os.environ.setdefault("MEDLIT_DATA_DIR", "./data")
os.environ.setdefault("MEDLIT_CACHE_DIR", "./data/cache")

# Make src/ importable when running on Streamlit Cloud.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from medlit.config import get_settings  # noqa: E402
from medlit.generation.base import GenerationRequest  # noqa: E402
from medlit.generation.factory import build_generator  # noqa: E402
from medlit.ingestion.base import IngestionQuery  # noqa: E402
from medlit.ingestion.biorxiv import BiorxivIngester  # noqa: E402
from medlit.ingestion.crossref import CrossrefIngester  # noqa: E402
from medlit.ingestion.pubmed import PubMedIngester  # noqa: E402
from medlit.pipeline import IngestionPipeline  # noqa: E402
from medlit.retrieval.pipeline import RetrievalPipeline  # noqa: E402
from medlit.storage.base import SearchFilter  # noqa: E402

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="medlit", page_icon="🧬", layout="wide")
st.title("🧬 medlit")
st.caption("Vectorized medical literature RAG · streamed answers with inline citations")


# ---------------------------------------------------------------------------
# Cached service singletons (Streamlit reruns the script on every interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _pipeline() -> IngestionPipeline:
    return IngestionPipeline()


@st.cache_resource(show_spinner=False)
def _retrieval() -> RetrievalPipeline:
    p = _pipeline()
    # Reranker disabled on cloud (cross-encoder requires ~1.3 GB).
    return RetrievalPipeline(embedder=p.embedder, store=p.store, reranker=None)


@st.cache_resource(show_spinner=False)
def _generator():  # type: ignore[no-untyped-def]
    return build_generator()


# ---------------------------------------------------------------------------
# Async-to-sync bridge for streaming generators
# ---------------------------------------------------------------------------
def stream_async(coro_factory):  # type: ignore[no-untyped-def]
    """Run an async generator in a worker thread, yield items synchronously
    so they can be passed to `st.write_stream` / a normal for-loop."""
    q: queue.Queue = queue.Queue()
    sentinel = object()

    def runner() -> None:
        async def consume() -> None:
            try:
                async for item in coro_factory():
                    q.put(item)
            except Exception as e:  # noqa: BLE001
                q.put(("__error__", repr(e)))
            finally:
                q.put(sentinel)

        asyncio.run(consume())

    threading.Thread(target=runner, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
            raise RuntimeError(item[1])
        yield item


# ---------------------------------------------------------------------------
# Sidebar — config, status, danger zone
# ---------------------------------------------------------------------------
settings = get_settings()
with st.sidebar:
    st.subheader("Setup")
    have_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    have_openai = bool(os.environ.get("OPENAI_API_KEY"))
    st.write(f"Embeddings: `{settings.embedding.backend}` / `{settings.embedding.model}`")
    st.write(f"Vector store: `{settings.vector_store.backend}`")
    st.write(f"Generator: `{settings.generation.backend}` / `{settings.generation.model}`")
    if not have_openai and settings.embedding.backend == "openai":
        st.error("OPENAI_API_KEY missing — needed for embeddings.")
    if not have_anthropic and settings.generation.backend == "anthropic":
        st.error("ANTHROPIC_API_KEY missing — needed for /ask.")
    st.divider()
    st.caption(
        "On Streamlit Cloud, ChromaDB persists to ephemeral disk: data survives "
        "during a session but resets on app reboot. For durable storage, point "
        "`QDRANT_URL` at a Qdrant Cloud cluster."
    )
    if st.button("Show index stats"):
        try:
            stats = _pipeline().store.stats()
            st.json(stats)
        except Exception as e:  # noqa: BLE001
            st.exception(e)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_ask, tab_search, tab_ingest = st.tabs(["💬 Ask", "🔎 Search", "📥 Ingest"])


# -------------------- Ask --------------------
with tab_ask:
    st.subheader("Ask a question of your indexed literature")
    if "answer" not in st.session_state:
        st.session_state.answer = ""

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            placeholder="What are the main delivery methods for CRISPR therapeutics in vivo?",
            height=80,
        )
        cols = st.columns(3)
        with cols[0]:
            top_k = st.slider("Top-k chunks", 1, 20, 8)
        with cols[1]:
            date_from = st.date_input("From", value=None)
        with cols[2]:
            date_to = st.date_input("To", value=None)
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted and question.strip():
        flt = SearchFilter(
            date_from=date_from if isinstance(date_from, date) else None,
            date_to=date_to if isinstance(date_to, date) else None,
        )
        with st.spinner("Retrieving context..."):
            try:
                results = _retrieval().search(question, filter_=flt, top_k=top_k)
            except Exception as e:  # noqa: BLE001
                st.exception(e)
                results = []

        if not results:
            st.warning("No relevant context found in the index. Ingest some papers first.")
        else:
            st.markdown("**Sources**")
            for i, r in enumerate(results, start=1):
                cite = r.chunk.citation_token or "—"
                title = r.chunk.title or "(untitled)"
                url = r.chunk.url or ""
                line = f"{i}. `{cite}` · {title}"
                if url:
                    line += f" · [link]({url})"
                st.markdown(line)

            st.markdown("**Answer**")
            req = GenerationRequest(
                question=question,
                context=results,
                max_tokens=settings.generation.max_tokens,
                temperature=settings.generation.temperature,
            )
            try:
                gen = _generator()
                st.write_stream(stream_async(lambda: gen.stream(req)))
            except Exception as e:  # noqa: BLE001
                st.exception(e)


# -------------------- Search --------------------
with tab_search:
    st.subheader("Hybrid search")
    q = st.text_input("Query", placeholder="off-target effects of base editors")
    cols = st.columns(3)
    with cols[0]:
        s_top_k = st.slider("Top-k", 1, 25, 10, key="s_top_k")
    with cols[1]:
        use_mmr = st.checkbox("MMR (diverse results)", value=False)
    with cols[2]:
        journals_raw = st.text_input("Filter: journals (comma-sep)", value="")

    if st.button("Search", key="search_btn") and q.strip():
        flt = SearchFilter(
            journals=[j.strip() for j in journals_raw.split(",") if j.strip()] or None,
        )
        with st.spinner("Searching..."):
            results = _retrieval().search(q, filter_=flt, top_k=s_top_k, use_mmr=use_mmr)
        if not results:
            st.info("No matches.")
        else:
            for i, r in enumerate(results, start=1):
                with st.expander(
                    f"#{i} · score {r.score:.3f} · {r.chunk.citation_token or ''} — {(r.chunk.title or '')[:120]}"
                ):
                    meta = []
                    if r.chunk.journal:
                        meta.append(f"**{r.chunk.journal}**")
                    if r.chunk.publication_date:
                        meta.append(r.chunk.publication_date.isoformat())
                    if r.chunk.section:
                        meta.append(f"§ {r.chunk.section}")
                    if meta:
                        st.caption(" · ".join(meta))
                    st.write(r.chunk.text)
                    if r.chunk.url:
                        st.markdown(f"[Open source]({r.chunk.url})")


# -------------------- Ingest --------------------
with tab_ingest:
    st.subheader("Ingest papers")
    sub = st.radio(
        "Source",
        ["PubMed", "bioRxiv / medRxiv", "DOI", "Upload PDF/XML/TXT"],
        horizontal=True,
    )

    if sub == "PubMed":
        with st.form("pubmed_form"):
            pq = st.text_input("Query", placeholder="CRISPR base editing")
            cols = st.columns(3)
            with cols[0]:
                pmax = st.number_input("Max results", min_value=1, max_value=2000, value=25)
            with cols[1]:
                pdf_from = st.date_input("From", value=None, key="pmf")
            with cols[2]:
                pdf_to = st.date_input("To", value=None, key="pmt")
            with_full = st.checkbox("Fetch open-access full text when available", value=True)
            go = st.form_submit_button("Ingest", type="primary")
        if go and pq.strip():
            iq = IngestionQuery(
                query=pq,
                max_results=int(pmax),
                date_from=pdf_from if isinstance(pdf_from, date) else None,
                date_to=pdf_to if isinstance(pdf_to, date) else None,
            )
            ingester = PubMedIngester()
            progress = st.progress(0.0, text="Starting…")
            log = st.empty()
            count = 0

            async def _gen():  # type: ignore[no-untyped-def]
                async for paper in _pipeline().ingest_from(
                    ingester, iq, with_full_text=with_full
                ):
                    yield paper

            try:
                for paper in stream_async(_gen):
                    count += 1
                    progress.progress(min(count / pmax, 1.0), text=f"{count}/{pmax}")
                    log.markdown(f"✅ `{paper.citation_token}` — {paper.title[:120]}")
                progress.progress(1.0, text=f"Done. {count} papers indexed.")
            except Exception as e:  # noqa: BLE001
                st.exception(e)

    elif sub == "bioRxiv / medRxiv":
        with st.form("biorxiv_form"):
            server = st.selectbox("Server", ["biorxiv", "medrxiv"])
            bq = st.text_input("Query (substring; leave empty for date-window only)", value="")
            cols = st.columns(3)
            with cols[0]:
                bmax = st.number_input("Max results", min_value=1, max_value=500, value=25)
            with cols[1]:
                bdf = st.date_input("From", value=None, key="brf")
            with cols[2]:
                bdt = st.date_input("To", value=None, key="brt")
            go = st.form_submit_button("Ingest", type="primary")
        if go:
            iq = IngestionQuery(
                query=bq or None,
                max_results=int(bmax),
                date_from=bdf if isinstance(bdf, date) else None,
                date_to=bdt if isinstance(bdt, date) else None,
            )
            ingester = BiorxivIngester(server=server)
            count = 0
            log = st.empty()

            async def _gen():  # type: ignore[no-untyped-def]
                async for paper in _pipeline().ingest_from(ingester, iq):
                    yield paper

            try:
                for paper in stream_async(_gen):
                    count += 1
                    log.markdown(f"✅ {paper.title[:140]}")
                st.success(f"{count} papers indexed.")
            except Exception as e:  # noqa: BLE001
                st.exception(e)

    elif sub == "DOI":
        doi = st.text_input("DOI", placeholder="10.1038/s41586-023-06547-x")
        if st.button("Ingest DOI", type="primary") and doi.strip():
            cr = CrossrefIngester(mailto=settings.unpaywall_email)

            async def _run() -> str:
                paper = await cr.fetch_one(doi.strip())
                if paper is None:
                    return "DOI not found in Crossref."
                ok = await _pipeline()._process_paper(paper, ingester=cr, with_full_text=True)
                return f"{'Indexed' if ok else 'Skipped'}: {paper.title}"

            try:
                msg = asyncio.run(_run())
                st.success(msg) if "Indexed" in msg else st.warning(msg)
            except Exception as e:  # noqa: BLE001
                st.exception(e)

    elif sub == "Upload PDF/XML/TXT":
        files = st.file_uploader(
            "Upload one or more files",
            accept_multiple_files=True,
            type=["pdf", "xml", "txt", "md"],
        )
        if files and st.button("Ingest uploads", type="primary"):
            up_dir = Path(settings.medlit_data_dir) / "uploads"
            up_dir.mkdir(parents=True, exist_ok=True)
            paths: list[Path] = []
            for f in files:
                dest = up_dir / f.name
                dest.write_bytes(f.getvalue())
                paths.append(dest)

            count = 0
            log = st.empty()

            async def _gen():  # type: ignore[no-untyped-def]
                async for paper in _pipeline().ingest_files(paths):
                    yield paper

            try:
                for paper in stream_async(_gen):
                    count += 1
                    log.markdown(f"✅ {paper.title[:140]}")
                st.success(f"{count} files indexed.")
            except Exception as e:  # noqa: BLE001
                st.exception(e)
