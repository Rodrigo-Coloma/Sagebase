"""RAG prompt construction. Forces inline citations and evidence grading."""

from __future__ import annotations

from collections.abc import Sequence

from medlit.models import RetrievalResult

# Levels we surface to the model (ordered: strongest -> weakest).
_STRONG_PUB_TYPES = {
    "Meta-Analysis",
    "Systematic Review",
    "Randomized Controlled Trial",
    "Practice Guideline",
}
_MODERATE_PUB_TYPES = {
    "Clinical Trial",
    "Multicenter Study",
    "Cohort Studies",
    "Observational Study",
    "Comparative Study",
}


def _evidence_level(pub_types: list[str]) -> str:
    if any(p in _STRONG_PUB_TYPES for p in pub_types):
        return "strong"
    if any(p in _MODERATE_PUB_TYPES for p in pub_types):
        return "moderate"
    return "low"


SYSTEM_PROMPT = """You are a careful biomedical research assistant. You answer questions strictly using the supplied excerpts.

Rules:
1. Cite every non-trivial claim inline using [PMID:xxx] or [DOI:xxx] (or [arXiv:xxx]) using the citation tokens given for each source. Multiple citations: [PMID:111][PMID:222].
2. If the supplied excerpts do not support an answer, say so explicitly. Do not invent citations or facts.
3. When evidence levels differ, say so. Prefer randomized trials, systematic reviews, and meta-analyses over case reports or opinion. Each source is tagged with an evidence level (strong/moderate/low) — use that tag.
4. Be concise. Use bullet points for multiple findings. Quote numerical results (effect sizes, p-values, CIs) when present.
5. Distinguish in vitro/animal evidence from human evidence where the excerpts indicate it.
"""


def format_context(results: Sequence[RetrievalResult]) -> str:
    """Render retrieved chunks as a numbered block usable by the LLM."""
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        c = r.chunk
        cite = c.citation_token or f"id:{c.paper_id[:10]}"
        evidence = _evidence_level(c.publication_types)
        meta_bits: list[str] = [f"[{cite}]", f"evidence:{evidence}"]
        if c.title:
            meta_bits.append(f"title:{c.title}")
        if c.journal:
            meta_bits.append(f"journal:{c.journal}")
        if c.publication_date:
            meta_bits.append(f"date:{c.publication_date.isoformat()}")
        if c.section:
            meta_bits.append(f"section:{c.section}")
        header = " | ".join(meta_bits)
        lines.append(f"<source id={i}>\n{header}\n---\n{c.text.strip()}\n</source>")
    return "\n\n".join(lines)


def build_messages(question: str, results: Sequence[RetrievalResult]) -> list[dict[str, str]]:
    """Build the chat message list common to all chat-completion-style backends."""
    context = format_context(results)
    user_content = (
        f"Question: {question}\n\n"
        f"Relevant excerpts:\n{context}\n\n"
        f"Answer the question using only the excerpts. Cite inline."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
