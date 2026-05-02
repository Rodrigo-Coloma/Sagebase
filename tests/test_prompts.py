from __future__ import annotations

from datetime import date

from medlit.generation.prompts import build_messages, format_context
from medlit.models import Chunk, RetrievalResult, Source


def _result(pub_types: list[str], cite: str = "PMID:42") -> RetrievalResult:
    chunk = Chunk(
        id=f"{cite}:0",
        paper_id="p",
        text="Patients receiving drug X had 30% reduced mortality (HR 0.70, p<0.01).",
        chunk_index=0,
        section="Results",
        title="Trial of X",
        journal="Lancet",
        publication_date=date(2023, 1, 1),
        publication_types=pub_types,
        source=Source.PUBMED,
        citation_token=cite,
        pmid="42",
    )
    return RetrievalResult(chunk=chunk, score=0.9)


def test_format_context_includes_evidence_level_and_citation():
    rct = _result(["Randomized Controlled Trial"], cite="PMID:1")
    obs = _result(["Observational Study"], cite="PMID:2")
    other = _result([], cite="PMID:3")
    text = format_context([rct, obs, other])
    assert "[PMID:1]" in text and "evidence:strong" in text
    assert "[PMID:2]" in text and "evidence:moderate" in text
    assert "[PMID:3]" in text and "evidence:low" in text


def test_build_messages_has_system_and_user():
    msgs = build_messages("What about X?", [_result(["Meta-Analysis"])])
    assert msgs[0]["role"] == "system"
    assert "cite" in msgs[0]["content"].lower()
    assert msgs[1]["role"] == "user"
    assert "What about X?" in msgs[1]["content"]
    assert "[PMID:42]" in msgs[1]["content"]
