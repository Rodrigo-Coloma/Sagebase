"""PMC NXML parser. Preserves IMRaD section structure for downstream chunking."""

from __future__ import annotations

from dataclasses import dataclass, field

from lxml import etree

from medlit.processing.normalizer import normalize_text


@dataclass(slots=True)
class Section:
    title: str
    text: str
    subsections: list["Section"] = field(default_factory=list)


@dataclass(slots=True)
class ParsedPMC:
    title: str | None
    abstract: str | None
    sections: list[Section]

    def iter_flat(self) -> list[tuple[str, str]]:
        """Flatten as (section_label, text) for the chunker."""
        out: list[tuple[str, str]] = []
        if self.abstract:
            out.append(("Abstract", self.abstract))
        for s in self.sections:
            _flatten(s, prefix="", acc=out)
        return out


def _flatten(s: Section, *, prefix: str, acc: list[tuple[str, str]]) -> None:
    label = f"{prefix}{s.title}" if s.title else prefix.rstrip(" >")
    if s.text.strip():
        acc.append((label or "Body", s.text))
    for child in s.subsections:
        _flatten(child, prefix=f"{label} > " if label else "", acc=acc)


def parse_pmc_xml(xml: bytes | str) -> ParsedPMC:
    """Parse NXML produced by efetch?db=pmc&rettype=xml."""
    if isinstance(xml, str):
        xml_bytes = xml.encode("utf-8")
    else:
        xml_bytes = xml
    root = etree.fromstring(xml_bytes)  # noqa: S320
    found = root.find(".//article")
    article = found if found is not None else root
    title = _text(article.find(".//title-group/article-title"))
    abstract_nodes = article.findall(".//abstract")
    abstract = "\n\n".join(filter(None, (_text(a) for a in abstract_nodes))) or None
    body = article.find(".//body")
    sections: list[Section] = []
    if body is not None:
        for sec in body.findall("./sec"):
            sections.append(_section(sec))
    return ParsedPMC(
        title=normalize_text(title) if title else None,
        abstract=normalize_text(abstract) if abstract else None,
        sections=sections,
    )


def _section(sec: etree._Element) -> Section:
    title = _text(sec.find("./title")) or ""
    paragraphs: list[str] = []
    subs: list[Section] = []
    for child in sec:
        tag = etree.QName(child).localname
        if tag == "title":
            continue
        if tag == "sec":
            subs.append(_section(child))
        elif tag in {"p", "list", "disp-quote"}:
            t = _text(child)
            if t:
                paragraphs.append(t)
        elif tag in {"table-wrap", "fig"}:
            # Keep a placeholder so chunker knows non-prose content was here.
            label = _text(child.find(".//label")) or tag
            caption = _text(child.find(".//caption")) or ""
            paragraphs.append(f"[{label}] {caption}".strip())
    text = normalize_text("\n\n".join(paragraphs)) if paragraphs else ""
    return Section(title=normalize_text(title), text=text, subsections=subs)


def _text(el: etree._Element | None) -> str | None:
    if el is None:
        return None
    s = "".join(el.itertext()).strip()
    return s or None
