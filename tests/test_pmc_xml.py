from __future__ import annotations

from pathlib import Path

from medlit.processing.pmc_xml import parse_pmc_xml

FIXTURES = Path(__file__).parent / "fixtures"


def test_pmc_iter_flat_emits_sections_in_order():
    parsed = parse_pmc_xml((FIXTURES / "pmc_sample.xml").read_bytes())
    flat = parsed.iter_flat()
    labels = [label for label, _ in flat]
    assert "Abstract" in labels
    # Subsection nesting should be reflected in the label.
    assert any("Methods > Cell culture" in lab for lab in labels)
