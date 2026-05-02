from __future__ import annotations

from medlit.processing.normalizer import normalize_text


def test_normalize_dehyphenates_linebreaks():
    text = "rapamy-\ncin inhibits mTOR"
    assert normalize_text(text) == "rapamycin inhibits mTOR"


def test_normalize_collapses_blank_runs():
    text = "para1\n\n\n\npara2"
    assert normalize_text(text) == "para1\n\npara2"


def test_normalize_handles_ligatures():
    assert normalize_text("ﬁnal" ) == "final"
