from __future__ import annotations

from medlit.scheduler.watches import _trigger_from, load_watches


def test_load_watches_from_default(tmp_path):
    p = tmp_path / "watches.yaml"
    p.write_text(
        """
        watches:
          - name: w1
            source: pubmed
            query: foo
            max_results: 10
            schedule:
              cron: "0 6 * * *"
        """
    )
    watches = load_watches(p)
    assert len(watches) == 1
    assert watches[0].name == "w1"
    assert watches[0].source == "pubmed"


def test_trigger_from_cron():
    t = _trigger_from({"cron": "0 6 * * *"})
    assert t is not None


def test_trigger_from_every():
    t = _trigger_from({"every": "12h"})
    assert t is not None


def test_trigger_from_invalid_every_raises():
    import pytest
    with pytest.raises(ValueError):
        _trigger_from({"every": "12 hours"})
