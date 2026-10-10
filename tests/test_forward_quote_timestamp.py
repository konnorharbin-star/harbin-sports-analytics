import pandas as pd
import pytest

from harbin.edge_forward import (
    EDGE_FORWARD_SCHEMA_VERSION,
    append_edge_candidate_snapshots,
    _clean_forward_entries,
)


def _edge(*, quote):
    return pd.DataFrame([{
        "game_id": "401000001",
        "date": "2026-10-10T19:00:00+00:00",
        "away_team": "Visitor", "home_team": "Host",
        "market": "spread", "side": "Host", "line": -3.5,
        "odds": -110, "edge": 6.5, "ev": .12,
        "book": "ResearchOnlyBook", "quote_at": quote,
        "edge_reliability_status": "SUPPORTED_SUBGROUP",
        "price_evidence_status": "CONFIRMED",
    }])


def test_forward_snapshot_accepts_quote_captured_before_observation(tmp_path):
    path = tmp_path / "snapshots.csv"
    stamp = "2026-10-09T23:12:00+00:00"
    out = append_edge_candidate_snapshots(
        _edge(quote="2026-10-09T23:11:45+00:00"),
        path, snapshot_at=stamp,
    )
    assert out["appended"] == 1
    raw = pd.read_csv(path, low_memory=False)
    assert int(raw.edge_snapshot_schema_version.iloc[0]) == EDGE_FORWARD_SCHEMA_VERSION
    clean = _clean_forward_entries(raw)
    assert len(clean) == 1


def test_forward_snapshot_refuses_quote_captured_after_observation(tmp_path):
    with pytest.raises(ValueError, match="snapshot precedes"):
        append_edge_candidate_snapshots(
            _edge(quote="2026-10-09T23:13:00+00:00"),
            tmp_path / "snapshots.csv",
            snapshot_at="2026-10-09T23:12:00+00:00",
        )


def test_forward_snapshot_refuses_started_matchup(tmp_path):
    frame = _edge(quote="2026-10-09T23:10:00+00:00")
    frame.loc[0, "date"] = "2026-10-09T23:11:00+00:00"
    with pytest.raises(ValueError, match="started games"):
        append_edge_candidate_snapshots(
            frame, tmp_path / "snapshots.csv",
            snapshot_at="2026-10-09T23:12:00+00:00",
        )


def test_unresolved_source_book_is_not_qualified_forward_evidence(tmp_path):
    path = tmp_path / "snapshots.csv"
    candidate = _edge(quote="2026-10-09T23:11:45+00:00")
    candidate.loc[0, "book"] = "ActionNetwork book 15"
    out = append_edge_candidate_snapshots(
        candidate, path, snapshot_at="2026-10-09T23:12:00+00:00",
    )
    assert out["appended"] == 1
    clean = _clean_forward_entries(pd.read_csv(path, low_memory=False))
    assert len(clean) == 0
