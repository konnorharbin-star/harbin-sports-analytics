from __future__ import annotations

import json

import pandas as pd

from harbin.edge_forward import (
    _clean_forward_entries,
    append_edge_candidate_snapshots,
    grade_edge_forward_history,
)


def _edge_row(**overrides):
    row = {
        "game_id": "g1",
        "date": "2026-10-10T18:00:00Z",
        "away_team": "Away",
        "home_team": "Home",
        "market": "spread",
        "side": "Away",
        "line": 7.5,
        "odds": -110,
        "probability": 0.62,
        "edge": 6.5,
        "ev": 0.18,
        "badge": "STRONG",
        "book": "Book A",
        "quote_at": "2026-10-10T15:59:00Z",
        "regime_status": "PERSISTENT_CANDIDATE",
        "edge_reliability_status": "SUPPORTED_SUBGROUP",
        "regime_band": "6-8",
        "historical_bets": 231,
        "historical_win_rate": 0.61,
        "historical_roi": 0.162,
        "historical_avg_clv": 0.027,
        "profitable_seasons": 3,
        "season_count": 3,
        "subgroup_key": "underdog|away",
        "subgroup_status": "SUPPORTED_SUBGROUP",
        "subgroup_bets": 89,
        "subgroup_win_rate": 0.629,
        "subgroup_roi": 0.201,
        "subgroup_avg_clv": 0.015,
        "subgroup_profitable_seasons": 3,
        "subgroup_season_count": 3,
        "subgroup_validation_design": "latest_season_holdout",
        "subgroup_discovery_seasons": "2023|2024",
        "subgroup_discovery_bets": 64,
        "subgroup_discovery_roi": 0.193,
        "subgroup_holdout_season": "2025",
        "subgroup_holdout_bets": 25,
        "subgroup_holdout_roi": 0.222,
        "subgroup_holdout_win_rate": 0.64,
        "subgroup_holdout_confirmed": True,
        "currently_selected": True,
        "selected_quant_signal": "STRONG",
        "selected_quant_ev": 0.18,
    }
    row.update(overrides)
    return row


def test_append_forward_edge_snapshots_keeps_supported_and_dedupes_retry(tmp_path):
    path = tmp_path / "edge_candidate_snapshots.csv"
    frame = pd.DataFrame(
        [
            _edge_row(),
            _edge_row(
                game_id="g2",
                edge_reliability_status="CONTRAINDICATED_SUBGROUP",
                subgroup_status="CONTRAINDICATED_SUBGROUP",
            ),
        ]
    )

    first = append_edge_candidate_snapshots(
        frame,
        path,
        "2026-10-10T16:00:00Z",
    )
    second = append_edge_candidate_snapshots(
        frame,
        path,
        "2026-10-10T16:00:00Z",
    )

    stored = pd.read_csv(path)
    assert first["appended"] == 1
    assert second["appended"] == 0
    assert len(stored) == 1
    assert stored.iloc[0]["game_id"] == "g1"
    assert stored.iloc[0]["edge_reliability_status"] == "SUPPORTED_SUBGROUP"


def test_clean_forward_entries_uses_first_complete_pre_kickoff_observation():
    frame = pd.DataFrame(
        [
            _edge_row(
                snapshot_at="2026-10-10T15:00:00Z",
                quote_at="2026-10-10T15:01:00Z",
                book="Book A",
            ),
            _edge_row(
                snapshot_at="2026-10-10T16:00:00Z",
                quote_at="2026-10-10T15:59:00Z",
                book="Book B",
                odds=-108,
            ),
            _edge_row(
                snapshot_at="2026-10-10T17:00:00Z",
                quote_at="2026-10-10T16:59:00Z",
                book="Book C",
                odds=-105,
            ),
            _edge_row(
                snapshot_at="2026-10-10T18:01:00Z",
                quote_at="2026-10-10T17:59:00Z",
                book="Book D",
                odds=-102,
            ),
        ]
    )

    clean = _clean_forward_entries(frame)

    assert len(clean) == 1
    entry = clean.iloc[0]
    assert entry["snapshot_at"] == "2026-10-10T16:00:00Z"
    assert entry["book"] == "Book B"
    assert entry["odds"] == -108


class _FakeClient:
    def season_frame(self, season):
        assert season == 2026
        return pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "completed": True,
                    "home_points": 24,
                    "away_points": 21,
                }
            ]
        )


def test_grade_forward_edge_history_uses_flat_units_and_pre_kickoff_close(tmp_path):
    history = tmp_path / "edge_candidate_snapshots.csv"
    markets = tmp_path / "market_snapshots.csv"
    reports = tmp_path / "reports"

    pd.DataFrame(
        [
            _edge_row(
                snapshot_at="2026-10-10T16:00:00Z",
                quote_at="2026-10-10T15:59:00Z",
            )
        ]
    ).to_csv(history, index=False)

    pd.DataFrame(
        [
            {
                "game_id": "g1",
                "captured_at": "2026-10-10T17:45:00Z",
                "consensus_home_spread": -6.5,
                "market_book_count": 5,
            }
        ]
    ).to_csv(markets, index=False)

    report = grade_edge_forward_history(
        _FakeClient(),
        history,
        markets,
        reports,
    )

    graded = pd.read_csv(reports / "edge_forward_graded.csv")
    persisted = json.loads((reports / "edge_forward_performance.json").read_text())

    # Away +7.5 against a 3-point home win covers.
    assert len(graded) == 1
    assert graded.iloc[0]["result"] == 1
    assert abs(graded.iloc[0]["profit"] - (100 / 110)) < 1e-9
    # Close was Away +6.5, so entry beat close by one point.
    assert graded.iloc[0]["execution_clv"] == 1.0
    assert report["graded_bets"] == 1
    assert report["status"] == "EARLY_SAMPLE"
    assert report["by_subgroup"]["underdog|away"]["wins"] == 1
    assert persisted["methodology"].startswith(
        "First provenance-complete pre-kickoff observation"
    )
