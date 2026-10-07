from __future__ import annotations

import json

import pandas as pd

from harbin.edge_forward import (
    apply_forward_validation,
    append_edge_shadow_candidates,
    grade_edge_shadow_history,
)


class _Client:
    def season_frame(self, season):
        assert season == 2026
        return pd.DataFrame(
            [
                {
                    "game_id": "g1",
                    "completed": True,
                    "home_points": 27,
                    "away_points": 20,
                }
            ]
        )


def _board():
    return pd.DataFrame(
        [
            {
                "game_id": "g1",
                "date": "2026-10-10T18:00:00Z",
                "away_team": "Away",
                "home_team": "Home",
                "market": "spread",
                "side": "Home",
                "line": -2.5,
                "odds": -110,
                "probability": 0.62,
                "edge": 6.5,
                "ev": 0.18,
                "badge": "STRONG",
                "book": "Book A",
                "quote_at": "2026-10-10T15:59:00Z",
                "regime_status": "HISTORICAL_HYPOTHESIS",
                "regime_band": "6-8",
                "historical_bets": 231,
                "historical_win_rate": 0.6096,
                "historical_roi": 0.1617,
                "historical_avg_clv": 0.02,
                "profitable_seasons": 3,
                "season_count": 3,
                "verified_entry_bets": 0,
                "verified_entry_rate": 0.0,
            }
        ]
    )


def test_shadow_snapshot_is_first_verified_entry_only(tmp_path):
    path = tmp_path / "edge_shadow_snapshots.csv"
    board = _board()

    first = append_edge_shadow_candidates(
        board,
        path,
        captured_at="2026-10-10T16:00:00Z",
    )
    second = append_edge_shadow_candidates(
        board.assign(line=-1.5, odds=-105),
        path,
        captured_at="2026-10-10T16:30:00Z",
    )

    saved = pd.read_csv(path)

    assert first == 1
    assert second == 0
    assert len(saved) == 1
    assert saved.iloc[0]["line"] == -2.5
    assert saved.iloc[0]["odds"] == -110
    assert saved.iloc[0]["book"] == "Book A"
    assert saved.iloc[0]["regime_status_at_entry"] == "HISTORICAL_HYPOTHESIS"


def test_shadow_snapshot_rejects_future_quote_or_post_kickoff_capture(tmp_path):
    path = tmp_path / "edge_shadow_snapshots.csv"
    board = _board()

    future_quote = board.copy()
    future_quote.loc[0, "quote_at"] = "2026-10-10T16:01:00Z"
    after_kick = board.copy()

    assert (
        append_edge_shadow_candidates(
            future_quote,
            path,
            captured_at="2026-10-10T16:00:00Z",
        )
        == 0
    )
    assert (
        append_edge_shadow_candidates(
            after_kick,
            path,
            captured_at="2026-10-10T18:01:00Z",
        )
        == 0
    )
    assert not path.exists()


def test_edge_shadow_grader_uses_entry_price_and_pre_kickoff_close(tmp_path):
    history = tmp_path / "history"
    reports = tmp_path / "reports"
    history.mkdir()

    append_edge_shadow_candidates(
        _board(),
        history / "edge_shadow_snapshots.csv",
        captured_at="2026-10-10T16:00:00Z",
    )
    pd.DataFrame(
        [
            {
                "captured_at": "2026-10-10T17:59:00Z",
                "game_id": "g1",
                "consensus_home_spread": -3.5,
                "market_book_count": 6,
            }
        ]
    ).to_csv(history / "market_snapshots.csv", index=False)

    report = grade_edge_shadow_history(_Client(), history, reports)
    graded = pd.read_csv(reports / "edge_forward_graded.csv")

    assert report["graded_bets"] == 1
    assert report["regimes"][0]["status"] == "EARLY_SAMPLE"
    assert graded.iloc[0]["result"] == 1
    assert abs(graded.iloc[0]["profit"] - (100 / 110)) < 1e-9
    assert graded.iloc[0]["execution_clv"] == 1.0
    assert graded.iloc[0]["clv_source"] == "consensus_latest_pre_kickoff_snapshot"


def test_forward_validation_is_required_before_hypothesis_promotion():
    historical = {
        "schema_version": 2,
        "regimes": [
            {
                "market": "spread",
                "edge_band": "6-8",
                "status": "HISTORICAL_HYPOTHESIS",
            }
        ],
        "persistent_regimes": 0,
        "historical_hypothesis_regimes": 1,
    }
    early = {
        "graded_bets": 12,
        "validated_regimes": 0,
        "status": "TRACKING",
        "regimes": [
            {
                "market": "spread",
                "edge_band": "6-8",
                "status": "DEVELOPING",
                "graded_bets": 12,
                "roi": 0.08,
                "roi_ci_95": [None, None],
                "avg_execution_clv": 0.4,
            }
        ],
    }
    validated = json.loads(json.dumps(early))
    validated["graded_bets"] = 110
    validated["validated_regimes"] = 1
    validated["regimes"][0].update(
        {
            "status": "FORWARD_VALIDATED",
            "graded_bets": 110,
            "roi_ci_95": [0.01, 0.15],
        }
    )

    not_promoted = apply_forward_validation(historical, early)
    promoted = apply_forward_validation(historical, validated)

    assert not_promoted["regimes"][0]["status"] == "HISTORICAL_HYPOTHESIS"
    assert not_promoted["persistent_regimes"] == 0
    assert promoted["regimes"][0]["status"] == "PERSISTENT_CANDIDATE"
    assert promoted["regimes"][0]["promotion_basis"] == "verified_forward_validation"
    assert promoted["persistent_regimes"] == 1
