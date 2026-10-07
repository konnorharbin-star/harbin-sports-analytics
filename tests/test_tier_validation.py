import pandas as pd

from harbin.tier_validation import (
    build_tier_performance,
    expected_display_tier,
    refresh_display_market_tiers,
    summarize_tier_rows,
)


def test_refresh_display_badges_uses_post_line_shopping_prices():
    row = {
        "home_team": "Home",
        "away_team": "Away",
        "calibrated_home_probability": 0.60,
        "ml_team": "Home",
        "best_home_ml": -110,
        "home_ml": -150,
        "best_home_ml_book": "Book A",
        "best_home_ml_quote_at": "2026-10-06T18:00:00Z",
        "model_margin_home": 7.0,
        "spread_team": "Home",
        "spread_line": -3.0,
        "best_home_spread_book": "Book B",
        "best_home_spread_quote_at": "2026-10-06T18:00:00Z",
        "model_total": 55.0,
        "market_total": 50.0,
        "total_dir": "O",
        "best_over_book": "Book C",
        "best_over_quote_at": "2026-10-06T18:00:00Z",
    }

    refreshed = refresh_display_market_tiers(row)

    assert refreshed["ml_odds"] == -110
    assert refreshed["ml_badge"] == "STRONG"
    assert refreshed["ml_book"] == "Book A"
    assert refreshed["spread_badge"] == "BET"
    assert refreshed["spread_book"] == "Book B"
    assert refreshed["total_badge"] == "BET"
    assert refreshed["total_book"] == "Book C"


def test_tier_summary_is_flat_one_unit_and_calibrated():
    rows = []
    for idx in range(120):
        win = idx < 80
        rows.append(
            {
                "result": 1 if win else -1,
                "flat_profit": 0.9090909091 if win else -1.0,
                "execution_odds": -110,
                "model_probability": 0.66,
                "model_edge": 5.0,
                "model_ev": 0.10,
                "execution_clv": 0.5,
            }
        )

    summary = summarize_tier_rows(pd.DataFrame(rows))

    assert summary["graded_bets"] == 120
    assert summary["wins"] == 80
    assert summary["losses"] == 40
    assert abs(summary["hit_rate"] - (80 / 120)) < 1e-12
    assert summary["flat_units"] > 30
    assert summary["flat_roi"] > 0.20
    assert abs(summary["calibration_gap"] - (0.66 - 80 / 120)) < 1e-12
    assert summary["brier"] < 0.23
    assert summary["status"] == "VALIDATED"
    assert summary["validated"] is True


def test_market_tier_matrix_keeps_markets_separate():
    frame = pd.DataFrame(
        [
            {
                "market": "spread",
                "tier": "STRONG",
                "result": 1,
                "flat_profit": 0.9090909091,
                "execution_odds": -110,
                "model_probability": 0.60,
                "model_edge": 6.0,
                "model_ev": 0.08,
                "execution_clv": 1.0,
            },
            {
                "market": "total",
                "tier": "STRONG",
                "result": -1,
                "flat_profit": -1.0,
                "execution_odds": -110,
                "model_probability": 0.60,
                "model_edge": 7.5,
                "model_ev": 0.08,
                "execution_clv": -0.5,
            },
            {
                "market": "moneyline",
                "tier": "BET",
                "result": 1,
                "flat_profit": 1.2,
                "execution_odds": 120,
                "model_probability": 0.52,
                "model_edge": 4.0,
                "model_ev": 0.14,
                "execution_clv": 0.02,
            },
        ]
    )

    report = build_tier_performance(frame)

    assert report["graded_tier_bets"] == 3
    assert report["by_market_tier"]["spread"]["STRONG"]["wins"] == 1
    assert report["by_market_tier"]["total"]["STRONG"]["losses"] == 1
    assert report["by_market_tier"]["moneyline"]["BET"]["flat_units"] == 1.2
    assert len(report["matrix"]) == 9


def test_expected_display_tier_detects_legacy_moneyline_mismatch():
    row = {
        "home_team": "Home",
        "away_team": "Away",
        "quant_side": "Home",
        "execution_odds": -200,
        "model_probability": 0.55,
    }
    assert expected_display_tier("moneyline", row) == ""


def test_clean_validation_excludes_inconsistent_or_unverified_rows():
    frame = pd.DataFrame(
        [
            {
                "market": "spread",
                "tier": "BET",
                "result": 1,
                "flat_profit": 0.9090909091,
                "execution_odds": -110,
                "model_probability": 0.60,
                "model_edge": 4.5,
                "model_ev": 0.10,
                "execution_clv": 0.5,
                "tier_consistent": True,
                "price_verified": True,
                "validation_eligible": True,
            },
            {
                "market": "spread",
                "tier": "STRONG",
                "result": -1,
                "flat_profit": -1.0,
                "execution_odds": -110,
                "model_probability": 0.75,
                "model_edge": 11.0,
                "model_ev": 0.40,
                "execution_clv": -1.0,
                "tier_consistent": False,
                "price_verified": False,
                "validation_eligible": False,
            },
        ]
    )

    report = build_tier_performance(frame)

    assert report["graded_tier_bets"] == 2
    assert report["validation_eligible_bets"] == 1
    assert report["excluded_from_validation"] == 1
    assert report["inconsistent_badge_rows"] == 1
    assert report["clean_by_market_tier"]["spread"]["BET"]["graded_bets"] == 1
    assert report["clean_by_market_tier"]["spread"]["STRONG"]["graded_bets"] == 0
