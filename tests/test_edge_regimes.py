import pandas as pd

from harbin.pro_market import select_best_market

from harbin.edge_regimes import (
    annotate_selected_regimes,
    build_edge_regime_report,
    current_edge_board,
    match_edge_regime,
)


def _rows(market, edge, season, bets, wins, clv=0.02):
    rows = []
    for idx in range(bets):
        win = idx < wins
        rows.append(
            {
                "market": market,
                "edge": edge,
                "season": season,
                "week": (idx % 14) + 1,
                "result": 1 if win else -1,
                "profit": 0.9090909091 if win else -1.0,
                "clv": clv,
            }
        )
    return rows


def test_persistent_regime_requires_cross_season_profitability():
    rows = []
    rows += _rows("spread", 6.5, 2023, 60, 36)
    rows += _rows("spread", 6.5, 2024, 60, 36)
    rows += _rows("spread", 6.5, 2025, 60, 36)
    rows += _rows("total", 6.5, 2023, 60, 36)
    rows += _rows("total", 6.5, 2024, 60, 36)
    rows += _rows("total", 6.5, 2025, 60, 24)

    report = build_edge_regime_report(pd.DataFrame(rows))

    spread = match_edge_regime(report, "spread", 6.7)
    total = match_edge_regime(report, "total", 6.7)

    assert spread["status"] == "PERSISTENT_CANDIDATE"
    assert spread["profitable_seasons"] == 3
    assert spread["min_season_bets"] == 60
    assert total["status"] != "PERSISTENT_CANDIDATE"


def test_current_edge_board_surfaces_supported_market_even_when_not_selected():
    rows = []
    rows += _rows("spread", 6.5, 2023, 60, 36)
    rows += _rows("spread", 6.5, 2024, 60, 36)
    rows += _rows("spread", 6.5, 2025, 60, 36)
    report = build_edge_regime_report(pd.DataFrame(rows))

    frame = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "date": "2026-10-10T18:00:00Z",
                "away_team": "Away",
                "home_team": "Home",
                "spread_team": "Away",
                "spread_line": 7.5,
                "spread_odds": -110,
                "spread_edge_pts": 6.5,
                "cover_probability": 0.62,
                "spread_badge": "STRONG",
                "spread_book": "Book A",
                "spread_quote_at": "2026-10-10T16:00:00Z",
                "quant_market": "moneyline",
                "quant_side": "Away",
                "quant_signal": "BET",
                "quant_ev": 0.20,
            }
        ]
    )

    board = current_edge_board(frame, report, {"PERSISTENT_CANDIDATE"})

    assert len(board) == 1
    assert board.iloc[0]["market"] == "spread"
    assert bool(board.iloc[0]["currently_selected"]) is False
    assert board.iloc[0]["regime_status"] == "PERSISTENT_CANDIDATE"


def test_selected_pick_annotation_carries_regime_evidence():
    rows = []
    rows += _rows("spread", 6.5, 2023, 60, 36)
    rows += _rows("spread", 6.5, 2024, 60, 36)
    rows += _rows("spread", 6.5, 2025, 60, 36)
    report = build_edge_regime_report(pd.DataFrame(rows))

    frame = pd.DataFrame(
        [
            {
                "quant_market": "spread",
                "quant_edge": 6.5,
                "quant_signal": "STRONG",
            }
        ]
    )
    out = annotate_selected_regimes(frame, report)

    assert out.iloc[0]["edge_regime_status"] == "PERSISTENT_CANDIDATE"
    assert bool(out.iloc[0]["edge_regime_candidate"]) is True
    assert out.iloc[0]["edge_regime_profitable_seasons"] == 3


def test_persistent_regime_outranks_higher_raw_ev_unsupported_market(tmp_path):
    rows = []
    rows += _rows("spread", 6.5, 2023, 60, 36)
    rows += _rows("spread", 6.5, 2024, 60, 36)
    rows += _rows("spread", 6.5, 2025, 60, 36)
    report = build_edge_regime_report(pd.DataFrame(rows))

    row = pd.Series(
        {
            "week": 6,
            "home_team": "UTSA",
            "away_team": "South Florida",
            "calibrated_home_probability": 0.46,
            "quant_best_ml_side": "South Florida",
            "quant_best_ml_edge_pp": 24.48,
            "quant_best_ml_roi": 0.782,
            "best_away_ml": 230,
            "best_away_ml_book": "Book ML",
            "best_away_ml_quote_at": "2026-10-07T03:00:00Z",
            "cover_probability": 0.6843,
            "spread_edge_pts": 7.816,
            "spread_team": "South Florida",
            "spread_line": 7.5,
            "best_away_spread_odds": -122,
            "best_away_spread_book": "Book Spread",
            "best_away_spread_quote_at": "2026-10-07T03:00:00Z",
        }
    )
    missing_policy = tmp_path / "missing_policy.json"

    raw = select_best_market(
        row,
        risk_multiplier=1.0,
        policy_path=str(missing_policy),
        edge_report=None,
    )
    supported = select_best_market(
        row,
        risk_multiplier=1.0,
        policy_path=str(missing_policy),
        edge_report=report,
    )

    assert raw["quant_market"] == "moneyline"
    assert raw["quant_ev"] > supported["quant_ev"]
    assert supported["quant_market"] == "spread"
    assert supported["quant_side"] == "South Florida"
    assert supported["selection_basis"] == "persistent_edge_regime"
    assert supported["edge_selection_override"] is True
    assert supported["raw_ev_best_market"] == "moneyline"
    assert supported["edge_regime_status"] == "PERSISTENT_CANDIDATE"
    assert supported["edge_regime_band"] == "6-8"
    assert supported["edge_regime_profitable_seasons"] == 3


def test_persistent_selection_does_not_inflate_probability_ev_or_stake(tmp_path):
    rows = []
    rows += _rows("spread", 6.5, 2023, 60, 36)
    rows += _rows("spread", 6.5, 2024, 60, 36)
    rows += _rows("spread", 6.5, 2025, 60, 36)
    report = build_edge_regime_report(pd.DataFrame(rows))

    row = pd.Series(
        {
            "week": 6,
            "home_team": "Home",
            "away_team": "Away",
            "cover_probability": 0.62,
            "spread_edge_pts": 6.5,
            "spread_team": "Away",
            "spread_line": 7.5,
            "best_away_spread_odds": -110,
            "best_away_spread_book": "Book A",
            "best_away_spread_quote_at": "2026-10-07T03:00:00Z",
        }
    )
    missing_policy = tmp_path / "missing_policy.json"

    raw = select_best_market(
        row,
        risk_multiplier=0.8,
        policy_path=str(missing_policy),
        edge_report=None,
    )
    supported = select_best_market(
        row,
        risk_multiplier=0.8,
        policy_path=str(missing_policy),
        edge_report=report,
    )

    assert supported["quant_probability"] == raw["quant_probability"] == 0.62
    assert supported["quant_ev"] == raw["quant_ev"]
    assert supported["stake_units"] == raw["stake_units"]
