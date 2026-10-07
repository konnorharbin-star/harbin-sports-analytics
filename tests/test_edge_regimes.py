import pandas as pd

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
