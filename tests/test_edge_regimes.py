import pandas as pd

from harbin.pro_market import select_best_market
from harbin.pipeline import _edge_evidence_html, _write_edge_html

from harbin.edge_regimes import (
    annotate_selected_regimes,
    build_edge_regime_report,
    current_edge_board,
    effective_edge_status,
    match_edge_regime,
    match_edge_subgroup,
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


def _role_rows(edge, season, bets, wins, role, location, clv=0.02):
    rows = _rows("spread", edge, season, bets, wins, clv=clv)
    for row in rows:
        row["market_role"] = role
        row["side_location"] = location
    return rows


def _subgroup_report():
    rows = []
    for season in (2023, 2024, 2025):
        rows += _role_rows(6.5, season, 30, 21, "favorite", "home", clv=0.04)
        rows += _role_rows(6.5, season, 30, 13, "underdog", "home", clv=0.02)
    return build_edge_regime_report(pd.DataFrame(rows))


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

    assert out.iloc[0]["edge_regime_status"] == "PERSISTENT_PARENT_ONLY"
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
    assert supported["selection_basis"] == "persistent_parent_regime"
    assert supported["edge_selection_override"] is True
    assert supported["raw_ev_best_market"] == "moneyline"
    assert supported["edge_regime_status"] == "PERSISTENT_PARENT_ONLY"
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


def test_supported_edge_label_is_explicit_about_subgroup_evidence():
    row = pd.Series(
        {
            "edge_regime_status": "SUPPORTED_SUBGROUP",
            "edge_regime_band": "6-8",
            "edge_subgroup_key": "favorite|home",
            "edge_subgroup_bets": 59,
            "edge_subgroup_roi": 0.408,
            "edge_subgroup_profitable_seasons": 3,
            "edge_subgroup_season_count": 3,
            "edge_selection_override": True,
        }
    )

    html = _edge_evidence_html(row)

    assert "SUPPORTED favorite|home" in html
    assert "6-8 parent" in html
    assert "59 subgroup bets" in html
    assert "40.8% ROI" in html
    assert "3/3 profitable seasons" in html
    assert "selected over raw-EV alternative" in html


def test_spread_subgroups_separate_supported_and_contraindicated_children():
    report = _subgroup_report()
    parent = match_edge_regime(report, "spread", 6.5)
    favorite_home = match_edge_subgroup(
        report, "spread", 6.5, "Home", -3.5, "Home", "Away"
    )
    underdog_home = match_edge_subgroup(
        report, "spread", 6.5, "Home", 7.5, "Home", "Away"
    )

    assert parent["status"] == "PERSISTENT_CANDIDATE"
    assert favorite_home["status"] == "SUPPORTED_SUBGROUP"
    assert favorite_home["profitable_seasons"] == 3
    assert favorite_home["roi"] > 0.05
    assert effective_edge_status(parent, favorite_home) == "SUPPORTED_SUBGROUP"

    assert underdog_home["status"] == "CONTRAINDICATED_SUBGROUP"
    assert underdog_home["profitable_seasons"] == 0
    assert underdog_home["roi"] < 0
    assert effective_edge_status(parent, underdog_home) == "CONTRAINDICATED_SUBGROUP"


def test_contraindicated_child_cannot_receive_persistent_override(tmp_path):
    report = _subgroup_report()
    row = pd.Series(
        {
            "week": 6,
            "home_team": "Home",
            "away_team": "Away",
            "calibrated_home_probability": 0.40,
            "quant_best_ml_side": "Away",
            "quant_best_ml_edge_pp": 12.0,
            "quant_best_ml_roi": 0.30,
            "best_away_ml": 110,
            "best_away_ml_book": "Book ML",
            "best_away_ml_quote_at": "2026-10-07T03:00:00Z",
            "cover_probability": 0.63,
            "spread_edge_pts": 6.5,
            "spread_team": "Home",
            "spread_line": 7.5,
            "best_home_spread_odds": -110,
            "best_home_spread_book": "Book Spread",
            "best_home_spread_quote_at": "2026-10-07T03:00:00Z",
        }
    )
    missing_policy = tmp_path / "missing_policy.json"

    selected = select_best_market(
        row,
        risk_multiplier=1.0,
        policy_path=str(missing_policy),
        edge_report=report,
    )

    assert selected["quant_market"] == "moneyline"
    assert selected["selection_basis"] == "highest_raw_ev"
    assert selected["edge_selection_override"] is False


def test_supported_subgroup_outranks_higher_raw_ev_market(tmp_path):
    report = _subgroup_report()
    row = pd.Series(
        {
            "week": 6,
            "home_team": "Home",
            "away_team": "Away",
            "calibrated_home_probability": 0.60,
            "quant_best_ml_side": "Home",
            "quant_best_ml_edge_pp": 12.0,
            "quant_best_ml_roi": 0.45,
            "best_home_ml": 130,
            "best_home_ml_book": "Book ML",
            "best_home_ml_quote_at": "2026-10-07T03:00:00Z",
            "cover_probability": 0.66,
            "spread_edge_pts": 6.5,
            "spread_team": "Home",
            "spread_line": -3.5,
            "best_home_spread_odds": -110,
            "best_home_spread_book": "Book Spread",
            "best_home_spread_quote_at": "2026-10-07T03:00:00Z",
        }
    )
    missing_policy = tmp_path / "missing_policy.json"

    selected = select_best_market(
        row,
        risk_multiplier=1.0,
        policy_path=str(missing_policy),
        edge_report=report,
    )

    assert selected["quant_market"] == "spread"
    assert selected["quant_side"] == "Home"
    assert selected["selection_basis"] == "supported_edge_subgroup"
    assert selected["edge_selection_override"] is True
    assert selected["edge_regime_status"] == "SUPPORTED_SUBGROUP"
    assert selected["edge_subgroup_key"] == "favorite|home"
    assert selected["edge_subgroup_status"] == "SUPPORTED_SUBGROUP"


def test_current_edge_board_excludes_contraindicated_child_by_default():
    report = _subgroup_report()
    frame = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "date": "2026-10-10T18:00:00Z",
                "away_team": "Away",
                "home_team": "Home",
                "spread_team": "Home",
                "spread_line": 7.5,
                "spread_odds": -110,
                "spread_edge_pts": 6.5,
                "cover_probability": 0.63,
                "spread_badge": "STRONG",
                "spread_book": "Book A",
                "spread_quote_at": "2026-10-10T16:00:00Z",
                "quant_market": "spread",
                "quant_side": "Home",
                "quant_signal": "STRONG",
                "quant_ev": 0.20,
            }
        ]
    )

    supported = current_edge_board(frame, report, {"PERSISTENT_CANDIDATE"})
    all_parent = current_edge_board(
        frame,
        report,
        {"PERSISTENT_CANDIDATE"},
        include_contraindicated=True,
    )

    assert supported.empty
    assert len(all_parent) == 1
    assert all_parent.iloc[0]["edge_reliability_status"] == "CONTRAINDICATED_SUBGROUP"
    assert all_parent.iloc[0]["subgroup_key"] == "underdog|home"


def test_supported_edge_board_contains_forward_validation_warning(tmp_path):
    edges = pd.DataFrame(
        [
            {
                "away_team": "Away",
                "home_team": "Home",
                "market": "spread",
                "side": "Away",
                "line": 7.5,
                "odds": -110,
                "probability": 0.62,
                "edge": 6.5,
                "ev": 0.18,
                "historical_bets": 231,
                "historical_win_rate": 0.6096,
                "historical_roi": 0.1617,
                "profitable_seasons": 3,
                "season_count": 3,
            }
        ]
    )
    path = tmp_path / "edge.html"

    _write_edge_html(edges, path, "Oct 7, 2026 · 5:30 AM CT")
    html = path.read_text()

    assert "SUPPORTED EDGE BOARD" in html
    assert "spread model-market disagreement of 6–8 points" in html
    assert "clean forward validation is still required" in html
    assert "Away @ Home" in html
    assert "16.2%" in html
