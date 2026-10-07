import json
import pandas as pd

from harbin.pro_market import select_best_market
from harbin.pipeline import _edge_evidence_html, _write_edge_html

from harbin.edge_regimes import (
    annotate_selected_regimes,
    build_edge_regime_report,
    current_edge_board,
    effective_edge_status,
    load_or_build_edge_regime_report,
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
            "best_away_spread_odds": -110,
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
            "edge_subgroup_holdout_season": "2025",
            "edge_subgroup_holdout_bets": 21,
            "edge_subgroup_holdout_roi": 0.727,
            "edge_selection_override": True,
        }
    )

    html = _edge_evidence_html(row)

    assert "HOLDOUT CONFIRMED favorite|home" in html
    assert "6-8 parent" in html
    assert "59 full-sample bets" in html
    assert "40.8% ROI" in html
    assert "2025 holdout 21 bets / 72.7% ROI" in html
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
    assert favorite_home["validation_design"] == "latest_season_holdout"
    assert favorite_home["discovery_seasons"] == ["2023", "2024"]
    assert favorite_home["holdout_season"] == "2025"
    assert favorite_home["holdout_bets"] == 30
    assert favorite_home["holdout_roi"] > 0
    assert favorite_home["holdout_confirmed"] is True
    assert effective_edge_status(parent, favorite_home) == "SUPPORTED_SUBGROUP"

    assert underdog_home["status"] == "CONTRAINDICATED_SUBGROUP"
    assert underdog_home["profitable_seasons"] == 0
    assert underdog_home["roi"] < 0
    assert underdog_home["holdout_season"] == "2025"
    assert underdog_home["holdout_roi"] < 0
    assert underdog_home["holdout_confirmed"] is True
    assert effective_edge_status(parent, underdog_home) == "CONTRAINDICATED_SUBGROUP"


def test_positive_discovery_that_fails_latest_season_is_not_supported():
    rows = []
    rows += _role_rows(6.5, 2023, 30, 20, "favorite", "home", clv=0.03)
    rows += _role_rows(6.5, 2024, 30, 20, "favorite", "home", clv=0.03)
    rows += _role_rows(6.5, 2025, 30, 13, "favorite", "home", clv=0.03)
    for season in (2023, 2024, 2025):
        rows += _role_rows(6.5, season, 30, 20, "underdog", "away", clv=0.03)

    report = build_edge_regime_report(pd.DataFrame(rows))
    parent = match_edge_regime(report, "spread", 6.5)
    subgroup = match_edge_subgroup(
        report, "spread", 6.5, "Home", -3.5, "Home", "Away"
    )

    assert parent["status"] == "PERSISTENT_CANDIDATE"
    assert subgroup["discovery_roi"] > 0.05
    assert subgroup["holdout_season"] == "2025"
    assert subgroup["holdout_bets"] == 30
    assert subgroup["holdout_roi"] < 0
    assert subgroup["status"] == "INCONCLUSIVE_SUBGROUP"
    assert subgroup["holdout_confirmed"] is False
    assert effective_edge_status(parent, subgroup) == "PERSISTENT_PARENT_ONLY"


def test_negative_discovery_with_positive_holdout_is_not_contraindicated():
    rows = []
    rows += _role_rows(6.5, 2023, 30, 13, "underdog", "home", clv=0.02)
    rows += _role_rows(6.5, 2024, 30, 13, "underdog", "home", clv=0.02)
    rows += _role_rows(6.5, 2025, 30, 20, "underdog", "home", clv=0.02)
    for season in (2023, 2024, 2025):
        rows += _role_rows(6.5, season, 30, 20, "favorite", "home", clv=0.03)

    report = build_edge_regime_report(pd.DataFrame(rows))
    parent = match_edge_regime(report, "spread", 6.5)
    subgroup = match_edge_subgroup(
        report, "spread", 6.5, "Home", 7.5, "Home", "Away"
    )

    assert parent["status"] == "PERSISTENT_CANDIDATE"
    assert subgroup["discovery_roi"] < -0.03
    assert subgroup["holdout_roi"] > 0
    assert subgroup["status"] == "INCONCLUSIVE_SUBGROUP"
    assert subgroup["holdout_confirmed"] is False
    assert effective_edge_status(parent, subgroup) == "PERSISTENT_PARENT_ONLY"


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
    assert selected["edge_subgroup_holdout_season"] == "2025"
    assert selected["edge_subgroup_holdout_bets"] == 30
    assert selected["edge_subgroup_holdout_roi"] > 0
    assert selected["edge_subgroup_holdout_confirmed"] is True


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


def test_loader_rebuilds_stale_edge_regime_schema(tmp_path):
    rows = []
    for season in (2023, 2024, 2025):
        rows += _role_rows(6.5, season, 30, 21, "favorite", "home", clv=0.04)
        rows += _role_rows(6.5, season, 30, 13, "underdog", "home", clv=0.02)

    reports = tmp_path / "reports"
    reports.mkdir()
    pd.DataFrame(rows).to_csv(reports / "backtest_bets.csv", index=False)
    stale = {
        "schema_version": 1,
        "status": "TRACKING",
        "regimes": [
            {
                "market": "spread",
                "edge_band": "6-8",
                "min_edge": 6,
                "max_edge": 8,
                "status": "PERSISTENT_CANDIDATE",
            }
        ],
    }
    (reports / "edge_regimes.json").write_text(json.dumps(stale))

    rebuilt = load_or_build_edge_regime_report(reports)

    assert rebuilt["schema_version"] == 3
    parent = match_edge_regime(rebuilt, "spread", 6.5)
    assert parent["subgroups"]["favorite|home"]["status"] == "SUPPORTED_SUBGROUP"
    assert parent["subgroups"]["underdog|home"]["status"] == "CONTRAINDICATED_SUBGROUP"
    persisted = json.loads((reports / "edge_regimes.json").read_text())
    assert persisted["schema_version"] == 3


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


def test_contraindicated_raw_ev_leader_is_removed_from_selection(tmp_path):
    report = _subgroup_report()
    row = pd.Series(
        {
            "week": 6,
            "home_team": "California",
            "away_team": "Virginia Tech",
            "calibrated_home_probability": 0.55,
            "quant_best_ml_side": "California",
            "quant_best_ml_edge_pp": 4.5,
            "quant_best_ml_roi": 0.08,
            "best_home_ml": -110,
            "best_home_ml_book": "Book ML",
            "best_home_ml_quote_at": "2026-10-07T03:00:00Z",
            "cover_probability": 0.66,
            "spread_edge_pts": 6.5,
            "spread_team": "California",
            "spread_line": 11.5,
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
    assert selected["quant_side"] == "California"
    assert selected["selection_basis"] == "highest_raw_ev_after_contraindicated_veto"
    assert selected["edge_contraindicated_veto"] is True
    assert selected["edge_contraindicated_candidates"] == 1
    assert selected["raw_ev_best_market"] == "spread"
    assert selected["raw_ev_best_edge_status"] == "CONTRAINDICATED_SUBGROUP"
    assert selected["quant_ev"] < selected["raw_ev_best_ev"]


def test_all_contraindicated_candidates_force_pass(tmp_path):
    report = _subgroup_report()
    row = pd.Series(
        {
            "week": 6,
            "home_team": "California",
            "away_team": "Virginia Tech",
            "cover_probability": 0.66,
            "spread_edge_pts": 6.5,
            "spread_team": "California",
            "spread_line": 11.5,
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

    assert selected["quant_signal"] == "PASS"
    assert selected["quant_market"] is None
    assert selected["stake_units"] == 0.0
    assert selected["selection_basis"] == "contraindicated_edge_veto"
    assert selected["edge_contraindicated_veto"] is True
    assert selected["edge_contraindicated_candidates"] == 1
    assert "contraindicated" in selected["policy_block_reason"]


def test_current_edge_board_adds_current_price_evidence():
    report = _subgroup_report()
    frame = pd.DataFrame(
        [
            {
                "game_id": "g-price",
                "date": "2026-10-10T18:00:00Z",
                "away_team": "Away",
                "home_team": "Home",
                "spread_team": "Home",
                "spread_line": -3.5,
                "spread_odds": -110,
                "spread_edge_pts": 6.5,
                "cover_probability": 0.66,
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

    board = current_edge_board(frame, report, {"PERSISTENT_CANDIDATE"})

    assert len(board) == 1
    row = board.iloc[0]
    assert row["edge_reliability_status"] == "SUPPORTED_SUBGROUP"
    assert row["price_evidence_status"] == "CONFIRMED"
    assert row["historical_price_wilson_lower"] > row["current_break_even_probability"]
    assert row["conservative_price_margin"] > 0
    assert row["historical_fair_odds_lower_bound"] < -110


def test_overpriced_supported_subgroup_does_not_receive_historical_promotion(tmp_path):
    report = _subgroup_report()
    row = pd.Series(
        {
            "week": 6,
            "home_team": "Home",
            "away_team": "Away",
            "calibrated_home_probability": 0.55,
            "quant_best_ml_side": "Home",
            "quant_best_ml_edge_pp": 10.0,
            "quant_best_ml_roi": 0.25,
            "best_home_ml": 150,
            "best_home_ml_book": "Book ML",
            "best_home_ml_quote_at": "2026-10-07T03:00:00Z",
            "cover_probability": 0.85,
            "spread_edge_pts": 6.5,
            "spread_team": "Home",
            "spread_line": -3.5,
            "best_home_spread_odds": -300,
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


def test_supported_subgroup_at_plausible_price_can_still_outrank_raw_ev(tmp_path):
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
            "cover_probability": 0.78,
            "spread_edge_pts": 6.5,
            "spread_team": "Home",
            "spread_line": -3.5,
            "best_home_spread_odds": -180,
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
    assert selected["edge_regime_status"] == "SUPPORTED_SUBGROUP"
    assert selected["price_evidence_status"] in {"CONFIRMED", "PLAUSIBLE"}
    assert selected["edge_selection_override"] is True


def _watch_parent_supported_child_report():
    rows = []
    # Away underdogs: profitable discovery seasons and positive latest-season holdout.
    rows += _role_rows(3.5, 2023, 31, 19, "underdog", "away", clv=0.02)
    rows += _role_rows(3.5, 2024, 32, 18, "underdog", "away", clv=0.02)
    rows += _role_rows(3.5, 2025, 31, 17, "underdog", "away", clv=0.02)
    # Other rows keep the parent positive overall but make the latest parent season lose,
    # so the broad 3-4 band is WATCH rather than PERSISTENT_CANDIDATE.
    rows += _role_rows(3.5, 2023, 28, 15, "favorite", "home", clv=0.02)
    rows += _role_rows(3.5, 2024, 31, 17, "favorite", "home", clv=0.02)
    rows += _role_rows(3.5, 2025, 41, 18, "favorite", "home", clv=0.02)
    return build_edge_regime_report(pd.DataFrame(rows))


def test_holdout_supported_child_can_stand_under_watch_parent():
    report = _watch_parent_supported_child_report()
    parent = match_edge_regime(report, "spread", 3.5)
    child = match_edge_subgroup(
        report, "spread", 3.5, "Away", 7.5, "Home", "Away"
    )

    assert parent["status"] == "WATCH"
    assert child["status"] == "SUPPORTED_SUBGROUP"
    assert child["discovery_roi"] > 0.05
    assert child["holdout_roi"] > 0
    assert child["holdout_confirmed"] is True
    assert effective_edge_status(parent, child) == "SUPPORTED_SUBGROUP"


def test_supported_child_under_unsupported_parent_is_not_promoted():
    report = _watch_parent_supported_child_report()
    parent = dict(match_edge_regime(report, "spread", 3.5))
    parent["status"] = "UNSUPPORTED"
    child = match_edge_subgroup(
        report, "spread", 3.5, "Away", 7.5, "Home", "Away"
    )

    assert child["status"] == "SUPPORTED_SUBGROUP"
    assert effective_edge_status(parent, child) == "UNSUPPORTED"


def test_watch_parent_supported_child_is_visible_on_watch_board():
    report = _watch_parent_supported_child_report()
    frame = pd.DataFrame(
        [
            {
                "game_id": "g-watch-child",
                "date": "2026-10-10T18:00:00Z",
                "away_team": "Away",
                "home_team": "Home",
                "spread_team": "Away",
                "spread_line": 7.5,
                "spread_odds": -108,
                "spread_edge_pts": 3.5,
                "cover_probability": 0.59,
                "spread_badge": "LEAN",
                "spread_book": "Book A",
                "spread_quote_at": "2026-10-10T16:00:00Z",
                "quant_market": "spread",
                "quant_side": "Away",
                "quant_signal": "LEAN",
                "quant_ev": 0.13,
            }
        ]
    )

    board = current_edge_board(frame, report, {"WATCH"})

    assert len(board) == 1
    row = board.iloc[0]
    assert row["regime_status"] == "WATCH"
    assert row["edge_reliability_status"] == "SUPPORTED_SUBGROUP"
    assert row["subgroup_key"] == "underdog|away"
    assert row["subgroup_holdout_confirmed"] is True


def test_watch_parent_supported_child_can_outrank_unsupported_raw_ev(tmp_path):
    report = _watch_parent_supported_child_report()
    row = pd.Series(
        {
            "week": 6,
            "home_team": "Home",
            "away_team": "Away",
            "calibrated_home_probability": 0.58,
            "quant_best_ml_side": "Home",
            "quant_best_ml_edge_pp": 12.0,
            "quant_best_ml_roi": 0.28,
            "best_home_ml": 120,
            "best_home_ml_book": "Book ML",
            "best_home_ml_quote_at": "2026-10-07T03:00:00Z",
            "cover_probability": 0.59,
            "spread_edge_pts": 3.5,
            "spread_team": "Away",
            "spread_line": 7.5,
            "best_away_spread_odds": -108,
            "best_away_spread_book": "Book Spread",
            "best_away_spread_quote_at": "2026-10-07T03:00:00Z",
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
    assert selected["quant_side"] == "Away"
    assert selected["edge_regime_parent_status"] == "WATCH"
    assert selected["edge_regime_status"] == "SUPPORTED_SUBGROUP"
    assert selected["edge_subgroup_key"] == "underdog|away"
    assert selected["selection_basis"] == "supported_edge_subgroup"
    assert selected["price_evidence_status"] in {"CONFIRMED", "PLAUSIBLE"}
