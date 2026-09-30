import pandas as pd

from harbin.context import (
    _context_quality,
    normalize_injury_rows,
    summarize_injury_rows,
    summarize_roster_rows,
    weather_risk,
)
from harbin.health import build_health_report


def test_latest_injury_state_supersedes_old_out_report():
    df = pd.DataFrame([
        {"team_id": 1, "athlete_id": 9, "athlete_name": "QB One", "position": "QB", "status": "Out", "date": "2026-09-20T12:00:00Z", "week": 4},
        {"team_id": 1, "athlete_id": 9, "athlete_name": "QB One", "position": "QB", "status": "Active", "date": "2026-09-28T12:00:00Z", "week": 5},
    ])
    latest = normalize_injury_rows(df, as_of="2026-09-29T12:00:00Z", target_week=5)
    summary = summarize_injury_rows(latest)
    assert len(latest) == 1
    assert summary["injury_count"] == 0
    assert summary["qb_injury_risk"] == 0


def test_future_injury_reports_and_future_weeks_are_excluded():
    df = pd.DataFrame([
        {"team_id": 1, "athlete_id": 7, "position": "QB", "status": "Questionable", "date": "2026-09-28T12:00:00Z", "week": 5},
        {"team_id": 1, "athlete_id": 7, "position": "QB", "status": "Out", "date": "2026-10-04T12:00:00Z", "week": 6},
    ])
    latest = normalize_injury_rows(df, as_of="2026-09-29T12:00:00Z", target_week=5)
    summary = summarize_injury_rows(latest)
    assert len(latest) == 1
    assert 0 < summary["qb_injury_risk"] < 0.5


def test_old_nonseason_injury_decays_but_season_ending_persists():
    df = pd.DataFrame([
        {"team_id": 1, "athlete_id": 1, "position": "WR", "status": "Out", "date": "2026-07-01T00:00:00Z", "week": 1},
        {"team_id": 1, "athlete_id": 2, "position": "LB", "status": "Out for season", "date": "2026-07-01T00:00:00Z", "week": 1},
    ])
    latest = normalize_injury_rows(df, as_of="2026-09-29T00:00:00Z", target_week=5)
    by_player = dict(zip(latest["_athlete_id"].astype(str), latest["_severity"]))
    assert by_player["1"] == 0
    assert by_player["2"] == 1


def test_roster_summary_tracks_inactive_share_and_qb_depth():
    df = pd.DataFrame([
        {"athlete_id": 1, "position": "QB", "active": True},
        {"athlete_id": 2, "position": "QB", "active": False},
        {"athlete_id": 3, "position": "WR", "active": True},
        {"athlete_id": 4, "position": "OL", "active": True},
    ])
    s = summarize_roster_rows(df)
    assert s["roster_count"] == 4
    assert s["roster_inactive_count"] == 1
    assert s["qb_roster_count"] == 2
    assert s["active_qb_count"] == 1
    assert s["roster_availability_risk"] >= 0.08


def test_indoor_weather_is_zero_risk():
    severe = {"wind_mph": 40, "wind_gust_mph": 60, "precip_probability": 100, "temperature_f": 20}
    assert weather_risk(severe, indoor=True) == 0
    assert weather_risk(severe, indoor=False) > 0.5


def test_context_quality_fails_closed_when_core_component_missing():
    assert _context_quality(True, True, True, True, 1.0) == 1.0
    assert _context_quality(True, False, True, True, 1.0) == 0.75
    assert _context_quality(True, True, True, True, 0.65) == 0.65


def test_health_uses_context_quality_not_source_name(tmp_path):
    meta = {
        "market_coverage": {"games": 10, "moneyline": 10, "spread": 10, "total": 10},
        "advanced_features": {"dynamic_coverage": 1.0, "live_coverage": 1.0},
        "metrics": {"margin_baseline_mae": 13, "margin_mae": 12.5, "margin_walkforward_folds": [{"mae": 12, "baseline_mae": 13}], "win_brier": .17, "win_ece": .04},
        "market_intelligence": {"multi_book_coverage": 1.0},
        "current_context": {"sources": ["injuries", "weather"], "quality_score": 0.4, "core_coverage": 0.4},
    }
    h = build_health_report(meta, reports_dir=tmp_path)
    assert h["components"]["current_context"] == 40.0
    assert any("context" in x for x in h["blockers"])
