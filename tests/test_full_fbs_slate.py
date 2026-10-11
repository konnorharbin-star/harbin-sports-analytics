"""All-FBS weekly slate coverage and FBS/FCS wagering quarantine tests."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from harbin.data import SportsDataVerseClient, fbs_schedule_mask
from harbin.fbs_slate import apply_slate_scope, classify
from scripts.audit_schedule_coverage import audit


def rows():
    return pd.DataFrame([
        {
            "game_id": "1", "season": 2026, "week": 6,
            "start_date": "2026-10-11T17:00:00Z",
            "home_team": "Alabama", "away_team": "Georgia",
            "home_id": 1, "away_id": 2,
            "home_division": "FBS", "away_division": "FBS",
            "completed": False, "neutral_site": False,
        },
        {
            "game_id": "2", "season": 2026, "week": 6,
            "start_date": "2026-10-11T18:00:00Z",
            "home_team": "Auburn", "away_team": "FCS Team",
            "home_id": 3, "away_id": 4,
            "home_division": " FBS ", "away_division": "FCS",
            "completed": False, "neutral_site": False,
        },
        {
            "game_id": "3", "season": 2026, "week": 6,
            "start_date": "2026-10-11T19:00:00Z",
            "home_team": "FCS Team", "away_team": "Ole Miss",
            "home_id": 4, "away_id": 5,
            "home_division": "FCS", "away_division": "fbs",
            "completed": False, "neutral_site": False,
        },
        {
            "game_id": "4", "season": 2026, "week": 6,
            "start_date": "2026-10-11T20:00:00Z",
            "home_team": "FCS A", "away_team": "FCS B",
            "home_id": 6, "away_id": 7,
            "home_division": "FCS", "away_division": "FCS",
            "completed": False, "neutral_site": False,
        },
    ])


def test_fbs_schedule_includes_fbs_fcs_both_directions():
    f = rows()
    assert f.loc[fbs_schedule_mask(f), "game_id"].tolist() == ["1", "2", "3"]
    assert f.loc[fbs_schedule_mask(f, both=True), "game_id"].tolist() == ["1"]


def test_missing_division_fields_refuse_to_publish_unfiltered_games():
    with pytest.raises(ValueError, match="missing home_division"):
        fbs_schedule_mask(rows().drop(columns=["home_division"]))


def test_client_source_includes_fbs_fcs_not_fcs_fcs(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: rows())
    c = SportsDataVerseClient()
    data = c.season_frame(2026)
    assert data["game_id"].tolist() == ["1", "2", "3"]
    assert {g.game_id for g in map(c._row_game, data.to_dict("records"))} == {
        "1", "2", "3"
    }
    assert c._row_game(data.iloc[1]).away_division == "FCS"
    assert c._row_game(data.iloc[2]).away_division == "FBS"


def test_historical_training_keeps_original_fbs_fbs_population(monkeypatch):
    f = rows()
    f["completed"] = True
    f["home_points"] = [21, 50, 10, 11]
    f["away_points"] = [20, 7, 30, 20]
    c = SportsDataVerseClient()
    monkeypatch.setattr(c, "season_frame", lambda _: f.copy())
    history = c.history(2025, 2026, 7)
    assert [g.game_id for g in history] == ["1", "1"]


def games():
    return [
        SimpleNamespace(game_id="1", home_division="FBS", away_division="FBS"),
        SimpleNamespace(game_id="2", home_division="FBS", away_division="FCS"),
        SimpleNamespace(game_id="3", home_division="FCS", away_division="FBS"),
    ]


def test_projections_include_all_fbs_while_fcs_matchups_block_bets():
    pred = pd.DataFrame([
        {"game_id": "1", "quant_signal": "BET", "stake_units": 1.0,
         "model_margin_home": 3.2, "edge_regime_candidate": True},
        {"game_id": "2", "quant_signal": "BET", "stake_units": 1.0,
         "model_margin_home": 22.5, "edge_regime_candidate": True},
        {"game_id": "3", "quant_signal": "STRONG", "stake_units": 1.5,
         "model_margin_home": -19.5, "edge_regime_candidate": True},
    ])
    out = apply_slate_scope(pred, games())
    assert len(out) == 3
    assert out.loc[0, "quant_signal"] == "BET"
    assert out.loc[1:, "quant_signal"].tolist() == ["PASS", "PASS"]
    assert out.loc[1:, "stake_units"].tolist() == [0.0, 0.0]
    assert out.loc[1:, "edge_regime_candidate"].tolist() == [False, False]
    assert out.loc[1:, "fbs_model_validation"].tolist() == [
        "NO_BET_UNVALIDATED_OPPONENT_CLASS"
    ] * 2
    assert out.loc[1:, "model_margin_home"].tolist() == [22.5, -19.5]


def test_missing_scheduled_fbs_game_fails_closed():
    pred = pd.DataFrame({"game_id": ["1"]})
    with pytest.raises(ValueError, match="do not cover all upcoming"):
        apply_slate_scope(pred, games())
    with pytest.raises(ValueError, match="Missing projections"):
        apply_slate_scope(pd.DataFrame(), games())


def test_fcs_only_games_cannot_enter_model():
    with pytest.raises(ValueError, match="not verified to include an FBS"):
        classify(SimpleNamespace(
            home_division="FCS", away_division="FCS",
        ))


def test_duplicate_source_ids_fail_loudly():
    g = games()
    with pytest.raises(ValueError, match="Duplicate game IDs"):
        apply_slate_scope(pd.DataFrame({"game_id": ["1"]}), g + [g[0]])


def _write_model(tmp_path, ids):
    root = tmp_path
    out = root / "outputs"
    out.mkdir()
    (out / "cfb_model_2026_week6_metadata.json").write_text(json.dumps({
        "season": 2026, "week": 6,
        "pregame_filter": {"cutoff_utc": "2026-10-10T23:00:00Z"},
    }))
    pd.DataFrame({"game_id": ids, "fbs_matchup_scope": [
        "FBS_VS_FBS" if gid == "1" else "FBS_VS_NON_FBS_UNVALIDATED"
        for gid in ids
    ]}).to_csv(out / "cfb_model_2026_week6.csv", index=False)
    return root


def test_schedule_audit_fails_if_fbs_fcs_games_missing(monkeypatch, tmp_path):
    _write_model(tmp_path, ["1"])
    f = rows().iloc[:3].copy()
    f["season_type"] = "regular"
    monkeypatch.setattr(
        "scripts.audit_schedule_coverage.SportsDataVerseClient.season_frame",
        lambda *_: f.copy()
    )
    with pytest.raises(SystemExit, match="upcoming game missing"):
        audit(tmp_path)
    obj = json.loads(
        (tmp_path / "outputs" / "schedule_coverage_audit.json").read_text()
    )
    assert obj["total_scheduled_fbs_games"] == 3
    assert obj["fbs_vs_non_fbs_games"] == 2
    assert obj["remaining_pregame_coverage_complete"] is False
    assert len(obj["missing_upcoming"]) == 2


def test_schedule_audit_passes_complete_all_fbs_slate(monkeypatch, tmp_path):
    _write_model(tmp_path, ["1", "2", "3"])
    f = rows().iloc[:3].copy()
    f["season_type"] = "regular"
    monkeypatch.setattr(
        "scripts.audit_schedule_coverage.SportsDataVerseClient.season_frame",
        lambda *_: f.copy()
    )
    result = audit(tmp_path)
    assert result["remaining_pregame_coverage_complete"] is True
    assert result["counts"]["PREDICTED"] == 3
    assert result["fbs_vs_non_fbs_games"] == 2
    assert result["started_or_completed_without_archived_pregame"] == 0
