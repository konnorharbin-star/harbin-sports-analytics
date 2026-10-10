from pathlib import Path
import json

import pandas as pd
import pytest

from scripts.archive_first_seen_predictions import archive
from scripts.audit_schedule_coverage import audit


def _published(tmp_path, games):
    out = tmp_path / "outputs"
    out.mkdir(exist_ok=True)
    meta = {"season": 2026, "week": 6,
            "pregame_filter": {"cutoff_utc": "2026-10-09T21:00:00+00:00"}}
    (out / "cfb_model_2026_week6_metadata.json").write_text(json.dumps(meta))
    pd.DataFrame(games).to_csv(out / "cfb_model_2026_week6.csv", index=False)


def _game(game_id="1", date="2026-10-10T23:00:00Z"):
    return {"game_id": game_id, "date": date,
            "away_team": "Away", "home_team": "Home",
            "away_score": 20, "home_score": 24, "model_margin_home": 4,
            "model_total": 44, "win_probability": 0.63}


def test_first_seen_preserved_after_prediction_changes(tmp_path):
    _published(tmp_path, [_game()])
    assert archive(tmp_path)["newly_archived"] == 1
    _published(tmp_path, [dict(_game(), home_score=99, model_margin_home=79)])
    assert archive(tmp_path)["newly_archived"] == 0
    ledger = pd.read_csv(tmp_path / "history" / "first_seen_pregame_predictions.csv")
    assert len(ledger) == 1
    assert ledger.iloc[0].home_score == 24


def test_archive_refuses_kickoff_or_later(tmp_path):
    _published(tmp_path, [_game(date="2026-10-09T20:00:00Z")])
    with pytest.raises(ValueError, match="post-kickoff"):
        archive(tmp_path)


def test_audit_started_game_is_not_missing(tmp_path, monkeypatch):
    _published(tmp_path, [_game(game_id="2")])
    source = pd.DataFrame([
        {"game_id": "1", "season_type": "regular", "week": 6,
         "start_date": "2026-10-09T20:00:00Z", "completed": False,
         "away_team": "Florida State", "home_team": "Louisville"},
        {"game_id": "2", "season_type": "regular", "week": 6,
         "start_date": "2026-10-10T23:00:00Z", "completed": False,
         "away_team": "Away", "home_team": "Home"}])
    monkeypatch.setattr("scripts.audit_schedule_coverage.SportsDataVerseClient.season_frame",
                        lambda self, season: source)
    summary = audit(tmp_path)
    assert summary["counts"] == {"STARTED_EXCLUDED": 1, "PREDICTED": 1}


def test_audit_fails_when_future_game_missing(tmp_path, monkeypatch):
    _published(tmp_path, [_game(game_id="2")])
    source = pd.DataFrame([
        {"game_id": "1", "season_type": "regular", "week": 6,
         "start_date": "2026-10-10T20:00:00Z", "completed": False,
         "away_team": "Florida State", "home_team": "Louisville"},
        {"game_id": "2", "season_type": "regular", "week": 6,
         "start_date": "2026-10-10T23:00:00Z", "completed": False,
         "away_team": "Away", "home_team": "Home"}])
    monkeypatch.setattr("scripts.audit_schedule_coverage.SportsDataVerseClient.season_frame",
                        lambda self, season: source)
    with pytest.raises(SystemExit, match="Schedule coverage failure"):
        audit(tmp_path)
