"""Adversarial source tests: correct publisher alone cannot certify a fact."""
import copy
import hashlib
import json

import pandas as pd
import pytest

from harbin.context import ContextStore, normalize_injury_rows, summarize_roster_rows
from harbin.public_inputs import SUMMARY, bind_event, event_venue, reported_injuries, kickoff_weather, fetch_summary
from harbin.public_input_quality import build_quality, render_quality

NOW = "2026-10-20T12:00:00+00:00"
KICK = "2026-10-24T17:00:00+00:00"


def game():
    return {"game_id": "123", "season": 2026, "week": 8, "date": KICK,
            "home_id": "1", "away_id": "2", "home_team": "Alabama", "away_team": "Georgia",
            "neutral_site": True, "venue_id": "90", "start_time_tbd": False}


def seal(snapshot):
    snapshot["normalized_payload_sha256"] = hashlib.sha256(json.dumps(snapshot["payload"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return snapshot


def snapshot():
    return seal({"source_url": SUMMARY + "?event=123", "captured_utc": NOW, "response_sha256": "a" * 64,
                 "payload": {"header": {"id": "123", "week": 8, "season": {"year": 2026}, "timeValid": True,
                 "competitions": [{"id": "123", "date": KICK, "dateValid": True, "neutralSite": True,
                                   "status": {"type": {"state": "pre", "completed": False, "name": "STATUS_SCHEDULED"}},
                                   "competitors": [{"homeAway": side, "team": {"id": tid, "location": team}}
                                                   for side, tid, team in [("home", "1", "Alabama"), ("away", "2", "Georgia")]]}]},
                 "gameInfo": {"venue": {"id": "90", "fullName": "Neutral Stadium"}}}})


def test_exact_pregame_match():
    result = bind_event(snapshot(), game(), as_of=NOW)
    assert result["status"] == "EXACT_PREGAME_EVENT_VERIFIED"
    assert result["facts"]["neutral_site"] is True


@pytest.mark.parametrize("change", ["kickoff", "team", "season", "week", "neutral", "venue", "tbd", "postgame", "duplicate", "hash"])
def test_wrong_game_and_invalid_sources_rejected(change):
    snap = snapshot()
    header = snap["payload"]["header"]
    comp = header["competitions"][0]
    if change == "kickoff": comp["date"] = "2026-10-24T18:00:00Z"
    if change == "team": comp["competitors"][0]["team"]["id"] = "77"
    if change == "season": header["season"]["year"] = 2025
    if change == "week": header["week"] = 7
    if change == "neutral": comp["neutralSite"] = False
    if change == "venue": snap["payload"]["gameInfo"]["venue"]["id"] = "91"
    if change == "tbd": header["timeValid"] = False
    if change == "postgame": comp["status"]["type"]["state"] = "post"
    if change == "duplicate": comp["competitors"].append(copy.deepcopy(comp["competitors"][0]))
    if change != "hash": seal(snap)
    else: header["week"] = 9
    assert bind_event(snap, game(), as_of=NOW)["status"] == "EVENT_NOT_VERIFIED"


@pytest.mark.parametrize("clock", ["2026-10-20T11:00:00Z", "2026-10-21T12:00:00Z", KICK])
def test_future_stale_and_postkickoff_capture_rejected(clock):
    assert bind_event(snapshot(), game(), as_of=clock)["status"] == "EVENT_NOT_VERIFIED"


def test_neutral_site_cannot_use_home_venue_coordinates():
    snap = snapshot()
    bound = bind_event(snap, game(), as_of=NOW)
    home = {"venue_id": "10", "latitude": 33.2, "longitude": -87.5, "indoor": False, "indoor_known": True}
    assert "latitude" not in event_venue(bound, snap, [home])
    neutral = {**home, "venue_id": "90", "latitude": 32.7}
    actual = event_venue(bound, snap, [home, neutral])
    assert actual["latitude"] == 32.7
    assert actual["roof_status"] == "OUTDOOR_ROOF_POSITION_NOT_APPLICABLE"
    conflict = {**neutral, "latitude": 30}
    assert "latitude" not in event_venue(bound, snap, [neutral, conflict])


def test_indoor_flag_does_not_confirm_retractable_roof_position():
    snap = snapshot()
    bound = bind_event(snap, game(), as_of=NOW)
    venue = event_venue(bound, snap, [{"venue_id": "90", "latitude": 33.2, "longitude": -87.5, "indoor": True, "indoor_known": True}])
    assert venue["indoor"] is False
    assert venue["roof_status"] == "GAME_DAY_ROOF_OR_ENCLOSURE_UNVERIFIED"


def test_dated_event_report_is_not_starter_or_final_active_list():
    snap = snapshot()
    snap["payload"]["injuries"] = [{"team": {"id": "1"}, "injuries": [
        {"date": "2026-10-20T11:00:00Z", "status": "Questionable", "athlete": {"id": "7", "displayName": "Player", "position": {"abbreviation": "QB"}}},
        {"date": "2026-10-01T11:00:00Z", "status": "Out", "athlete": {"id": "8"}},
    ]}]
    seal(snap)
    rows = reported_injuries(bind_event(snap, game(), as_of=NOW), snap, as_of=NOW)
    assert len(rows["records"]) == 1
    assert rows["records"][0]["starter_verified"] is False
    assert rows["records"][0]["final_game_availability_verified"] is False
    assert rows["not_listed_does_not_prove_active"] is True
    assert rows["rejections"] == ["STALE_FUTURE_OR_POSTGAME_INJURY"]


def weather():
    return {"utc_offset_seconds": 0, "hourly_units": {"temperature_2m": "°F", "wind_speed_10m": "mp/h", "wind_gusts_10m": "mp/h", "precipitation_probability": "%", "precipitation": "mm", "weather_code": "wmo code"},
            "hourly": {"time": ["2026-10-24T17:00"], "temperature_2m": [72], "wind_speed_10m": [8], "wind_gusts_10m": [12], "precipitation_probability": [20], "precipitation": [25.4], "weather_code": [1]}}


def test_weather_exact_covering_hour_and_units():
    result = kickoff_weather(weather(), KICK, captured=NOW, as_of=NOW)
    assert result["precipitation_in"] == 1
    assert result["wind_mph"] == 8


@pytest.mark.parametrize("change", ["nearest_day", "null", "unit", "timezone", "range", "duplicate"])
def test_weather_failures_do_not_earn_coverage(change):
    payload = weather()
    if change == "nearest_day": payload["hourly"]["time"] = ["2026-10-23T17:00"]
    if change == "null": payload["hourly"]["wind_speed_10m"] = [None]
    if change == "unit": payload["hourly_units"]["wind_speed_10m"] = "km/h"
    if change == "timezone": payload["utc_offset_seconds"] = -18000
    if change == "range": payload["hourly"]["precipitation_probability"] = [101]
    if change == "duplicate": payload["hourly"]["time"] *= 2
    with pytest.raises(ValueError): kickoff_weather(payload, KICK, captured=NOW, as_of=NOW)


def test_injury_export_date_does_not_refresh_old_report():
    frame = pd.DataFrame([{"team_id": 1, "athlete_id": 7, "status": "Out", "as_of_date": NOW, "injury_date": "2026-09-01T00:00:00Z"}])
    norm = normalize_injury_rows(frame, as_of=NOW)
    assert norm.iloc[0]["_report_age_days"] > 40
    frame = frame.drop(columns="injury_date")
    assert pd.isna(normalize_injury_rows(frame, as_of=NOW).iloc[0]["_report_age_days"])


def test_unavailable_and_unknown_roster_members_are_not_active():
    rows = pd.DataFrame([{"id": 1, "status": "Unavailable"}, {"id": 2, "status": ""}, {"id": 3, "status": "Active"}])
    result = summarize_roster_rows(rows)
    assert result["roster_active_count"] == 1
    assert result["roster_inactive_count"] == 1
    assert result["roster_status_coverage"] == pytest.approx(2 / 3)


def test_quality_bound_to_original_forecast_and_rechecks_expiry():
    source = json.dumps([game()]).encode()
    record = {**game(), "kickoff_utc": KICK, "event_binding": bind_event(snapshot(), game(), as_of=NOW),
              "missing_required_inputs": ["confirmed_starting_qbs"]}
    report = build_quality(source, [record], generated_at=NOW, as_of=NOW)
    assert report["exact_events_verified"] == 1
    assert report["source_forecast_sha256"] == hashlib.sha256(source).hexdigest()
    assert report["score_adjustments_from_web_context_enabled"] is False
    expired = build_quality(source, [record], generated_at=NOW, as_of="2026-10-21T12:00:00Z")
    assert expired["exact_events_verified"] == 0
    assert "fresh_pregame_source_at_publication" in expired["missing_input_counts"]
    assert "confirmed starting qbs" in render_quality(report)


def test_structured_redirect_rejected():
    class Response:
        status_code = 302
        url = SUMMARY + "?event=123"
        def raise_for_status(self): pass
    class Session:
        def get(self, *args, **kwargs):
            assert kwargs["allow_redirects"] is False
            return Response()
    with pytest.raises(ValueError, match="REDIRECT"):
        fetch_summary("123", Session())


def test_missing_team_reports_do_not_get_credit_from_another_team(monkeypatch, tmp_path):
    monkeypatch.setattr(ContextStore, "_load_injuries", lambda *_: None)
    monkeypatch.setattr(ContextStore, "_load_rosters", lambda *_: None)
    monkeypatch.setattr(ContextStore, "_load_team_info", lambda *_: None)
    store = ContextStore(2026, pd.DataFrame([{**game(), "start_date": KICK}]), cache_dir=tmp_path, as_of=NOW)
    store.injuries_raw = pd.DataFrame([{"team_id": 99, "athlete_id": 7, "status": "Out", "injury_date": NOW}])
    monkeypatch.setattr(store, "_event_context", lambda *_: (bind_event(snapshot(), game(), as_of=NOW), {}, {}))
    forecasts = pd.DataFrame([{**game(), "model_margin_home": 3.2, "model_total": 48.1}])
    out, metadata = store.attach(forecasts)
    assert metadata["injury_source_coverage"] == 0
    assert metadata["weather_coverage"] == 0
    assert out.iloc[0]["context_quality_score"] == 0
    assert out.iloc[0]["model_margin_home"] == 3.2
    assert out.iloc[0]["model_total"] == 48.1
    assert out.iloc[0]["home_qb_starter_verified"] == False
