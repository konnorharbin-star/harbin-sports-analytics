"""Free, pregame CFB event observations. Source identity is not model accuracy.

Raw response hashes and observation clocks are retained; current observations
never become historical training features or confirmation of an entire roster.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

import requests

SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
MAX_CAPTURE_AGE = timedelta(hours=6)


def utc(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("AWARE_TIMESTAMP_REQUIRED")
    return parsed.astimezone(timezone.utc)


def sid(value):
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text if text.isdigit() else ""


def canon(value):
    from .data import _canon
    return _canon(value)


def number(value):
    try:
        out = float(value)
        return out if not isinstance(value, bool) and math.isfinite(out) else None
    except (ValueError, TypeError):
        return None


def fetch_summary(game_id, session=None):
    """No redirects, no arbitrary URLs; hash the bytes that were actually read."""
    gid = sid(game_id)
    if not gid:
        raise ValueError("INVALID_ESPN_EVENT_ID")
    url = SUMMARY + "?" + urlencode({"event": gid})
    response = (session or requests).get(url, timeout=12, allow_redirects=False)
    response.raise_for_status()
    if response.status_code != 200 or response.url != url:
        raise ValueError("STRUCTURED_SOURCE_REDIRECT_OR_URL_MISMATCH")
    raw = response.content
    return {
        "source_url": url,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "normalized_payload_sha256": hashlib.sha256(json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "payload": json.loads(raw),
    }


def bind_event(snapshot, game, *, as_of):
    """Require IDs AND team names, exact kickoff, season/week, scheduled state."""
    result = {"status": "EVENT_NOT_VERIFIED", "reasons": [], "facts": {}}
    try:
        now, captured, kickoff = utc(as_of), utc(snapshot["captured_utc"]), utc(game["date"])
        if not captured <= now < kickoff or now - captured > MAX_CAPTURE_AGE:
            raise ValueError("STALE_FUTURE_OR_POSTKICKOFF_CAPTURE")
        gid = sid(game["game_id"])
        if not gid or snapshot.get("source_url") != SUMMARY + "?" + urlencode({"event": gid}):
            raise ValueError("WRONG_SOURCE_EVENT_URL")
        digest = snapshot.get("response_sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("MISSING_RAW_RESPONSE_HASH")
        payload = snapshot["payload"]
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if actual != snapshot.get("normalized_payload_sha256"):
            raise ValueError("NORMALIZED_PAYLOAD_HASH_MISMATCH")
        header = payload["header"]
        if sid(header.get("id")) != gid:
            raise ValueError("EVENT_ID_MISMATCH")
        competitions = header.get("competitions") or []
        if len(competitions) != 1:
            raise ValueError("AMBIGUOUS_COMPETITION")
        comp = competitions[0]
        if sid(comp.get("id")) != gid or utc(comp["date"]) != kickoff:
            raise ValueError("COMPETITION_OR_KICKOFF_MISMATCH")
        if header.get("timeValid") is not True or comp.get("dateValid") is not True or game.get("start_time_tbd") is True:
            raise ValueError("KICKOFF_TIME_NOT_CONFIRMED")
        status = comp.get("status", {}).get("type", {})
        if status.get("state") != "pre" or status.get("completed") is not False or status.get("name") != "STATUS_SCHEDULED":
            raise ValueError("EVENT_NOT_SCHEDULED_PREGAME")
        season = header.get("season") or {}
        if int(season.get("year", -1)) != int(game["season"]) or int(header.get("week", -1)) != int(game["week"]):
            raise ValueError("SEASON_OR_WEEK_MISMATCH")
        competitors = comp.get("competitors") or []
        if len(competitors) != 2:
            raise ValueError("AMBIGUOUS_TEAMS")
        team_ids = {}
        for side in ("home", "away"):
            matches = [c for c in competitors if c.get("homeAway") == side]
            if len(matches) != 1:
                raise ValueError("AMBIGUOUS_TEAM_SIDE")
            team = matches[0].get("team") or {}
            tid = sid(team.get("id"))
            expected = sid(game.get(side + "_id"))
            names = {canon(team.get(k, "")) for k in ("displayName", "shortDisplayName", "location", "name", "abbreviation")}
            if not expected or tid != expected or canon(game[side + "_team"]) not in names:
                raise ValueError("TEAM_ID_OR_NAME_MISMATCH")
            team_ids[side] = tid
        if team_ids["home"] == team_ids["away"]:
            raise ValueError("DUPLICATE_TEAM_ID")
        neutral = comp.get("neutralSite")
        if not isinstance(neutral, bool) or neutral != game.get("neutral_site"):
            raise ValueError("NEUTRAL_SITE_MISMATCH_OR_UNKNOWN")
        venue = (payload.get("gameInfo") or {}).get("venue") or comp.get("venue") or {}
        vid = sid(venue.get("id"))
        name = venue.get("fullName")
        if not vid or not isinstance(name, str) or not name.strip():
            raise ValueError("EVENT_VENUE_MISSING")
        if sid(game.get("venue_id")) and sid(game["venue_id"]) != vid:
            raise ValueError("SCHEDULE_EVENT_VENUE_MISMATCH")
        result.update(status="EXACT_PREGAME_EVENT_VERIFIED", facts={
            "game_id": gid, "kickoff_utc": kickoff.isoformat(), "team_ids": team_ids,
            "venue_id": vid, "venue_name": name, "neutral_site": neutral,
            "venue_reported_indoor": venue.get("indoor") if isinstance(venue.get("indoor"), bool) else None,
        })
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        result["reasons"].append(str(exc))
    result["source"] = {k: snapshot.get(k) for k in ("source_url", "captured_utc", "response_sha256", "normalized_payload_sha256")}
    return result


def event_venue(binding, snapshot, team_metadata):
    """Only event coordinates or an exact venue ID join; no home-city guess."""
    if binding["status"] != "EXACT_PREGAME_EVENT_VERIFIED":
        return {}
    facts = binding["facts"]
    venue = (snapshot["payload"].get("gameInfo") or {}).get("venue") or {}
    candidates = [m for m in team_metadata if sid(m.get("venue_id")) == facts["venue_id"]]
    points = {(number(m.get("latitude")), number(m.get("longitude"))) for m in candidates}
    points = {(lat, lon) for lat, lon in points if lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180}
    lat, lon = number(venue.get("latitude")), number(venue.get("longitude"))
    method = "ESPN_EVENT_VENUE_COORDINATES"
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        if len(points) != 1:
            return {"venue_id": facts["venue_id"], "venue_name": facts["venue_name"], "indoor": False, "indoor_known": False}
        lat, lon = next(iter(points))
        method = "EXACT_EVENT_VENUE_ID_METADATA_JOIN"
    # Outdoor classification requires a known metadata value and exact venue ID.
    # Generic indoor flags never certify a retractable roof's operating position.
    indoor_values = {m.get("indoor") for m in candidates if m.get("indoor_known") is True}
    roof = "OUTDOOR_ROOF_POSITION_NOT_APPLICABLE" if indoor_values == {False} else "GAME_DAY_ROOF_OR_ENCLOSURE_UNVERIFIED"
    return {"venue_id": facts["venue_id"], "venue_name": facts["venue_name"],
            "latitude": lat, "longitude": lon, "coordinate_binding": method,
            "indoor": False, "indoor_known": False,
            "roof_status": roof,
            "reported_indoor": facts["venue_reported_indoor"]}


def reported_injuries(binding, snapshot, *, as_of):
    """Event-scoped, dated secondary reports. Omission never establishes active."""
    output, rejected = [], []
    if binding["status"] != "EXACT_PREGAME_EVENT_VERIFIED":
        return {"records": [], "rejections": ["EVENT_NOT_VERIFIED"]}
    captured, now = utc(snapshot["captured_utc"]), utc(as_of)
    kickoff = utc(binding["facts"]["kickoff_utc"])
    valid_teams = set(binding["facts"]["team_ids"].values())
    seen = set()
    for group in snapshot["payload"].get("injuries") or []:
        tid = sid((group.get("team") or {}).get("id"))
        for item in group.get("injuries") or []:
            try:
                athlete = item.get("athlete") or {}
                pid = sid(athlete.get("id"))
                athlete_team = sid((athlete.get("team") or {}).get("id"))
                date = utc(item.get("date"))
                status = item.get("status")
                if tid not in valid_teams or (athlete_team and athlete_team != tid):
                    raise ValueError("INJURY_TEAM_MISMATCH")
                if not pid or not isinstance(status, str) or not status.strip():
                    raise ValueError("MISSING_ATHLETE_ID_OR_STATUS")
                if not date <= captured <= now < kickoff or captured - date > timedelta(days=7):
                    raise ValueError("STALE_FUTURE_OR_POSTGAME_INJURY")
                if (tid, pid) in seen:
                    raise ValueError("DUPLICATE_PLAYER_REPORT")
                seen.add((tid, pid))
                output.append({"team_id": tid, "athlete_id": pid, "athlete_name": athlete.get("displayName"),
                               "position": (athlete.get("position") or {}).get("abbreviation"),
                               "status": status, "date": date.isoformat(),
                               "evidence_status": "DATED_EVENT_SECONDARY_REPORT", "starter_verified": False,
                               "final_game_availability_verified": False})
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                rejected.append(str(exc))
    return {"records": output, "rejections": rejected, "not_listed_does_not_prove_active": True}


def kickoff_weather(payload, kickoff, *, captured, as_of):
    """Validate forecast units and exact covering UTC hour, never nearest day."""
    kick, capture, now = utc(kickoff), utc(captured), utc(as_of)
    if not capture <= now < kick or now - capture > MAX_CAPTURE_AGE:
        raise ValueError("WEATHER_CAPTURE_NOT_FRESH_PREGAME")
    units = payload.get("hourly_units") or {}
    required = {"temperature_2m": "°F", "wind_speed_10m": "mp/h", "wind_gusts_10m": "mp/h",
                "precipitation_probability": "%", "precipitation": "mm", "weather_code": "wmo code"}
    if any(units.get(k) != v for k, v in required.items()) or payload.get("utc_offset_seconds") != 0:
        raise ValueError("WEATHER_UNITS_OR_TIMEZONE_MISMATCH")
    h = payload.get("hourly") or {}
    times = [utc(t + "+00:00") for t in h.get("time", [])]
    indices = [i for i, t in enumerate(times) if t <= kick < t + timedelta(hours=1)]
    if len(indices) != 1:
        raise ValueError("NO_UNIQUE_KICKOFF_WEATHER_HOUR")
    index = indices[0]
    values = {}
    for key in required:
        row = h.get(key) or []
        if len(row) != len(times) or number(row[index]) is None:
            raise ValueError("WEATHER_VALUES_MISSING_OR_INVALID")
        values[key] = number(row[index])
    if not (0 <= values["precipitation_probability"] <= 100) or any(values[k] < 0 for k in ("wind_speed_10m", "wind_gusts_10m", "precipitation")):
        raise ValueError("WEATHER_VALUES_OUT_OF_RANGE")
    return {"temperature_f": values["temperature_2m"], "precip_probability": values["precipitation_probability"],
            "precipitation_in": values["precipitation"] / 25.4, "weather_code": values["weather_code"],
            "wind_mph": values["wind_speed_10m"], "wind_gust_mph": values["wind_gusts_10m"],
            "forecast_hour_utc": times[index].isoformat(), "captured_utc": capture.isoformat()}
