"""Publish inspectable source checks tied to the exact forecast file."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone, timedelta
from html import escape
from pathlib import Path

from .public_inputs import utc


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_quality(source, records, *, generated_at, as_of):
    forecasts = json.loads(source)
    ids = [str(g["game_id"]) for g in forecasts]
    if len(ids) != len(set(ids)):
        raise ValueError("DUPLICATE_FORECAST_GAME")
    by_id = {}
    for row in records:
        gid = str(row["game_id"])
        if gid in by_id:
            raise ValueError("DUPLICATE_SOURCE_GAME")
        by_id[gid] = row
    games = []
    now = utc(as_of)
    for forecast in forecasts:
        gid = str(forecast["game_id"])
        row = json.loads(json.dumps(clean(by_id.get(gid) or {"game_id": gid})))
        missing = list(row.get("missing_required_inputs") or [])
        valid = True
        if any(row.get(k) != forecast.get(k) for k in ("away_team", "home_team")) or row.get("kickoff_utc") != utc(forecast["date"]).isoformat():
            missing.append("source_forecast_identity_mismatch")
            valid = False
        binding = row.get("event_binding") or {}
        try:
            capture = utc(binding["source"]["captured_utc"])
            if not capture <= now < utc(forecast["date"]) or now - capture > timedelta(hours=6):
                raise ValueError("SOURCE_EXPIRED_AT_PUBLICATION")
        except (ValueError, TypeError, KeyError):
            missing.append("fresh_pregame_source_at_publication")
            valid = False
        row["event_verified_at_publication"] = valid and binding.get("status") == "EXACT_PREGAME_EVENT_VERIFIED"
        row["missing_required_inputs"] = sorted(set(missing))
        row["ready_for_context_adjusted_prediction"] = False
        games.append(row)
    return {
        "spec": "CFB_PUBLIC_INPUT_QUALITY_V1", "checked_at_utc": now.isoformat(),
        "source_forecast_generated_at": generated_at,
        "source_forecast_sha256": hashlib.sha256(source).hexdigest(),
        "status": "SOURCE_CHECKS_AVAILABLE_UNVERIFIED_INPUTS_REMAIN",
        "games": games, "games_checked": len(games),
        "exact_events_verified": sum(g["event_verified_at_publication"] for g in games),
        "missing_input_counts": dict(Counter(k for g in games for k in g["missing_required_inputs"])),
        "current_context_enters_historical_training": False,
        "score_adjustments_from_web_context_enabled": False,
        "limitations": ["Trusted publisher does not guarantee the prediction outcome",
                        "Dated secondary statuses do not establish starters or final game availability",
                        "A season roster is not a game-day active list",
                        "Unconfirmed retractable roof position remains unknown"],
    }


def render_quality(report):
    rows = []
    for game in report["games"]:
        binding = game.get("event_binding") or {}
        facts = binding.get("facts") or {}
        url = (binding.get("source") or {}).get("source_url") or ""
        source = f'<a href="{escape(url, quote=True)}">Source</a>' if url.startswith("https://site.api.espn.com/") else "Unavailable"
        values = [game.get("away_team", "?") + " at " + game.get("home_team", "?"),
                  "Verified" if game["event_verified_at_publication"] else "Unverified",
                  facts.get("venue_name", "Unverified"),
                  str(game.get("fresh_home_team_reports", 0) + game.get("fresh_away_team_reports", 0)),
                  "Verified" if game.get("weather") else "Unverified",
                  "Not applicable (outdoor)" if game.get("roof_status") == "OUTDOOR_ROOF_POSITION_NOT_APPLICABLE" else "Unverified",
                  ", ".join(k.replace("_", " ") for k in game["missing_required_inputs"])]
        rows.append("<tr>" + "".join("<td>" + escape(v) + "</td>" for v in values) + "<td>" + source + "</td></tr>")
    return ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>College football source checks</title><style>body{font:16px system-ui;background:#111820;color:#edf2f7;margin:24px}a{color:#9bc8ff}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #526070;padding:12px;text-align:left;vertical-align:top}.scroll{overflow:auto}p{max-width:950px}</style></head><body>'
            '<a href="index.html">Back to predictions</a><h1>College football source checks</h1>'
            '<p>Game identity, venue, dated injury reports and weather are checked separately. A roster listing does not confirm a player will play or start. Missing details remain unverified. Current context affects uncertainty and risk; it does not adjust the fair scores.</p>'
            f'<p>Checked {escape(report["checked_at_utc"])} · {report["exact_events_verified"]}/{report["games_checked"]} exact events verified. <a href="public_input_quality.json">Detailed evidence</a></p>'
            '<div class="scroll"><table><tr><th>Game</th><th>Event</th><th>Venue</th><th>Fresh reports</th><th>Weather</th><th>Roof position</th><th>Still needed</th><th>Evidence</th></tr>'
            + "".join(rows) + '</table></div></body></html>')


def write_quality(source_path, records, *, generated_at, output_dirs, as_of=None):
    report = build_quality(Path(source_path).read_bytes(), records, generated_at=generated_at,
                           as_of=as_of or datetime.now(timezone.utc).isoformat())
    raw = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    for directory in output_dirs:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "public_input_quality.json").write_text(raw, encoding="utf-8")
        (directory / "public_input_quality.html").write_text(render_quality(report), encoding="utf-8")
    return {k: report[k] for k in ("status", "checked_at_utc", "source_forecast_sha256", "games_checked", "exact_events_verified", "missing_input_counts")}
