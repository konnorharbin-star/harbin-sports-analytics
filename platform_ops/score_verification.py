"""Independent, no-cost ESPN scoreboard verification for archived football results.

Public, undocumented ESPN scoreboard JSON is an independent reporting source,
NOT an official league certificate or a guaranteed stable/contracted API.
GET only, no credentials, no sportsbook endpoint, no paid providers.
Ambiguous/missing/unfinalized/conflicting scores must not be graded.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import json
import re
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from platform_ops.free_observer import finite_number, utc_datetime

ESPN_ENDPOINTS = {
    "NFL": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard",
    "CFB": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard",
}
NFL_TEAM_ALIASES = {"JAC": "JAX", "LA": "LAR", "WSH": "WAS"}
MAX_REQUEST_DATES = 70
MAX_RESPONSE_BYTES = 3_000_000
TIME_TOLERANCE = timedelta(minutes=10)
EASTERN = ZoneInfo("America/New_York")


def normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def matches_team(label: object, espn: dict[str, Any], league: str) -> bool:
    if not isinstance(espn, dict) or not label:
        return False
    label_norm = normalize_name(label)
    if league == "NFL":
        expected = NFL_TEAM_ALIASES.get(str(label).upper(), str(label).upper())
        espn_abbr = NFL_TEAM_ALIASES.get(
            str(espn.get("abbreviation") or "").upper(),
            str(espn.get("abbreviation") or "").upper(),
        )
        return expected == espn_abbr
    names = [
        espn.get("abbreviation"), espn.get("location"),
        espn.get("displayName"), espn.get("shortDisplayName"),
        espn.get("name"),
    ]
    # ESPN's event ID is the primary CFB key; reject obviously crossed teams.
    # Some display names append a mascot. No fuzzy/approximate matching.
    return len(label_norm) >= 3 and any(
        value and (normalize_name(value) == label_norm
                   or normalize_name(value).startswith(label_norm))
        for value in names
    )


def parse_scoreboard(payload: dict[str, Any], league: str) -> list[dict[str, Any]]:
    """Accept only completed final events with two unambiguous integer scores."""
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError(f"{league}: unexpected ESPN scoreboard response")
    parsed = []
    for event in payload["events"]:
        if not isinstance(event, dict):
            continue
        events = event.get("competitions")
        if not isinstance(events, list) or len(events) != 1:
            continue
        competition = events[0]
        if not isinstance(competition, dict):
            continue
        statuses = [event.get("status"), competition.get("status")]
        types = [x.get("type") for x in statuses if isinstance(x, dict)]
        completed = any(
            t.get("completed") is True and
            str(t.get("name") or "").upper().startswith("STATUS_FINAL")
            for t in types if isinstance(t, dict)
        )
        if not completed:
            continue
        competitors = competition.get("competitors")
        if not isinstance(competitors, list) or len(competitors) != 2:
            continue
        sides = {
            side.get("homeAway"): side
            for side in competitors if isinstance(side, dict)
        }
        if set(sides) != {"home", "away"}:
            continue
        scores = []
        for role in ("home", "away"):
            raw = sides[role].get("score")
            number = finite_number(raw)
            if number is None or not float(number).is_integer() or not 0 <= number <= 120:
                break
            scores.append(int(number))
        if len(scores) != 2:
            continue
        kickoff = utc_datetime(event.get("date") or competition.get("date"))
        if kickoff is None or not str(event.get("id") or "").strip():
            continue
        parsed.append({
            "espn_event_id": str(event["id"]), "league": league,
            "kickoff_utc": kickoff.isoformat(),
            "home": sides["home"].get("team") or {},
            "away": sides["away"].get("team") or {},
            "home_score": scores[0], "away_score": scores[1],
            "score_source": "ESPN public final scoreboard (nonofficial API)",
        })
    return parsed


def public_scoreboard(league: str, date: str) -> dict[str, Any]:
    """Bounded public GET; never authenticate or follow an arbitrary input URL."""
    if league not in ESPN_ENDPOINTS or not re.fullmatch(r"\d{8}", date):
        raise ValueError("Unsupported league or invalid date")
    query = {"dates": date, "limit": "500"}
    if league == "CFB":
        query["groups"] = "80"
    request = Request(
        ESPN_ENDPOINTS[league] + "?" + urlencode(query),
        headers={"Accept": "application/json", "User-Agent": "HarbinResearchVerifier/1.0"},
        method="GET",
    )
    with urlopen(request, timeout=15) as response:  # noqa: S310 - constant ESPN hostname
        data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("Oversized ESPN scoreboard; fail closed")
    return json.loads(data)


def date_for_kickoff(kickoff: datetime) -> str:
    return kickoff.astimezone(EASTERN).strftime("%Y%m%d")


def verification_index(
    candidates: list[dict[str, Any]],
    *,
    fetcher: Callable[[str, str], dict[str, Any]] = public_scoreboard,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Build a strict independent result index for known archived game IDs.

    Never use a scored result to *choose* a candidate; all candidates have
    already been frozen by the first-observation selection policy.
    """
    games = {}
    for candidate in candidates:
        kickoff = utc_datetime(candidate.get("kickoff_utc"))
        league = candidate.get("league")
        game_id = str(candidate.get("game_id") or "")
        if kickoff is None or league not in ESPN_ENDPOINTS or not game_id:
            continue
        key = (league, game_id)
        if key in games and games[key].get("kickoff_utc") != candidate.get("kickoff_utc"):
            games[key] = {"conflict": True}
        elif key not in games:
            games[key] = candidate

    jobs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (league, _), candidate in games.items():
        if candidate.get("conflict"):
            continue
        date = date_for_kickoff(utc_datetime(candidate["kickoff_utc"]))
        jobs[(league, date)].append(candidate)
    if len(jobs) > MAX_REQUEST_DATES:
        raise ValueError("Too many scoreboard dates; split/archive the season before retrying")

    verified: dict[tuple[str, str], dict[str, Any]] = {}
    mismatches = 0
    missing = 0
    unavailable = {}
    for league, date in sorted(jobs):
        try:
            events = parse_scoreboard(fetcher(league, date), league)
        except (OSError, TimeoutError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            unavailable[f"{league}:{date}"] = type(exc).__name__
            continue

        for candidate in jobs[(league, date)]:
            kickoff = utc_datetime(candidate["kickoff_utc"])
            if league == "CFB":
                possible = [e for e in events if e["espn_event_id"] == str(candidate["game_id"])]
            else:
                possible = [e for e in events if
                            matches_team(candidate.get("home_team"), e["home"], league)
                            and matches_team(candidate.get("away_team"), e["away"], league)]
            matched = [
                e for e in possible
                if abs(utc_datetime(e["kickoff_utc"]) - kickoff) <= TIME_TOLERANCE
                and matches_team(candidate.get("home_team"), e["home"], league)
                and matches_team(candidate.get("away_team"), e["away"], league)
            ]
            if len(matched) != 1:
                if possible:
                    mismatches += 1
                else:
                    missing += 1
                continue
            final = matched[0]
            verified[(league, str(candidate["game_id"]))] = {
                **final,
                "game_id": str(candidate["game_id"]),
                "home_team": candidate["home_team"],
                "away_team": candidate["away_team"],
                "margin_home": final["home_score"] - final["away_score"],
                "total": final["home_score"] + final["away_score"],
                "kickoff": final["kickoff_utc"],
            }
    return verified, {
        "independent_final_scores": len(verified),
        "requested_league_dates": len(jobs),
        "provider_unavailable_dates": unavailable,
        "match_or_kickoff_conflicts": mismatches,
        "not_yet_final_or_missing_games": missing,
        "candidate_identity_conflicts": sum(x.get("conflict") is True for x in games.values()),
        "provider": "ESPN public scoreboard, undocumented unauthenticated endpoint",
    }
