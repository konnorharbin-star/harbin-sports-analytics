"""Independent scoreboard verification tests: refuse phantom scores and late/ambiguous games."""
from datetime import datetime, timezone
from pathlib import Path

import pytest

from platform_ops.grade_observations import grade_archive
from platform_ops.score_verification import (
    date_for_kickoff, matches_team, parse_scoreboard, public_scoreboard,
    verification_index,
)

KICKOFF = "2026-10-09T00:15:00+00:00"
OBSERVED = "2026-10-08T22:30:00+00:00"
NOW = datetime(2026, 10, 16, tzinfo=timezone.utc)


def espn_event(*, league="NFL", complete=True, home="DAL", away="TB",
               home_score="27", away_score="17", event_id="401910111",
               kickoff=KICKOFF):
    if league == "CFB":
        home = "Liberty"
        away = "Sam Houston"
    return {
        "id": event_id, "date": kickoff,
        "status": {"type": {
            "completed": complete,
            "name": "STATUS_FINAL" if complete else "STATUS_IN_PROGRESS",
        }},
        "competitions": [{"competitors": [
            {"homeAway": "away", "score": away_score,
             "team": {"abbreviation": away, "location": away,
                      "displayName": away + " Mascot"}},
            {"homeAway": "home", "score": home_score,
             "team": {"abbreviation": home, "location": home,
                      "displayName": home + " Mascot"}},
        ]}],
    }


def candidate(league="NFL"):
    base = {
        "league": league, "game_id": "2026_05_TB_DAL",
        "home_team": "DAL", "away_team": "TB",
        "kickoff_utc": KICKOFF,
        "market": "total", "side": "under", "line": 49.5,
        "odds": -110, "observed_at_utc": OBSERVED,
        "quoted_at_utc": "2026-10-08T22:20:00+00:00",
    }
    if league == "CFB":
        base.update(game_id="401910111", home_team="Liberty",
                    away_team="Sam Houston")
    return base


def model_results(*, home_score=27, away_score=17, cfb=False):
    return ("game_id,home_team,away_team,kickoff,actual_margin_home,actual_total,result\n"
            + (f"401910111,Liberty,Sam Houston,{KICKOFF},"
               if cfb else f"2026_05_TB_DAL,DAL,TB,{KICKOFF},")
            + f"{home_score-away_score},{home_score+away_score},win\n")


def archive(tmp_path: Path):
    day = tmp_path / "2026" / "10"
    day.mkdir(parents=True)
    record = {
        "schema_version": 1, "mode": "READ_ONLY_RESEARCH",
        "automatic_betting_enabled": False, "snapshot_id": "a" * 64,
        "observed_at_utc": OBSERVED,
        "leagues": {
            "NFL": {"research_watchlist": [{
                "game_id": "2026_05_TB_DAL", "home_team": "DAL",
                "away_team": "TB", "kickoff_utc": KICKOFF,
                "market": "total", "side": "under", "line": 49.5,
                "american_odds": -110, "book": "Example Book",
                "model_probability": .55, "recalculated_ev": .05,
                "quoted_at_utc": "2026-10-08T22:20:00+00:00",
            }]},
            "CFB": {"research_watchlist": []},
        },
    }
    import json
    (day / "08.jsonl").write_text(json.dumps(record)+"\n")


def test_final_score_parses_away_home_independent_of_order():
    rows = parse_scoreboard({"events": [espn_event()]}, "NFL")
    assert len(rows) == 1
    assert rows[0]["home_score"] == 27
    assert rows[0]["away_score"] == 17


def test_in_progress_canceled_and_invalid_scores_not_accepted():
    cases = [espn_event(complete=False), espn_event(home_score=""),
             espn_event(home_score="27.5"), espn_event(home_score="-1")]
    assert not parse_scoreboard({"events": cases}, "NFL")
    with pytest.raises(ValueError):
        parse_scoreboard({"error": "rate limit"}, "NFL")


def test_team_aliases_and_cfb_full_names():
    assert matches_team("LA", {"abbreviation": "LAR"}, "NFL")
    assert matches_team("WSH", {"abbreviation": "WAS"}, "NFL")
    assert not matches_team("SEA", {"abbreviation": "DAL"}, "NFL")
    assert matches_team("South Florida", {
        "abbreviation": "USF", "location": "South Florida",
        "displayName": "South Florida Bulls",
    }, "CFB")
    assert not matches_team("Sam Houston", {
        "abbreviation": "LIB", "location": "Liberty"}, "CFB")


def test_timezone_correctly_chooses_eastern_game_date():
    kickoff = datetime.fromisoformat(KICKOFF)
    assert date_for_kickoff(kickoff) == "20261008"


def test_external_scoreboard_fetcher_bound_to_supported_routes():
    with pytest.raises(ValueError):
        public_scoreboard("MLB", "20261008")
    with pytest.raises(ValueError):
        public_scoreboard("NFL", "2026-10-08")


def test_nfl_match_by_exact_teams_and_kickoff():
    calls = []
    def fetch(league, date):
        calls.append((league, date))
        return {"events": [espn_event()]}
    scores, meta = verification_index([candidate()], fetcher=fetch)
    assert calls == [("NFL", "20261008")]
    assert scores[("NFL", "2026_05_TB_DAL")]["margin_home"] == 10
    assert scores[("NFL", "2026_05_TB_DAL")]["total"] == 44
    assert meta["independent_final_scores"] == 1


def test_cfb_match_by_exact_espn_event_id_plus_team_and_time():
    scores, _ = verification_index([candidate("CFB")], fetcher=lambda l,d: {
        "events": [espn_event(league="CFB")]
    })
    assert scores[("CFB", "401910111")]["total"] == 44


def test_wrong_kickoff_duplicate_or_wrong_team_never_verified():
    wrongtime = espn_event(kickoff="2026-10-09T01:15:00Z")
    wrongteam = espn_event(home="NYG")
    for events in [[wrongtime], [wrongteam], [espn_event(), espn_event()]]:
        scores, _ = verification_index([candidate()], fetcher=lambda l,d: {
            "events": events
        })
        assert not scores


def test_network_failure_means_no_verified_scores():
    def outage(league, date):
        raise TimeoutError("source unavailable")
    scores, meta = verification_index([candidate()], fetcher=outage)
    assert not scores
    assert meta["provider_unavailable_dates"]


def test_deployed_grading_requires_both_independent_and_model_score(tmp_path):
    archive(tmp_path)
    match, meta = verification_index([candidate()], fetcher=lambda l,d: {
        "events": [espn_event()]
    })
    good = grade_archive(
        tmp_path, {"NFL": model_results()}, NOW,
        independent_scores=match, require_independent=True, independent_meta=meta)
    assert len(good["graded"]) == 1
    assert good["graded"][0]["independent_score_verified"]
    assert not good["graded"][0]["bet_was_placed"]
    assert not good["verified_profitability_proven"]


def test_mismatch_between_model_and_espn_withholds_grade(tmp_path):
    archive(tmp_path)
    match, meta = verification_index([candidate()], fetcher=lambda l,d: {
        "events": [espn_event()]
    })
    bad = grade_archive(
        tmp_path, {"NFL": model_results(home_score=31)}, NOW,
        independent_scores=match, require_independent=True, independent_meta=meta)
    assert len(bad["graded"]) == 0
    assert bad["unresolved"]["independent_scores_disagree"] == 1


def test_missing_independent_score_withholds_grade(tmp_path):
    archive(tmp_path)
    bad = grade_archive(
        tmp_path, {"NFL": model_results()}, NOW,
        independent_scores={}, require_independent=True)
    assert len(bad["graded"]) == 0
    assert bad["summary"]["NFL"]["hypothetical_flat_unit_roi"] is None
    assert bad["unresolved"]["independent_scores_missing"] == 1
