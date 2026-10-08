"""Tests for archived observation grading with no bookmaker accounts or paid data."""
from datetime import datetime, timedelta, timezone
import json

from platform_ops.grade_observations import (
    collect_archived_candidates,
    grade_archive,
    grade_candidate,
    report_markdown,
    result_index,
)

NOW = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
START = datetime(2026, 10, 8, 22, 30, tzinfo=timezone.utc)
KICKOFF = datetime(2026, 10, 9, 0, 15, tzinfo=timezone.utc)


def candidate(
    *,
    market="spread", side="away", line=7.5, odds=-110,
    quoted=None, kickoff=None,
):
    return {
        "game_id": "2026_05_TB_DAL", "away_team": "TB", "home_team": "DAL",
        "market": market, "side": side, "line": line,
        "american_odds": odds, "book": "Public Example Book",
        "model_probability": .60, "recalculated_ev": .14,
        "quoted_at_utc": (quoted or START - timedelta(minutes=9)).isoformat(),
        "kickoff_utc": (kickoff or KICKOFF).isoformat(), "blockers": [
            "model_not_production_validated",
        ],
    }


def archive(tmp_path, candidates, *, observed=START, league="NFL", snap_id="a" * 64):
    folder = tmp_path / "2026" / "10"
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "08.jsonl"
    record = {
        "schema_version": 1, "mode": "READ_ONLY_RESEARCH",
        "automatic_betting_enabled": False,
        "snapshot_id": snap_id, "observed_at_utc": observed.isoformat(),
        "leagues": {
            "NFL": {"research_watchlist": candidates if league == "NFL" else [],
                    "source_contract_status": "PASS",
                    "audit_status": "WARN", "reconciliation_status": "PASS"},
            "CFB": {"research_watchlist": candidates if league == "CFB" else [],
                    "source_contract_status": "PASS",
                    "audit_status": "WARN", "reconciliation_status": "PASS"},
        },
    }
    with file.open("a") as out:
        out.write(json.dumps(record) + "\n")
    return file


def results_csv(
    *,
    margin=4, total=44, kickoff=KICKOFF, game="2026_05_TB_DAL",
    home="DAL", away="TB",
):
    return (
        "game_id,home_team,away_team,kickoff,actual_margin_home,actual_total,result\n"
        f"{game},{home},{away},{kickoff.isoformat()},{margin},{total},win\n"
    )


def test_spread_grading_from_pregame_locked_price(tmp_path):
    archive(tmp_path, [candidate()])
    report = grade_archive(tmp_path, {"NFL": results_csv()}, NOW)
    graded = report["graded"]
    assert len(graded) == 1
    assert graded[0]["result"] == "WIN"
    assert abs(graded[0]["hypothetical_profit_units"] - 100/110) < 1e-9
    assert graded[0]["bet_was_placed"] is False
    assert graded[0]["verified_executable_entry"] is False
    assert graded[0]["evidence_tier"] == "RESEARCH_OUTCOME_ONLY"
    assert report["summary"]["NFL"]["qualified_profitability_proven"] is False
    assert "hypothetical" in report_markdown(report).lower()


def test_spread_favorites_and_pushtypes():
    base = {
        **candidate(market="spread", side="home", line=-4),
        "observed_at_utc": START.isoformat(), "odds": -110
    }
    actual = {
        "game_id": "2026_05_TB_DAL", "home_team": "DAL", "away_team": "TB",
        "kickoff": KICKOFF.isoformat(), "margin_home": 4.0, "total": 44.0
    }
    assert grade_candidate(base, actual)["result"] == "PUSH"
    base["line"] = -4.5
    assert grade_candidate(base, actual)["result"] == "LOSS"
    base["line"] = -3.5
    assert grade_candidate(base, actual)["result"] == "WIN"


def test_totals_and_moneyline():
    actual = {
        "game_id": "2026_05_TB_DAL", "home_team": "DAL", "away_team": "TB",
        "kickoff": KICKOFF.isoformat(), "margin_home": 4.0, "total": 44.0,
    }
    t = {**candidate(market="total", side="under", line=47), "odds": -110,
         "observed_at_utc": START.isoformat()}
    assert grade_candidate(t, actual)["result"] == "WIN"
    t["line"] = 44
    assert grade_candidate(t, actual)["result"] == "PUSH"
    t["side"] = "over"
    t["line"] = 47
    assert grade_candidate(t, actual)["result"] == "LOSS"
    m = {**candidate(market="moneyline", side="home", line=None, odds=150),
         "observed_at_utc": START.isoformat()}
    assert grade_candidate(m, actual)["hypothetical_profit_units"] == 1.5
    m["side"] = "away"
    assert grade_candidate(m, actual)["result"] == "LOSS"


def test_never_grade_games_before_kickoff(tmp_path):
    archive(tmp_path, [candidate()])
    report = grade_archive(tmp_path, {"NFL": results_csv()}, START)
    assert not report["graded"]
    assert report["unresolved"]["pending_games"] == 1


def test_wrong_outcome_identity_or_kickoff_is_not_scored(tmp_path):
    archive(tmp_path, [candidate()])
    data = grade_archive(tmp_path, {"NFL": results_csv(home="PHI")}, NOW)
    assert not data["graded"]
    assert data["unresolved"]["outcomes_unavailable_or_mismatch"] == 1
    data = grade_archive(tmp_path, {"NFL": results_csv(kickoff=KICKOFF+timedelta(hours=1))}, NOW)
    assert not data["graded"]


def test_post_kickoff_and_future_quotes_rejected(tmp_path):
    archive(tmp_path, [candidate(quoted=KICKOFF)])
    observations, counters = collect_archived_candidates(tmp_path)
    assert not observations
    assert counters["invalid_timing"] == 1


def test_stale_untimestamped_and_missing_prices_rejected(tmp_path):
    old=START-timedelta(days=1)
    c = candidate(quoted=old)
    archive(tmp_path, [c])
    observations, audit = collect_archived_candidates(tmp_path)
    assert not observations
    assert audit["missing_quote_or_price"] == 1


def test_duplicate_snapshots_freeze_earliest_market_choice(tmp_path):
    archive(tmp_path, [candidate(line=7.5)], observed=START)
    archive(tmp_path, [candidate(line=10.5)], observed=START+timedelta(minutes=3), snap_id="b"*64)
    rows, counts = collect_archived_candidates(tmp_path)
    assert counts["snapshots"] == 2
    assert len(rows) == 1
    assert rows[0]["line"] == 7.5


def test_conflicted_outcomes_fail_closed(tmp_path):
    archive(tmp_path, [candidate()])
    text = results_csv(margin=4) + results_csv(margin=14).split("\n",1)[1]
    index, conflicts = result_index("NFL", text)
    assert not index and conflicts == {"2026_05_TB_DAL"}
    report = grade_archive(tmp_path, {"NFL": text}, NOW)
    assert not report["graded"]
    assert report["summary"]["NFL"]["result_source_conflicts"] == 1


def test_result_must_be_from_an_explicitly_graded_row():
    invalid = results_csv().replace(",win", ",")
    scores, _ = result_index("NFL", invalid)
    assert scores == {}


def test_unverified_archives_are_never_counted_as_executed_bets(tmp_path):
    archive(tmp_path, [candidate()])
    data = grade_archive(tmp_path, {"NFL": results_csv()}, NOW)
    assert not data["verified_profitability_proven"]
    assert not data["automatic_betting_enabled"]
    assert not data["paid_data_used"]
    assert "NOT independent official scores" in data["result_provenance"]

def test_cfb_abbreviated_ou_side_is_settled_correctly():
    actual = {
        "game_id": "2026_05_TB_DAL", "home_team": "DAL", "away_team": "TB",
        "kickoff": KICKOFF.isoformat(), "margin_home": 4.0, "total": 44.0,
    }
    record = {**candidate(market="total", side="U", line=47),
              "odds": -110, "observed_at_utc": START.isoformat()}
    assert grade_candidate(record, actual)["result"] == "WIN"
    record["side"] = "O"
    assert grade_candidate(record, actual)["result"] == "LOSS"
