"""Read-only free observer tests: timestamp, price, risk and persistence checks."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

from platform_ops.free_observer import (
    american_profit,
    append_archive,
    build_snapshot,
    safe_candidate,
    research_report,
    utc_datetime,
    write_sqlite_snapshot,
)


NOW = datetime(2026, 10, 8, 22, 30, tzinfo=timezone.utc)


def audit(league: str, *, status: str = "WARN", recon: str = "PASS",
          generated: datetime | None = None) -> dict:
    stamp = (generated or NOW - timedelta(minutes=5)).isoformat()
    return {
        "generated_at": stamp,
        "status": status,
        "identity": {"season": 2026, "week": 6 if league == "CFB" else 5},
        "reconciliation": {"status": recon},
        "data_quality": {"status": "PASS" if league == "CFB" else "OK"},
        "release": {"state": "RESEARCH", "production_eligible": False,
                    "blockers": ["historical validation incomplete"]},
        "portfolio": {"approved_units": 0},
    }


def row(league: str) -> dict[str, str]:
    result = {
        "game_id": "g1", "home_team": "DAL", "away_team": "TB",
        "date": "2026-10-09T00:15:00+00:00",
        "kickoff": "2026-10-09T00:15:00+00:00",
        "quant_signal": "STRONG" if league == "CFB" else "BET",
        "quant_market": "total",
        "quant_side": "under", "quant_book": "Samplebook",
        "quant_price": "49.5", "quant_odds": "-110",
        "quant_probability": "0.55",
        "quant_ev": str(0.55 * 100/110 - 0.45),
        "quant_quote_at": "2026-10-08T22:21:00+00:00",
        "execution_ready": "True",
        "market_execution_verified": "True",
        "market_quote_timestamp_verified": "True",
        "probability_reliability_ready": "True",
        "regime_reliability_ready": "True",
    }
    return result


def inputs(*, nfl_recon: str = "PASS") -> tuple[dict, dict]:
    sources = {league: {"audit": audit(league, recon=nfl_recon), "rows": [row(league)]}
               for league in ("NFL", "CFB")}
    return sources, {"NFL": "a" * 64, "CFB": "b" * 64}


def test_odds_and_timestamp_rules() -> None:
    assert american_profit(-110) == 100/110
    assert american_profit(200) == 2.0
    assert american_profit(-99) is None
    assert american_profit(0) is None
    assert utc_datetime("2026-10-08T22:21:00+00:00") is not None
    assert utc_datetime("2026-10-08T22:21:00") is None


def test_research_only_even_for_high_model_ev() -> None:
    sources, hashes = inputs()
    snapshot = build_snapshot(sources, hashes, NOW)
    assert snapshot["automatic_betting_enabled"] is False
    assert snapshot["paid_data_sources_used"] is False
    assert snapshot["mode"] == "READ_ONLY_RESEARCH"
    for league in ("CFB", "NFL"):
        data = snapshot["leagues"][league]
        assert data["release_state"] == "RESEARCH"
        assert data["production_eligible"] is False
        assert len(data["research_watchlist"]) == 1
        candidate = data["research_watchlist"][0]
        assert candidate["qualification"] == "RESEARCH_ONLY"
        assert "model_not_production_validated" in candidate["blockers"]
        assert candidate["recalculated_ev"] > 0
        assert "stake" not in candidate
        assert "order" not in candidate


def test_malformed_and_post_kickoff_quotes_are_blocked() -> None:
    entry = row("NFL")
    entry["quant_quote_at"] = "2026-10-09T00:15:00+00:00"
    entry["quant_odds"] = "-99"
    candidate = safe_candidate(entry, "NFL", NOW, NOW, "RESEARCH", True)
    assert candidate is not None
    assert "quote_future_of_observation" in candidate["blockers"]
    assert "quote_not_pregame" in candidate["blockers"]
    assert "invalid_probability_or_price" in candidate["blockers"]

    entry["kickoff"] = "2026-10-08T21:00:00+00:00"
    assert safe_candidate(entry, "NFL", NOW, NOW, "RESEARCH", True) is None


def test_no_untimestamped_market_promotions() -> None:
    entry = row("CFB")
    entry["quant_quote_at"] = ""
    entry["quant_probability"] = "0.55"
    candidate = safe_candidate(entry, "CFB", NOW, NOW, "PRODUCTION", True)
    assert candidate is not None
    assert "no_timestamped_quote" in candidate["blockers"]
    assert candidate["qualification"] == "RESEARCH_ONLY"


def test_regime_veto_blocks_nfl_candidate() -> None:
    entry = row("NFL")
    entry["regime_reliability_ready"] = "False"
    entry["context_veto"] = "True"
    candidate = safe_candidate(entry, "NFL", NOW, NOW, "PRODUCTION", True)
    assert candidate is not None
    assert "unvalidated_nfl_regime" in candidate["blockers"]
    assert "context_veto" in candidate["blockers"]


def test_audit_failure_and_stale_publication_are_recorded() -> None:
    sources, hashes = inputs(nfl_recon="FAIL")
    snapshot = build_snapshot(sources, hashes, NOW)
    nfl = snapshot["leagues"]["NFL"]
    assert nfl["reconciliation_status"] == "FAIL"
    assert "published_audit_not_clean" in nfl["research_watchlist"][0]["blockers"]
    sources["NFL"]["audit"] = audit("NFL", generated=NOW - timedelta(days=3))
    snapshot = build_snapshot(sources, hashes, NOW)
    assert snapshot["leagues"]["NFL"]["publication_fresh"] is False


def test_duplicate_snapshots_are_not_appended_twice(tmp_path: Path) -> None:
    snapshot = build_snapshot(*inputs(), observed_at=NOW)
    assert append_archive(snapshot, tmp_path)
    assert not append_archive(snapshot, tmp_path)
    files = list(tmp_path.rglob("*.jsonl"))
    assert len(files) == 1
    assert len(files[0].read_text().splitlines()) == 1


def test_sqlite_is_idempotent_and_never_contains_bet_orders(tmp_path: Path) -> None:
    snapshot = build_snapshot(*inputs(), observed_at=NOW)
    database = tmp_path / "observer.sqlite"
    write_sqlite_snapshot(snapshot, database)
    write_sqlite_snapshot(snapshot, database)
    with sqlite3.connect(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM research_candidates").fetchone()[0] == 2
        values = conn.execute("SELECT DISTINCT qualification FROM research_candidates").fetchall()
    assert values == [("RESEARCH_ONLY",)]


def test_observer_has_no_sportsbook_placement_or_paid_services() -> None:
    source = Path("platform_ops/free_observer.py").read_text()
    assert "urlopen(request" in source
    assert "raw.githubusercontent.com" in source
    for banned in ("place_bet(", "submit_wager(", "bet_slip(", "Stripe", "THE_ODDS_API_KEY"):
        assert banned not in source
    assert "automatic_betting_enabled" in source


def test_report_shows_ranked_research_but_not_approved_wagers() -> None:
    sources, hashes = inputs()
    doc = research_report(build_snapshot(sources, hashes, NOW))
    assert "# Harbin free model research" in doc
    assert "NO AUTOMATIC WAGERS" in doc
    assert "Validated best bets: NONE" in doc
    assert "RESEARCH" in doc
    assert "model_not_production_validated" in doc
    assert "Samplebook" in doc
    assert "unvalidated" in doc
