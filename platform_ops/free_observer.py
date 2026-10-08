"""Free, read-only CFB/NFL research observer.

Never communicates with sportsbooks or sends betting orders. The only network
requests are GETs to already-public GitHub repository artifacts. Archive writes
go to local files; a separate GitHub workflow can commit them to a dedicated
history-only branch.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
from urllib.request import Request, urlopen

SOURCE_REPOS = {
    "CFB": "konnorharbin-star/harbin-sports-analytics",
    "NFL": "konnorharbin-star/harbin-nfl-analytics",
}
AUDIT_PATH = "docs/audit_snapshot.json"
PREDICTIONS_PATH = "docs/latest.csv"
USER_AGENT = "HarbinFreeResearchObserver/1.0"
MAX_WATCHLIST_PER_LEAGUE = 12
STALE_REPORT_HOURS = 36
STALE_QUOTE_MINUTES = 90
SCHEMA_VERSION = 1


def utc_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        date = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if date.tzinfo is None:
        return None
    return date.astimezone(timezone.utc)


def finite_number(value: object) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if v == v and abs(v) != float("inf") else None


def boolean(value: object) -> bool:
    return value is True or (isinstance(value, str) and value.lower() in ("true", "1"))


def status(value: object) -> str:
    return str(value or "UNKNOWN").upper()


def get_public_bytes(repo: str, path: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{repo}/main/{path}"
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/plain"})
    with urlopen(request, timeout=18) as response:  # noqa: S310 - fixed trusted host
        result = response.read(5_000_001)
    if len(result) > 5_000_000:
        raise ValueError(f"Refusing oversized public source {repo}:{path}")
    return result


def load_published_sources() -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    sources: dict[str, dict[str, object]] = {}
    hashes: dict[str, str] = {}
    for league, repo in SOURCE_REPOS.items():
        audit_raw = get_public_bytes(repo, AUDIT_PATH)
        predictions_raw = get_public_bytes(repo, PREDICTIONS_PATH)
        audit = json.loads(audit_raw)
        rows = list(csv.DictReader(io.StringIO(predictions_raw.decode("utf-8-sig"))))
        if not isinstance(audit, dict):
            raise ValueError(f"{league}: malformed audit")
        if not rows:
            raise ValueError(f"{league}: missing prediction rows")
        sources[league] = {"audit": audit, "rows": rows}
        hashes[league] = hashlib.sha256(audit_raw + b"\0" + predictions_raw).hexdigest()
    return sources, hashes


def american_profit(odds: object) -> float | None:
    number = finite_number(odds)
    if number is None or -100 < number < 100:
        return None
    return number / 100 if number >= 100 else 100 / abs(number)


def safe_candidate(
    row: dict[str, str], league: str, observed_at: datetime, audit_time: datetime | None,
    release_state: str, audit_ok: bool,
) -> dict[str, object] | None:
    """Preserve research observations; never convert them into betting permission."""
    signal = status(row.get("quant_signal"))
    if signal in {"", "UNKNOWN", "PASS", "NONE", "NO BET"}:
        return None
    kickoff = utc_datetime(row.get("kickoff") or row.get("date"))
    if kickoff is None or kickoff <= observed_at:
        return None
    probability = finite_number(row.get("quant_probability"))
    odds = finite_number(row.get("quant_odds"))
    payout = american_profit(odds)
    quote_at = utc_datetime(row.get("quant_quote_at"))
    market = str(row.get("quant_market") or "").lower()
    price = finite_number(row.get("quant_price"))
    if market not in {"moneyline", "spread", "total"}:
        return None
    # Keep source data distinct from a verified, executable order.
    reasons: list[str] = []
    if not audit_ok:
        reasons.append("published_audit_not_clean")
    if release_state != "PRODUCTION":
        reasons.append("model_not_production_validated")
    if probability is None or not 0 < probability < 1 or payout is None:
        reasons.append("invalid_probability_or_price")
    if market in {"spread", "total"} and price is None:
        reasons.append("missing_point_line")
    if quote_at is None:
        reasons.append("no_timestamped_quote")
    else:
        if quote_at > observed_at + timedelta(minutes=2):
            reasons.append("quote_future_of_observation")
        if quote_at >= kickoff:
            reasons.append("quote_not_pregame")
        if observed_at - quote_at > timedelta(minutes=STALE_QUOTE_MINUTES):
            reasons.append("quote_stale")
        if audit_time is not None and quote_at > audit_time + timedelta(minutes=2):
            reasons.append("quote_newer_than_published_model")
    if league == "NFL":
        if not boolean(row.get("market_execution_verified")) or not boolean(
            row.get("market_quote_timestamp_verified")
        ):
            reasons.append("unverified_market_quote")
        if not boolean(row.get("regime_reliability_ready")) or not boolean(
            row.get("probability_reliability_ready")
        ):
            reasons.append("unvalidated_nfl_regime")
        if boolean(row.get("context_veto")) or boolean(row.get("context_freshness_veto")):
            reasons.append("context_veto")
    else:
        if not boolean(row.get("execution_ready")):
            reasons.append("execution_context_not_ready")
    calculated_ev = (
        probability * payout - (1 - probability)
        if probability is not None and 0 < probability < 1 and payout is not None
        else None
    )
    published_ev = finite_number(row.get("quant_ev"))
    if calculated_ev is None or published_ev is None or abs(calculated_ev - published_ev) > 0.025:
        reasons.append("expected_value_disagreement")
    if not str(row.get("quant_book") or "").strip():
        reasons.append("missing_book")
    # No amount, bankroll, wallet, order endpoint or staking approval in output.
    return {
        "league": league,
        "game_id": str(row.get("game_id") or ""),
        "home_team": str(row.get("home_team") or ""),
        "away_team": str(row.get("away_team") or ""),
        "kickoff_utc": kickoff.isoformat(),
        "market": market,
        "side": str(row.get("quant_side") or ""),
        "line": price,
        "book": str(row.get("quant_book") or ""),
        "american_odds": odds,
        "model_probability": probability,
        "published_model_ev": published_ev,
        "recalculated_ev": calculated_ev,
        "quoted_at_utc": quote_at.isoformat() if quote_at else None,
        "source_signal": signal,
        "qualification": "RESEARCH_ONLY",
        "blockers": sorted(set(reasons)),
        "notes": "No betting execution is implemented. A positive model EV is not verified edge.",
    }


def build_snapshot(
    sources: dict[str, dict[str, object]],
    hashes: dict[str, str],
    observed_at: datetime,
) -> dict[str, object]:
    if observed_at.tzinfo is None:
        raise ValueError("Observation time must contain UTC offset")
    observed_at = observed_at.astimezone(timezone.utc)
    leagues: dict[str, object] = {}
    for league in SOURCE_REPOS:
        if league not in sources or league not in hashes:
            raise ValueError(f"Missing public source: {league}")
        audit = sources[league]["audit"]
        rows = sources[league]["rows"]
        if not isinstance(audit, dict) or not isinstance(rows, list):
            raise ValueError(f"{league}: malformed source objects")
        identity = audit.get("identity") or {}
        reconciliation = audit.get("reconciliation") or {}
        quality = audit.get("data_quality") or {}
        release = audit.get("release") or {}
        portfolio = audit.get("portfolio") or {}
        audit_time = utc_datetime(audit.get("generated_at"))
        expected_rows = finite_number(identity.get("prediction_rows"))
        row_count_matches = (
            expected_rows is not None and expected_rows.is_integer()
            and int(expected_rows) == len(rows)
        )
        identity_matches = all(
            str(row.get(key)) == str(identity[key])
            for row in rows
            for key in ("season", "week")
            if identity.get(key) is not None
        )
        source_contract_ok = row_count_matches and identity_matches
        fresh = (
            audit_time is not None
            and timedelta(minutes=-2) <= observed_at - audit_time <= timedelta(hours=STALE_REPORT_HOURS)
        )
        clean = (
            fresh
            and source_contract_ok
            and status(audit.get("status")) not in {"FAIL", "ERROR", "UNKNOWN"}
            and status(reconciliation.get("status")) == "PASS"
            and status(quality.get("status")) in {"OK", "PASS"}
        )
        release_state = status(release.get("state"))
        candidates = [
            candidate
            for row in rows
            if (candidate := safe_candidate(row, league, observed_at, audit_time, release_state, clean))
        ]
        candidates.sort(
            key=lambda item: (
                item["recalculated_ev"] is not None,
                item["recalculated_ev"] if item["recalculated_ev"] is not None else -99,
            ),
            reverse=True,
        )
        leagues[league] = {
            "source_repo": SOURCE_REPOS[league],
            "audit_generated_at": audit.get("generated_at"),
            "audit_status": status(audit.get("status")),
            "reconciliation_status": status(reconciliation.get("status")),
            "data_quality_status": status(quality.get("status")),
            "publication_fresh": fresh,
            "source_contract_status": "PASS" if source_contract_ok else "FAIL",
            "source_contract_detail": (
                f"prediction_rows published={identity.get('prediction_rows')} "
                f"downloaded={len(rows)}; season/week identity matches={identity_matches}"
            ),
            "model_version": identity.get("model_version"),
            "model_generated_at": identity.get("model_generated_at"),
            "season": identity.get("season"),
            "week": identity.get("week"),
            "release_state": release_state,
            "production_eligible": release.get("production_eligible") is True,
            "published_approved_units": portfolio.get("approved_units"),
            "raw_rows_read": len(rows),
            "research_candidates_found": len(candidates),
            "research_watchlist": candidates[:MAX_WATCHLIST_PER_LEAGUE],
            "release_blockers": (release.get("blockers") or [])[:10],
            "source_sha256": hashes[league],
            "summary": "Observational model evidence only, not bets to execute.",
        }
    snapshot_id = hashlib.sha256(
        json.dumps(
            {key: hashes[key] for key in sorted(hashes)},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "observed_at_utc": observed_at.isoformat(),
        "mode": "READ_ONLY_RESEARCH",
        "automatic_betting_enabled": False,
        "paid_data_sources_used": False,
        "leagues": leagues,
    }


def append_archive(snapshot: dict[str, object], root: Path) -> bool:
    """Append each published source version only once; never mutate existing rows."""
    observed = utc_datetime(snapshot.get("observed_at_utc"))
    digest = snapshot.get("snapshot_id")
    if observed is None or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Invalid snapshot identity")
    archive = root / f"{observed:%Y}" / f"{observed:%m}" / f"{observed:%d}.jsonl"
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Latest observation is overwritten only in this dedicated evidence branch.
    # Immutable date-sharded source versions remain in the JSONL files.
    (root / "latest.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if archive.exists():
        for line in archive.read_text().splitlines():
            if json.loads(line).get("snapshot_id") == digest:
                return False
    with archive.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n")
    return True


def write_sqlite_snapshot(snapshot: dict[str, object], path: Path) -> None:
    """Pure-stdlib portable SQLite export for analysis; no sportsbook integrations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS snapshots
                   (snapshot_id TEXT PRIMARY KEY, observed_at_utc TEXT NOT NULL,
                    league TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                    audit_status TEXT NOT NULL, release_state TEXT NOT NULL,
                    research_candidates_found INTEGER NOT NULL,
                    UNIQUE(snapshot_id, league))""")
        db.execute("""CREATE TABLE IF NOT EXISTS research_candidates
                   (snapshot_id TEXT NOT NULL, league TEXT NOT NULL,
                    game_id TEXT NOT NULL, market TEXT NOT NULL,
                    side TEXT NOT NULL, book TEXT NOT NULL,
                    quoted_at_utc TEXT, kickoff_utc TEXT NOT NULL,
                    american_odds REAL, line REAL, model_probability REAL,
                    recalculated_ev REAL, blockers_json TEXT NOT NULL,
                    qualification TEXT NOT NULL)""")
        for league, data in snapshot["leagues"].items():
            sid = f"{snapshot['snapshot_id']}:{league}"
            existing = db.execute("SELECT 1 FROM snapshots WHERE snapshot_id = ?", (sid,)).fetchone()
            if existing:
                continue
            db.execute(
                "INSERT INTO snapshots VALUES (?,?,?,?,?,?,?)",
                (sid, snapshot["observed_at_utc"], league, data["source_sha256"],
                 data["audit_status"], data["release_state"], data["research_candidates_found"]),
            )
            for item in data["research_watchlist"]:
                db.execute(
                    "INSERT INTO research_candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (sid, league, item["game_id"], item["market"], item["side"],
                     item["book"], item["quoted_at_utc"], item["kickoff_utc"],
                     item["american_odds"], item["line"], item["model_probability"],
                     item["recalculated_ev"], json.dumps(item["blockers"]), item["qualification"]),
                )


def research_report(snapshot: dict[str, object]) -> str:
    """Readable ranking for human review only; never claims executable wagers."""
    lines = [
        "# Harbin free model research — candidate report",
        "",
        "**READ-ONLY / NO AUTOMATIC WAGERS.** Rankings are research signals,",
        "not verified profitable bets or bookmaker execution instructions.",
        "",
        f"Observed at: {snapshot['observed_at_utc']}",
        f"Snapshot: `{snapshot['snapshot_id']}`",
        "",
    ]
    for league, data in snapshot["leagues"].items():
        lines.extend([
            f"## {league} — {data['release_state']}",
            "",
            f"Publication: {data['audit_status']} | Reconciliation: "
            f"{data['reconciliation_status']} | Fresh: {data['publication_fresh']}",
            f"Research candidates: {data['research_candidates_found']}",
            "",
            "**Validated best bets: NONE established by this observer.** "
            "The source models' calibration and execution quality need independent review.",
            "",
        ])
        candidates = data["research_watchlist"]
        if not candidates:
            lines.extend(["No current research candidates from published signals.", ""])
            continue
        lines.extend([
            "| Candidate | Published line/odds | Model EV | Research blockers |",
            "|---|---|---:|---|",
        ])
        for item in candidates[:8]:
            def safe(value: object) -> str:
                return str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")
            odds = item["american_odds"]
            odds_text = "—" if odds is None else f"{odds:+g}"
            line = "—" if item["line"] is None else f"{item['line']:g}"
            ev = item["recalculated_ev"]
            ev_text = "—" if ev is None else f"{ev * 100:+.1f}% (unvalidated)"
            flags = ", ".join(item["blockers"]) or "No data-quality flags; edge still unverified"
            match = f"{item['away_team']} @ {item['home_team']}: " \
                f"{item['side']} {item['market']}"
            lines.append(
                f"| {safe(match)} | {safe(item['book'])} {line} ({odds_text}) | "
                f"{ev_text} | {safe(flags)} |"
            )
        lines.extend([
            "",
            "Research ranking is based on source-model EV, which may be miscalibrated. "
            "Do not interpret this table as approval to wager.",
            "",
        ])
    return "\n".join(lines) + "\n"


def capture(output: Path, sqlite_path: Path, report_path: Path | None = None) -> None:
    sources, hashes = load_published_sources()
    snapshot = build_snapshot(sources, hashes, datetime.now(timezone.utc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_sqlite_snapshot(snapshot, sqlite_path)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(research_report(snapshot))
    print(json.dumps({
        "snapshot_id": snapshot["snapshot_id"],
        "mode": snapshot["mode"],
        "auto_bet": snapshot["automatic_betting_enabled"],
        "data_cost": "public sources only",
        "leagues": {
            league: {
                "status": data["audit_status"],
                "fresh": data["publication_fresh"],
                "research_candidates": data["research_candidates_found"],
            } for league, data in snapshot["leagues"].items()
        },
    }, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Free read-only sports-model evidence capture")
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("capture")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--sqlite", type=Path, required=True)
    collect.add_argument("--report", type=Path)
    archive = sub.add_parser("archive")
    archive.add_argument("--snapshot", type=Path, required=True)
    archive.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        capture(args.output, args.sqlite, args.report)
    else:
        snapshot = json.loads(args.snapshot.read_text())
        print("appended" if append_archive(snapshot, args.archive_root) else "duplicate_snapshot_skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
