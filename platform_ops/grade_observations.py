"""Read-only forward grading of archived model suggestions against FREE result ledgers.

The outcome ledgers are the models' existing public postgame files, NOT an
independently verified official scoreboard. No wagering, wallets, or paid APIs.
Only archives observed before kickoff can become hypothetical graded entries.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
from typing import Any

from platform_ops.free_observer import (
    SOURCE_REPOS, american_profit, finite_number, get_public_bytes, utc_datetime
)
from platform_ops.score_verification import verification_index

RESULTS_PATH = "reports/live_graded_bets.csv"
MAX_KICKOFF_DIFF = timedelta(minutes=5)


def result_index(league: str, text: str) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """One unique unambiguous completed outcome per game, or omit it entirely."""
    entries: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for row in csv.DictReader(io.StringIO(text)):
        game = str(row.get("game_id") or "").strip()
        margin = finite_number(row.get("actual_margin_home"))
        total = finite_number(row.get("actual_total"))
        kickoff = utc_datetime(row.get("kickoff"))
        away = str(row.get("away_team") or "").strip()
        home = str(row.get("home_team") or "").strip()
        if not game or kickoff is None or margin is None or total is None or not away or not home:
            continue
        if total < 0 or abs(margin) > total or total > 160 or abs(margin) > 100:
            conflicts.add(game)
            continue
        if str(row.get("result") or "").strip().lower() not in {"win", "loss", "push", "1", "-1", "0"}:
            continue
        record = {
            "game_id": game, "home_team": home, "away_team": away,
            "kickoff": kickoff.isoformat(), "margin_home": margin, "total": total,
            "league": league,
        }
        existing = entries.get(game)
        if existing is not None and existing != record:
            conflicts.add(game)
        else:
            entries[game] = record
    for game in conflicts:
        entries.pop(game, None)
    return entries, conflicts


def collect_archived_candidates(
    archive_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Freeze the earliest pregame candidate per league/game/market/side.

    This selection rule is specified in advance and never considers outcomes.
    """
    chosen: dict[tuple[str, str, str], dict[str, Any]] = {}
    counts = {"snapshots": 0, "candidate_rows": 0, "invalid_timing": 0,
              "missing_quote_or_price": 0, "malformed_snapshots": 0}
    for file in sorted(archive_root.glob("20??/??/??.jsonl")):
        for line in file.read_text().splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                counts["malformed_snapshots"] += 1
                continue
            if (payload.get("schema_version") != 1
                or payload.get("mode") != "READ_ONLY_RESEARCH"
                or payload.get("automatic_betting_enabled") is not False
                or not isinstance(payload.get("leagues"), dict)):
                counts["malformed_snapshots"] += 1
                continue
            observed = utc_datetime(payload.get("observed_at_utc"))
            if observed is None:
                counts["malformed_snapshots"] += 1
                continue
            counts["snapshots"] += 1
            for league in ("CFB", "NFL"):
                state = payload["leagues"].get(league, {})
                rows = state.get("research_watchlist", [])
                if not isinstance(rows, list):
                    continue
                for c in rows:
                    counts["candidate_rows"] += 1
                    if not isinstance(c, dict):
                        counts["malformed_snapshots"] += 1
                        continue
                    kickoff = utc_datetime(c.get("kickoff_utc"))
                    quote = utc_datetime(c.get("quoted_at_utc"))
                    if kickoff is None or kickoff <= observed or (
                        quote is not None and (quote > observed + timedelta(minutes=2)
                                               or quote >= kickoff)):
                        counts["invalid_timing"] += 1
                        continue
                    odds = finite_number(c.get("american_odds"))
                    payout = american_profit(odds)
                    prob = finite_number(c.get("model_probability"))
                    market = str(c.get("market") or "")
                    line_val = finite_number(c.get("line"))
                    if (quote is None or observed - quote > timedelta(minutes=90)
                        or payout is None or prob is None or not 0 < prob < 1
                        or market not in ("spread", "total", "moneyline")
                        or (market in ("spread", "total") and line_val is None)
                        or not str(c.get("book") or "").strip()):
                        counts["missing_quote_or_price"] += 1
                        continue
                    game = str(c.get("game_id") or "")
                    side = str(c.get("side") or "").strip().lower()
                    if not game or not side:
                        counts["missing_quote_or_price"] += 1
                        continue
                    # At most one opportunity per game × market. Lock first seen;
                    # same-timestamp ties ordered deterministically by source row.
                    key = league, game, market
                    frozen = {
                        "league": league, "game_id": game, "market": market,
                        "side": c.get("side"), "line": line_val, "odds": odds,
                        "book": c.get("book"), "quoted_at_utc": quote.isoformat(),
                        "observed_at_utc": observed.isoformat(), "kickoff_utc": kickoff.isoformat(),
                        "home_team": c.get("home_team"), "away_team": c.get("away_team"),
                        "model_probability": prob, "model_ev": c.get("recalculated_ev"),
                        "blockers": c.get("blockers") or [],
                        "snapshot_id": payload.get("snapshot_id"),
                        "source_contract_status": state.get("source_contract_status"),
                        "source_audit_status": state.get("audit_status"),
                        "source_reconciliation_status": state.get("reconciliation_status"),
                    }
                    prev = chosen.get(key)
                    if prev is None or observed.isoformat() < prev["observed_at_utc"]:
                        chosen[key] = frozen
    return list(chosen.values()), counts


def resolve_side(side: str, home: str, away: str) -> str | None:
    """Accept exact team identity or HOME/AWAY, never approximate team names."""
    value = side.casefold().strip()
    if value in ("home", home.casefold()): return "home"
    if value in ("away", away.casefold()): return "away"
    return None


def grade_candidate(candidate: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    """Grade from final scores; return None when team/timing identity is uncertain."""
    if candidate["game_id"] != result["game_id"]:
        return None
    if (str(candidate["home_team"]).casefold() != str(result["home_team"]).casefold()
        or str(candidate["away_team"]).casefold() != str(result["away_team"]).casefold()):
        return None
    c_kickoff = utc_datetime(candidate.get("kickoff_utc"))
    r_kickoff = utc_datetime(result.get("kickoff"))
    observed = utc_datetime(candidate.get("observed_at_utc"))
    if c_kickoff is None or r_kickoff is None or observed is None:
        return None
    if abs(c_kickoff - r_kickoff) > MAX_KICKOFF_DIFF or observed >= c_kickoff:
        return None
    market = candidate["market"]
    margin = result["margin_home"]
    total = result["total"]
    line = candidate["line"]
    side = str(candidate["side"])
    if market == "spread":
        role = resolve_side(side, result["home_team"], result["away_team"])
        if role is None or line is None: return None
        diff = (margin if role == "home" else -margin) + line
    elif market == "total":
        selection = side.casefold().strip()
        if line is None or selection not in ("over", "under", "o", "u"): return None
        diff = (total - line) * (1 if selection in ("over", "o") else -1)
    elif market == "moneyline":
        role = resolve_side(side, result["home_team"], result["away_team"])
        if role is None or margin == 0: return None  # Tie settlement varies by book.
        diff = margin if role == "home" else -margin
    else:
        return None
    outcome = "WIN" if diff > 1e-8 else "LOSS" if diff < -1e-8 else "PUSH"
    # Accept the frozen archive field or a source-shaped research fixture.
    # Missing/malformed odds fail closed instead of raising during grading.
    payout = american_profit(candidate.get("odds", candidate.get("american_odds")))
    if payout is None: return None
    profit = payout if outcome == "WIN" else -1.0 if outcome == "LOSS" else 0.0
    return {
        **candidate, "result": outcome, "hypothetical_profit_units": profit,
        "actual_margin_home": margin, "actual_total": total,
        "outcome_source": f"{SOURCE_REPOS[candidate['league']]}/{RESULTS_PATH}",
        "verified_executable_entry": False, "bet_was_placed": False,
        "evidence_tier": "RESEARCH_OUTCOME_ONLY",
    }


def grade_archive(
    archive_root: Path, results: dict[str, str],
    as_of: datetime,
    *,
    independent_scores: dict[tuple[str, str], dict[str, Any]] | None = None,
    require_independent: bool = False,
    independent_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if as_of.tzinfo is None: raise ValueError("as_of must be timezone aware")
    candidates, audit_counts = collect_archived_candidates(archive_root)
    outcome_index, conflicting = {}, {}
    for league in ("CFB", "NFL"):
        outcome_index[league], conflicting[league] = result_index(
            league, results.get(league, "")
        )
    graded: list[dict[str, Any]] = []
    pending = 0
    unknown = 0
    independent_missing = 0
    independent_disagreement = 0
    for candidate in sorted(candidates, key=lambda c: (
        c["league"], c["kickoff_utc"], c["game_id"], c["market"]
    )):
        kickoff = utc_datetime(candidate["kickoff_utc"])
        if kickoff is None or kickoff >= as_of:
            pending += 1
            continue
        result = outcome_index[candidate["league"]].get(candidate["game_id"])
        if result is None:
            unknown += 1
            continue
        if require_independent:
            second = (independent_scores or {}).get(
                (candidate["league"], str(candidate["game_id"]))
            )
            if second is None:
                independent_missing += 1
                continue
            same_scores = (
                result["margin_home"] == second["margin_home"]
                and result["total"] == second["total"]
                and abs(utc_datetime(result["kickoff"]) -
                        utc_datetime(second["kickoff"])) <= MAX_KICKOFF_DIFF
                and str(result["home_team"]).casefold() ==
                    str(second["home_team"]).casefold()
                and str(result["away_team"]).casefold() ==
                    str(second["away_team"]).casefold()
            )
            if not same_scores:
                independent_disagreement += 1
                continue
        grade = grade_candidate(candidate, result)
        if grade is None:
            unknown += 1
            continue
        grade["independent_score_verified"] = require_independent
        grade["independent_score_source"] = (
            second["score_source"] if require_independent else None
        )
        grade["espn_event_id"] = (
            second["espn_event_id"] if require_independent else None
        )
        graded.append(grade)
    summary: dict[str, dict[str, Any]] = {}
    for league in ("CFB", "NFL"):
        rows = [x for x in graded if x["league"] == league]
        decisive = [x for x in rows if x["result"] != "PUSH"]
        units = sum(float(x["hypothetical_profit_units"]) for x in rows)
        summary[league] = {
            "graded_observations": len(rows),
            "wins": sum(x["result"] == "WIN" for x in rows),
            "losses": sum(x["result"] == "LOSS" for x in rows),
            "pushes": sum(x["result"] == "PUSH" for x in rows),
            "hypothetical_flat_unit_roi": units / len(rows) if rows else None,
            "units_if_flat_staked": units if rows else None,
            "win_rate_excluding_pushes": (
                sum(x["result"] == "WIN" for x in rows) / len(decisive)
                if decisive else None
            ),
            "qualified_profitability_proven": False,
            "result_source_conflicts": len(conflicting[league]),
        }
    return {
        "schema_version": 1,
        "as_of_utc": as_of.astimezone(timezone.utc).isoformat(),
        "mode": "POSTGAME_RESEARCH_AUDIT",
        "automatic_betting_enabled": False,
        "paid_data_used": False,
        "result_provenance": (
            "Existing model result ledger cross-checked against final-score"
            " ESPN public scoreboard (unofficial API)" if require_independent else
            "model-authored graded-bets ledgers, NOT independently confirmed"
        ),
        "independent_score_verification_required": require_independent,
        "independent_score_report": independent_meta or {},

        "sample_limitation": (
            "First-seen research watchlist candidates only; subject to upstream selection "
            "and incomplete outcome coverage. Not representative of all games or executable bets."
        ),
        "verified_profitability_proven": False,
        "unresolved": {
            "pending_games": pending,
            "model_outcomes_unavailable_or_mismatch": unknown,
            "independent_scores_missing": independent_missing,
            "independent_scores_disagree": independent_disagreement,
        },
        "archive_integrity": audit_counts,
        "summary": summary,
        "graded": graded,
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Free postgame research audit",
        "",
        "**No wagers placed. No automatic betting. All ROI here is hypothetical.**",
        "",
        ("Only rows matching both the existing model's grading ledger and a "
         "completed ESPN public scoreboard are graded. ESPN's endpoint is "
         "undocumented, not an official league certification."
         if report["independent_score_verification_required"] else
         "Model-authored results only; no independent score confirmation."),
        "",
        "| League | Graded observations | W / L / P | Hypothetical flat-stake ROI |",
        "|---|---:|---:|---:|",
    ]
    for league in ("NFL", "CFB"):
        stat = report["summary"][league]
        roi = stat["hypothetical_flat_unit_roi"]
        rate = "N/A" if roi is None else f"{roi:+.1%}"
        lines.append(
            f"| {league} | {stat['graded_observations']} | "
            f"{stat['wins']}/{stat['losses']}/{stat['pushes']} | {rate} |"
        )
    pending = report["unresolved"]
    lines += [
        "",
        f"Independent scores unavailable: {pending['independent_scores_missing']}; "
        f"independent score disagreements: {pending['independent_scores_disagree']}.",
        "",
        "Not a validated betting edge. Late/missing/ambiguous"
              " quotes or results are omitted; no profitability promotion.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Free human-only research outcome grader")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args(argv)
    results = {
        league: get_public_bytes(repo, RESULTS_PATH).decode("utf-8-sig")
        for league, repo in SOURCE_REPOS.items()
    }
    as_of = datetime.now(timezone.utc)
    frozen, _counts = collect_archived_candidates(args.archive_root)
    # Only request independently reported finals for games whose kickoff passed.
    # This avoids unnecessary free-source calls for future games.
    completed = [
        row for row in frozen
        if (kickoff := utc_datetime(row.get("kickoff_utc"))) is not None
        and kickoff < as_of
    ]
    independent_scores, verification_meta = verification_index(completed)
    data = grade_archive(
        args.archive_root, results, as_of,
        independent_scores=independent_scores,
        require_independent=True,
        independent_meta=verification_meta,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False))
    args.markdown_out.write_text(report_markdown(data))
    print(json.dumps({
        "summary": data["summary"],
        "pending": data["unresolved"],
        "automatic_betting_enabled": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
