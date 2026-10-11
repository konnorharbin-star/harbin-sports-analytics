"""Every-game FBS market-edge evidence report, never a manufactured betting edge.

The model already estimates every pregame FBS-involved game, while its
final-edge shortlist can contain only one or no games. Merge by stable ESPN
game_id to disclose the full independent projection and the candidate
evidence and blockers for every game. This output cannot approve bets or
replace the price, validation, portfolio or production release gates.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from scripts.final_edge_board import timestamp, yes

FIELDS = (
    "season", "week", "game_id", "kickoff", "away_team", "home_team",
    "fbs_matchup_scope", "fbs_model_validation", "model_away_score",
    "model_home_score", "model_home_margin", "model_total",
    "published_market_home_spread", "spread_disagreement_points",
    "model_candidate_market", "model_candidate_side", "model_candidate_line",
    "model_candidate_american_odds", "model_candidate_book",
    "raw_model_ev_hypothesis", "model_candidate_signal",
    "final_edge_shortlisted", "conservative_edge_ev",
    "final_edge_decision", "source_quote_timestamp_verified",
    "book_execution_verified", "release_gate_state",
    "research_status", "recommended_bet", "blockers",
)


def number(v):
    if isinstance(v, bool) or v is None or v == "":
        return None
    try:
        value = float(v)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _table(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def _ids(rows, field):
    names = [str(r.get(field, "")).strip() for r in rows]
    if len(names) != len(set(names)) or not all(names):
        raise ValueError("Duplicate or missing game_id in " + field + " source")
    return set(names)


def build(model, final, coverage, gate, *, evaluated_at):
    """Produce a coverage-reconciled research report for all pregame FBS games."""
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        raise ValueError("Audit evaluation time must be timezone-aware")
    if coverage.get("scope") != "EVERY_FBS_INVOLVED_REGULAR_SEASON_MATCHUP":
        raise ValueError("Full-FBS coverage audit has wrong scope")
    if coverage.get("remaining_pregame_coverage_complete") is not True:
        raise ValueError("Cannot publish incomplete FBS slate edge report")
    if "production_eligible" not in gate or "release_state" not in gate:
        raise ValueError("Missing authoritative market/betting release gate")
    source_games = coverage.get("source_game_status")  # only if source list supplied
    if source_games is not None and not isinstance(source_games, list):
        raise ValueError("Invalid source coverage status")
    ids = _ids(model, "game_id")
    final_ids = _ids(final, "game_id")
    if not final_ids.issubset(ids):
        raise ValueError("Final edge shortlist includes game not in pregame FBS slate")
    counts = coverage.get("counts", {})
    if (len(model) != int(counts.get("PREDICTED", -1))
            or len(model) != int(coverage.get("remaining_pregame_games", -1))):
        raise ValueError("Model row count mismatches full FBS schedule audit")
    allowed_scopes = {"FBS_VS_FBS", "FBS_VS_NON_FBS_UNVALIDATED"}
    final_by_id = {f["game_id"]: f for f in final}
    out = []
    for m in model:
        gid = m["game_id"]
        scope = m.get("fbs_matchup_scope")
        if scope not in allowed_scopes:
            raise ValueError(f"Missing validated FBS opponent classification: {gid}")
        if not str(m.get("date") or ""):
            raise ValueError("Missing kickoff on full FBS slate")
        kickoff = timestamp(m["date"])
        if kickoff is None:
            raise ValueError("Invalid timezone-aware kickoff on game " + gid)
        audit_cutoff = timestamp(coverage.get("model_cutoff_utc"))
        if audit_cutoff is None or kickoff <= audit_cutoff:
            raise ValueError("Current model includes already-started game " + gid)
        line = number(m.get("market_spread_home"))
        margin = number(m.get("model_margin_home"))
        total = number(m.get("model_total"))
        away_score = number(m.get("away_score_exact"))
        home_score = number(m.get("home_score_exact"))
        if None in (margin, total, away_score, home_score):
            raise ValueError("No complete independent model forecast for " + gid)
        if abs((home_score-away_score) - margin) > 0.05:
            raise ValueError("Score/margin inconsistency on " + gid)
        if abs((home_score+away_score)-total) > 0.05:
            raise ValueError("Score/total inconsistency on " + gid)
        short = final_by_id.get(gid)
        blockers = []
        unvalidated = scope == "FBS_VS_NON_FBS_UNVALIDATED"
        raw_ev = number(m.get("quant_ev"))
        raw_signal = str(m.get("quant_signal") or "PASS").upper()
        if unvalidated:
            blockers.append("UNVALIDATED_FBS_NON_FBS_MATCHUP")
        if line is None:
            blockers.append("NO_PUBLISHED_MARKET_SPREAD")
        if raw_ev is None:
            blockers.append("NO_PRICED_MODEL_CANDIDATE")
        elif raw_ev <= 0:
            blockers.append("NO_POSITIVE_RAW_MODEL_EV")
        if raw_signal not in ("LEAN", "BET", "STRONG"):
            blockers.append("NO_QUALIFIED_RAW_MODEL_SIGNAL")
        if not short:
            blockers.append("NOT_IN_EVIDENCE_QUALIFIED_EDGE_SHORTLIST")
        else:
            blockers.extend(x for x in str(short.get("reasons") or "").split(";") if x)
            if str(short.get("decision") or "") != "BET_READY":
                blockers.append("SHORTLIST_NOT_BET_READY")
        if not yes(m.get("market_quote_timestamp_verified")):
            blockers.append("PROVIDER_TIMESTAMP_UNVERIFIED")
        if not yes(m.get("market_execution_verified")):
            blockers.append("BOOK_EXECUTION_UNVERIFIED")
        if not yes(gate.get("production_eligible")):
            blockers.append("GLOBAL_MODEL_RELEASE_NOT_VALIDATED")
        # A scheduled research run NEVER authorizes or places wagers.
        # Conserve full independent market evidence with no overstatement.
        plausible = bool(
            not unvalidated and raw_ev is not None and raw_ev > 0
            and raw_signal in ("LEAN", "BET", "STRONG")
        )
        status = (
            "RAW_EDGE_HYPOTHESIS_UNVERIFIED" if plausible
            else "NO_MODEL_EDGE_HYPOTHESIS"
        )
        if unvalidated:
            status = "UNVALIDATED_OPPONENT_CLASS_NO_BET"
        row = {
            "season": m.get("season"), "week": m.get("week"),
            "game_id": gid, "kickoff": m["date"],
            "away_team": m.get("away_team"), "home_team": m.get("home_team"),
            "fbs_matchup_scope": scope,
            "fbs_model_validation": m.get("fbs_model_validation"),
            "model_away_score": away_score, "model_home_score": home_score,
            "model_home_margin": margin, "model_total": total,
            "published_market_home_spread": line,
            "spread_disagreement_points": margin+line if line is not None else None,
            "model_candidate_market": m.get("quant_market"),
            "model_candidate_side": m.get("quant_side"),
            "model_candidate_line": number(m.get("quant_price")),
            "model_candidate_american_odds": number(m.get("quant_odds")),
            "model_candidate_book": m.get("quant_book"),
            "raw_model_ev_hypothesis": raw_ev,
            "model_candidate_signal": raw_signal,
            "final_edge_shortlisted": bool(short),
            "conservative_edge_ev": number(short.get("conservative_ev")) if short else None,
            "final_edge_decision": short.get("decision") if short else "NOT_SHORTLISTED",
            "source_quote_timestamp_verified": yes(m.get("market_quote_timestamp_verified")),
            "book_execution_verified": yes(m.get("market_execution_verified")),
            "release_gate_state": gate["release_state"],
            "research_status": status,
            "recommended_bet": False,
            "blockers": ";".join(dict.fromkeys(blockers)),
        }
        out.append(row)
    out.sort(key=lambda x: (x["kickoff"], x["game_id"]))
    statues = Counter(x["research_status"] for x in out)
    return out, {
        "spec": "walters_complete_fbs_edge_scan_v1",
        "status": "FULL_SLATE_RESEARCH_ONLY_NO_VALIDATED_PROFIT",
        "evaluated_at": evaluated_at.astimezone(UTC).isoformat(),
        "season": coverage.get("season"), "week": coverage.get("week"),
        "total_fbs_involved_weekly_games": coverage.get("total_scheduled_fbs_games"),
        "remaining_pregame_game_count": len(out),
        "started_or_completed_excluded": int(counts.get("STARTED_EXCLUDED", 0))
        + int(counts.get("COMPLETED_EXCLUDED", 0)),
        "games_with_raw_positive_model_hypothesis": statues.get(
            "RAW_EDGE_HYPOTHESIS_UNVERIFIED", 0),
        "games_shortlisted_by_independent_edge_gate": len(final),
        "games_without_reliable_quote_timestamp": sum(
            not x["source_quote_timestamp_verified"] for x in out),
        "games_without_verified_book_execution": sum(
            not x["book_execution_verified"] for x in out),
        "classification_counts": dict(statues),
        "every_upcoming_fbs_game_scanned": True,
        "all_games_proven_positively_priced": False,
        "model_improves_market_probability_validated": False,
        "historical_verified_profitable_edge_proven": False,
        "full_slate_betting_eligible": False,
        "automated_wagers_placed": 0,
        "bet_recommendations": 0,
        "production_release_state": gate.get("release_state"),
        "production_eligible": yes(gate.get("production_eligible")),
        "release_blockers": gate.get("blockers", [])[:10],
        "pricing_limitations": (
            "Raw per-game quant_ev estimates use the production probability "
            "engine and are not proven accurate; archived market quotes are "
            "not independently book-executable, and integer line pushes "
            "must be treated as win/push/loss when verified."
        ),
        "selection_notice": (
            "This ledger includes every FBS pregame matchup, including no-price "
            "and no-shortlist games, but does not turn a raw EV discrepancy "
            "into a validated bet. Research classification is not a bet."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--shortlist", type=Path, required=True)
    p.add_argument("--coverage", type=Path, required=True)
    p.add_argument("--gate", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()
    source = _table(args.model)
    shortlist = _table(args.shortlist)
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    result, summary = build(
        source, shortlist, coverage, gate, evaluated_at=datetime.now(UTC)
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(result)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_fbs_games_scanned": summary["remaining_pregame_game_count"],
        "research_hypotheses": summary["games_with_raw_positive_model_hypothesis"],
        "final_edge_shortlist": summary[
            "games_shortlisted_by_independent_edge_gate"
        ],
        "verified_profitable_edges": 0,
        "bets_recommended": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
