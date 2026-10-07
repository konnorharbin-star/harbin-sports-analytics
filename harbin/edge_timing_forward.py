"""Pre-registered, immutable forward evaluation of research-only CFB timing actions.

The signal is frozen *before* later market observations. All comparisons use
observed, same-book spread offers, not fills or certified sportsbook closing lines.
No model inputs, betting decisions, staking or production gates are modified.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .edge_timing import (
    _captured_book_observations,
    _implied,
    _num,
    _observations,
    _ts,
    MAX_QUOTE_AGE_MINUTES,
)

SCHEMA_VERSION = 1
TIMING_ACTIONS = frozenset({"BET_NOW_RESEARCH", "WAIT_MONITOR"})
PRIMARY_HORIZON_HOURS = 6
PRIMARY_TOLERANCE_HOURS = 3
NEAR_KICKOFF_WINDOW_MINUTES = 60
MIN_REVIEW_OBSERVATIONS = 100
MIN_REVIEW_DISTINCT_GAMES = 80
MIN_REVIEW_DISTINCT_WEEKS = 8
LEDGER_COLUMNS = [
    "schema_version", "decision_id", "decision_at", "game_id", "kickoff",
    "away_team", "home_team", "market", "side", "book", "quote_at",
    "entry_line", "entry_odds", "entry_implied_probability", "model_probability",
    "model_ev", "model_edge", "bet_to_line", "conservative_price_margin",
    "edge_priority", "timing_action", "timing_reason", "timing_evidence_source",
    "timing_previous_line", "timing_previous_odds", "timing_last_observed_at",
    "decision_mode",
]
GRADED_COLUMNS = LEDGER_COLUMNS + [
    "primary_status", "primary_observed_at", "primary_source",
    "primary_line", "primary_odds", "primary_line_change",
    "primary_implied_change_pp", "primary_movement",
    "primary_action_correct",
    "near_kickoff_status", "near_kickoff_observed_at", "near_kickoff_source",
    "near_kickoff_line", "near_kickoff_odds", "near_kickoff_line_change",
    "near_kickoff_implied_change_pp", "near_kickoff_movement",
    "near_kickoff_action_correct",
]


def _read_csv(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _safe_number(value):
    n = _num(value)
    return float(n) if math.isfinite(n) else None


def _string(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _decision_key(row):
    # One pre-registered primary decision per game / market / selected side,
    # regardless of later book-shopping, priority upgrades or model reruns.
    return "|".join((str(row["game_id"]), str(row["market"]), str(row["side"])))


def append_timing_decisions(
    priority_edges, path="history/timing_decisions_v1.csv", observed_at=None
):
    """Freeze the FIRST clean research action; append-only, no retroactive signals.

    A failed row never overwrites an earlier valid decision. Skipped candidates
    can be considered on later runs until their first clean signal is frozen.
    """
    when = _ts(observed_at if observed_at is not None else datetime.now(timezone.utc))
    if pd.isna(when):
        raise ValueError("observed_at must be a valid UTC timestamp")
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    old = _read_csv(file)
    if not old.empty and not set(LEDGER_COLUMNS).issubset(old.columns):
        raise ValueError("Existing timing decision ledger has incompatible schema")
    seen = set(old["decision_id"].astype(str)) if not old.empty else set()
    new = []
    if priority_edges is not None and not priority_edges.empty:
        for _, row in priority_edges.iterrows():
            if _string(row.get("timing_action")) not in TIMING_ACTIONS:
                continue
            if _string(row.get("timing_status")) != "RESEARCH_ONLY":
                continue
            if _string(row.get("edge_priority")) not in {"ROBUST_CORE", "CORE"}:
                continue
            if "price_evidence_status" in row and _string(row.get("price_evidence_status")) != "CONFIRMED":
                continue
            if "edge_reliability_status" in row and _string(row.get("edge_reliability_status")) != "SUPPORTED_SUBGROUP":
                continue
            if _string(row.get("market")) != "spread":
                continue
            gid, side, book = (_string(row.get(x)) for x in ("game_id", "side", "book"))
            if not gid or not side or not book:
                continue
            kickoff, quote = _ts(row.get("date")), _ts(row.get("quote_at"))
            if pd.isna(kickoff) or pd.isna(quote) or when >= kickoff or quote > when:
                continue
            age = (when - quote).total_seconds() / 60
            if not 0 <= age <= MAX_QUOTE_AGE_MINUTES:
                continue
            line, odds, bet_to = (_num(row.get(x)) for x in ("line", "odds", "bet_to_line"))
            cushion = _num(row.get("conservative_price_margin"))
            if (not all(math.isfinite(v) for v in (line, odds, bet_to, cushion))
                    or odds == 0 or cushion <= 0 or line < bet_to - 1e-9):
                continue
            decision_id = "|".join([gid, "spread", side])
            if decision_id in seen:
                continue
            item = dict.fromkeys(LEDGER_COLUMNS)
            item.update({
                "schema_version": SCHEMA_VERSION, "decision_id": decision_id,
                "decision_at": when.isoformat(), "game_id": gid,
                "kickoff": kickoff.isoformat(),
                "away_team": row.get("away_team"), "home_team": row.get("home_team"),
                "market": "spread", "side": side, "book": book,
                "quote_at": quote.isoformat(), "entry_line": line, "entry_odds": odds,
                "entry_implied_probability": _safe_number(_implied(odds)),
                "model_probability": _safe_number(row.get("probability")),
                "model_ev": _safe_number(row.get("ev")),
                "model_edge": _safe_number(row.get("edge")),
                "bet_to_line": bet_to, "conservative_price_margin": cushion,
                "edge_priority": _string(row.get("edge_priority")),
                "timing_action": _string(row.get("timing_action")),
                "timing_reason": _string(row.get("timing_reason")),
                "timing_evidence_source": _string(row.get("timing_evidence_source")),
                "timing_previous_line": _safe_number(row.get("timing_previous_line")),
                "timing_previous_odds": _safe_number(row.get("timing_previous_odds")),
                "timing_last_observed_at": (
                    _ts(row.get("timing_last_observed_at")).isoformat()
                    if pd.notna(_ts(row.get("timing_last_observed_at"))) else None
                ),
                "decision_mode": "RESEARCH_ONLY_NO_STAKING",
            })
            new.append(item)
            seen.add(decision_id)
    if new:
        combined = pd.concat([old, pd.DataFrame(new, columns=LEDGER_COLUMNS)],
                             ignore_index=True, sort=False)
        combined.to_csv(file, index=False)
    return {
        "status": "APPENDED" if new else "NO_NEW_DECISIONS",
        "schema_version": SCHEMA_VERSION,
        "appended": len(new), "total_decisions": len(old) + len(new),
        "path": str(file), "research_only": True, "approved_units": 0,
    }


def _read_forward_offers(market_history_path, candidate_history_path, entries, as_of):
    """Construct timestamp-correct same-book spread offers for *frozen* entries."""
    if entries.empty:
        return pd.DataFrame()
    candidate_raw = _read_csv(candidate_history_path)
    prior = _observations(candidate_raw, as_of)
    if not prior.empty:
        prior["_source_name"] = "candidate_quote"
    fake_candidates = pd.DataFrame([
        {"game_id": r["game_id"], "market": "spread",
         "side": r["side"], "book": r["book"]}
        for _, r in entries.iterrows()
    ])
    market = _captured_book_observations(market_history_path, fake_candidates, as_of)
    if not market.empty:
        market = _observations(market, as_of)
        market["_source_name"] = "captured_market_quote"
    combined = pd.concat([prior, market], ignore_index=True, sort=False)
    if combined.empty:
        return combined
    combined = combined[combined["market"].eq("spread")].copy()
    combined["_source_order"] = combined["_source_name"].map(
        {"captured_market_quote": 0, "candidate_quote": 1}
    ).fillna(2)
    combined = combined.sort_values(["_seen", "_source_order"])
    # One recorded opportunity per game/book/side/timestamp. Prefer the market
    # capture when the model happens to echo the same observed quote.
    combined = combined.drop_duplicates(
        ["game_id", "market", "side", "book", "_seen"], keep="first"
    )
    return combined


def _movement(entry_line, entry_odds, future_line, future_odds):
    """Price-inclusive comparison; conflicting line/price terms remain mixed."""
    line = _num(future_line) - _num(entry_line)
    probability_pp = 100 * (_implied(future_odds) - _implied(entry_odds))
    if not all(math.isfinite(v) for v in (line, probability_pp)):
        return "UNCOMPARABLE", None, None
    if line >= 0.5 - 1e-8:
        label = "MIXED_LINE_PRICE" if probability_pp > 0.05 else "BETTER"
    elif line <= -0.5 + 1e-8:
        label = "MIXED_LINE_PRICE" if probability_pp < -0.05 else "WORSE"
    elif abs(line) <= 1e-8:
        if probability_pp <= -1:
            label = "BETTER"
        elif probability_pp >= 1:
            label = "WORSE"
        else:
            label = "FLAT"
    else:
        label = "MIXED_LINE_PRICE"
    return label, round(float(line), 3), round(float(probability_pp), 3)


def _score(action, movement):
    if movement not in ("BETTER", "WORSE"):
        return None
    return bool(
        (action == "BET_NOW_RESEARCH" and movement == "WORSE")
        or (action == "WAIT_MONITOR" and movement == "BETTER")
    )


def _assign_observation(out, prefix, offer, action, entry):
    movement, line_delta, price_delta = _movement(
        entry["entry_line"], entry["entry_odds"], offer["line"], offer["odds"]
    )
    out.update({
        prefix + "_status": "OBSERVED",
        prefix + "_observed_at": offer["_seen"].isoformat(),
        prefix + "_source": offer.get("_source_name"),
        prefix + "_line": float(offer["line"]), prefix + "_odds": float(offer["odds"]),
        prefix + "_line_change": line_delta,
        prefix + "_implied_change_pp": price_delta,
        prefix + "_movement": movement,
        prefix + "_action_correct": _score(action, movement),
    })


def grade_timing_forward(
    ledger_path="history/timing_decisions_v1.csv",
    market_history_path="history/market_snapshots.csv",
    candidate_history_path="history/edge_candidate_snapshots.csv",
    reports_dir="reports",
    as_of=None,
):
    """Grade the frozen signal against predeclared 6-9h and final-hour offers.

    Primary comparison: FIRST observed, same-book quote in [t+6h, t+9h].
    Final-hour comparison: LAST observed quote in [kickoff-60m, kickoff).
    Neither is an actual betting fill or an official closing price.
    """
    when = _ts(as_of if as_of is not None else datetime.now(timezone.utc))
    if pd.isna(when):
        raise ValueError("as_of must be valid UTC")
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    raw = _read_csv(ledger_path)
    entries = raw.copy()
    if not raw.empty:
        if not set(LEDGER_COLUMNS).issubset(raw):
            raise ValueError("Timing decision ledger missing required columns")
        entries["_decision"] = pd.to_datetime(entries["decision_at"], utc=True, errors="coerce")
        entries["_kick"] = pd.to_datetime(entries["kickoff"], utc=True, errors="coerce")
        entries["_quote"] = pd.to_datetime(entries["quote_at"], utc=True, errors="coerce")
        entries["entry_line"] = pd.to_numeric(entries["entry_line"], errors="coerce")
        entries["entry_odds"] = pd.to_numeric(entries["entry_odds"], errors="coerce")
        entries["_age_min"] = (entries["_decision"] - entries["_quote"]).dt.total_seconds() / 60
        valid = (
            entries["decision_id"].notna() & entries["game_id"].notna()
            & entries["_decision"].notna() & entries["_kick"].notna()
            & entries["_quote"].notna()
            & (entries["_quote"] <= entries["_decision"])
            & entries["_age_min"].between(0, MAX_QUOTE_AGE_MINUTES)
            & (entries["_decision"] < entries["_kick"])
            & (entries["_decision"] <= when)
            & entries["entry_line"].notna()
            & entries["entry_odds"].notna() & entries["entry_odds"].ne(0)
            & entries["timing_action"].isin(TIMING_ACTIONS)
            & entries["decision_mode"].eq("RESEARCH_ONLY_NO_STAKING")
            & entries["market"].eq("spread")
            & entries["book"].notna()
            & pd.to_numeric(entries["schema_version"], errors="coerce").eq(SCHEMA_VERSION)
        )
        entries = entries[valid].copy()
        entries["game_id"] = entries["game_id"].astype(str)
        entries = entries.sort_values("_decision").drop_duplicates(
            subset=["decision_id"], keep="first"
        )
    offers = _read_forward_offers(market_history_path, candidate_history_path, entries, when)
    graded = []
    for _, entry in entries.iterrows():
        decision = entry["_decision"]
        kickoff = entry["_kick"]
        action = str(entry["timing_action"])
        record = {col: entry.get(col) for col in LEDGER_COLUMNS}
        record.update({
            "primary_status": "PENDING_HORIZON",
            "near_kickoff_status": "PENDING_KICKOFF",
            "primary_action_correct": None, "near_kickoff_action_correct": None,
        })
        for prefix in ("primary", "near_kickoff"):
            for suffix in ("observed_at", "source", "line", "odds", "line_change",
                           "implied_change_pp", "movement"):
                record[prefix + "_" + suffix] = None
        target = decision + pd.Timedelta(hours=PRIMARY_HORIZON_HOURS)
        deadline = target + pd.Timedelta(hours=PRIMARY_TOLERANCE_HOURS)
        if target >= kickoff:
            record["primary_status"] = "INSUFFICIENT_PREGAME_WINDOW"
        elif when >= deadline or when >= kickoff:
            record["primary_status"] = "NO_HORIZON_QUOTE"
        if offers.empty:
            matching = offers
        else:
            matching = offers[
                offers["game_id"].eq(str(entry["game_id"]))
                & offers["side"].eq(str(entry["side"]))
                & offers["book"].eq(str(entry["book"]))
                & offers["market"].eq("spread")
                & offers["_seen"].gt(decision)
                & offers["_seen"].lt(kickoff)
            ].sort_values(["_seen", "_source_order"])
        if target < kickoff and when >= target:
            horizon = matching[
                matching["_seen"].ge(target)
                & matching["_seen"].le(deadline)
            ]
            if not horizon.empty:
                _assign_observation(record, "primary", horizon.iloc[0], action, entry)
        if when >= kickoff:
            record["near_kickoff_status"] = "NO_FINAL_HOUR_QUOTE"
            final_hour = matching[
                matching["_seen"].ge(kickoff - pd.Timedelta(minutes=NEAR_KICKOFF_WINDOW_MINUTES))
            ]
            if not final_hour.empty:
                _assign_observation(record, "near_kickoff", final_hour.iloc[-1], action, entry)
        graded.append(record)
    result = pd.DataFrame(graded, columns=GRADED_COLUMNS)
    result.to_csv(reports / "edge_timing_forward_graded.csv", index=False)
    report = _timing_summary(result, len(raw))
    (reports / "edge_timing_forward_performance.json").write_text(json.dumps(report, indent=2))
    return report


def _wilson(success, total, z=1.96):
    if total < 1:
        return None
    p = success / total
    den = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / den
    span = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [round(mid - span, 5), round(mid + span, 5)]


def _one_summary(frame):
    n = len(frame)
    if not n:
        return {"decisions": 0, "primary_observed": 0, "conclusive": 0,
                "correct": 0, "accuracy": None, "accuracy_wilson_95": None,
                "near_kickoff_observed": 0}
    primary = frame[frame["primary_status"].eq("OBSERVED")]
    scored = primary[primary["primary_movement"].isin(["BETTER", "WORSE"])]
    right = sum(v is True or v == True for v in scored["primary_action_correct"])
    accuracy = right / len(scored) if len(scored) else None
    return {
        "decisions": n,
        "distinct_games": int(frame["game_id"].nunique()),
        "primary_observed": len(primary),
        "conclusive": len(scored),
        "correct": int(right),
        "incorrect": int(len(scored) - right),
        "accuracy": accuracy,
        "accuracy_wilson_95": _wilson(right, len(scored)),
        "primary_mixed": int(primary["primary_movement"].eq("MIXED_LINE_PRICE").sum()),
        "primary_flat": int(primary["primary_movement"].eq("FLAT").sum()),
        "primary_pending": int(frame["primary_status"].eq("PENDING_HORIZON").sum()),
        "primary_missing": int(frame["primary_status"].eq("NO_HORIZON_QUOTE").sum()),
        "primary_insufficient_window": int(frame["primary_status"].eq("INSUFFICIENT_PREGAME_WINDOW").sum()),
        "near_kickoff_observed": int(frame["near_kickoff_status"].eq("OBSERVED").sum()),
        "near_kickoff_missing": int(frame["near_kickoff_status"].eq("NO_FINAL_HOUR_QUOTE").sum()),
    }


def _timing_summary(graded, raw_count):
    overall = _one_summary(graded)
    actions = {
        action: _one_summary(graded[graded["timing_action"].eq(action)])
        for action in sorted(TIMING_ACTIONS)
    }
    conclusive = graded[
        graded["primary_status"].eq("OBSERVED")
        & graded["primary_movement"].isin(["BETTER", "WORSE"])
    ].copy() if not graded.empty else graded
    game_count = int(conclusive["game_id"].nunique()) if not conclusive.empty else 0
    weeks = 0
    if not conclusive.empty:
        kicks = pd.to_datetime(conclusive["kickoff"], utc=True)
        iso = kicks.dt.isocalendar()
        weeks = int((iso["year"].astype(str) + "-" + iso["week"].astype(str)).nunique())
    ready = (len(conclusive) >= MIN_REVIEW_OBSERVATIONS
             and game_count >= MIN_REVIEW_DISTINCT_GAMES
             and weeks >= MIN_REVIEW_DISTINCT_WEEKS
             and all(actions[a]["conclusive"] >= 20 for a in TIMING_ACTIONS))
    status = ("PENDING_FORWARD" if len(graded) == 0 else
              "REVIEW_READY_NOT_APPROVED" if ready else
              "EARLY_FORWARD" if len(conclusive) < 25 else "COLLECTING_FORWARD")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "research_only": True,
        "betting_authorized": False,
        "approved_units": 0,
        "no_official_close_claim": True,
        "no_executed_fill_claim": True,
        "evaluation_design": {
            "freeze": "first_clean_signal_per_game_market_side",
            "primary": "first_observed_same_book_quote_6_to_9_hours_after_decision",
            "near_kickoff": "last_observed_same_book_quote_in_final_60_minutes",
            "price_comparison": "selected_side_spread_and_break_even_prob; opposing_changes_are_mixed",
            "selection": "all_primary_observations; no_after_the_fact_best_quote",
            "staked": False,
            "independent_review_thresholds": {
                "conclusive": MIN_REVIEW_OBSERVATIONS,
                "distinct_games": MIN_REVIEW_DISTINCT_GAMES,
                "distinct_weeks": MIN_REVIEW_DISTINCT_WEEKS,
                "each_action": 20,
            },
        },
        "raw_decision_rows": int(raw_count),
        "clean_unique_decisions": len(graded),
        "invalid_or_duplicate_decisions": max(0, raw_count - len(graded)),
        "primary_distinct_games": game_count,
        "primary_distinct_weeks": weeks,
        "overall": overall,
        "by_action": actions,
        "notes": [
            "Observed sportsbook quotations are not guaranteed fills.",
            "Last-hour observation is not an official closing line.",
            "Wilson intervals assume independent rows; same-game clustering may reduce effective sample.",
            "Directional accuracy alone does not establish expected-value profitability.",
            "No automatic promotion or change to production staking.",
        ],
    }
