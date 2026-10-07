"""Research-only CFB timing diagnostics from timestamped, same-book edge observations.

Never call a first/last observed quote an opening/official closing line.
Never extrapolate a single quote into a trend or change stakes/model probabilities.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

TIMING_SCHEMA = 1
MAX_QUOTE_AGE_MINUTES = 120
MAX_SNAPSHOT_QUOTE_GAP_MINUTES = 120
CLOCK_TOLERANCE_MINUTES = 5
MIN_MOVEMENT_MINUTES = 10


def _ts(value):
    return pd.to_datetime(value, utc=True, errors="coerce")


def _num(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _implied(american):
    odds = _num(american)
    if not math.isfinite(odds) or odds == 0:
        return np.nan
    return 100 / (odds + 100) if odds > 0 else -odds / (100 - odds)


def _observations(history, as_of):
    """Only valid, contemporaneously captured, pre-kickoff posted offers."""
    cols = ["game_id", "market", "side", "book", "line", "odds",
            "snapshot_at", "quote_at", "date"]
    if history is None or history.empty or not set(cols).issubset(history):
        return pd.DataFrame(columns=cols + ["_seen"])
    data = history.copy()
    for col in ("snapshot_at", "quote_at", "date"):
        data["_" + col] = pd.to_datetime(data[col], utc=True, errors="coerce")
    for col in ("line", "odds"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    # A quote may be fetched seconds after model execution begins. The true
    # observation time is the later of capture start and source quote time.
    data["_seen"] = data[["_snapshot_at", "_quote_at"]].max(axis=1)
    gap = (data["_snapshot_at"] - data["_quote_at"]).dt.total_seconds() / 60
    good = (
        data["_seen"].notna() & data["_date"].notna()
        & (data["_seen"] < data["_date"])
        & (data["_seen"] <= as_of + pd.Timedelta(minutes=CLOCK_TOLERANCE_MINUTES))
        & (gap >= -CLOCK_TOLERANCE_MINUTES)
        & (gap <= MAX_SNAPSHOT_QUOTE_GAP_MINUTES)
        & data["line"].notna() & data["odds"].notna()
        & (data["odds"] != 0)
        & data["book"].fillna("").astype(str).str.strip().ne("")
    )
    data = data[good].copy()
    data["game_id"] = data["game_id"].astype(str)
    for col in ("market", "side", "book"):
        data[col] = data[col].astype(str)
    return data.sort_values("_seen").drop_duplicates(
        ["game_id", "market", "side", "book", "_quote_at", "line", "odds"], keep="last"
    )


def _status(row, history, as_of):
    priority = str(row.get("edge_priority") or "")
    cutoff = _ts(row.get("date"))
    quote = _ts(row.get("quote_at"))
    line = _num(row.get("line"))
    price = _num(row.get("odds"))
    bet_to = _num(row.get("bet_to_line"))
    result = {
        "timing_action": "NO_TIMING_SIGNAL",
        "timing_reason": "NO_COMPARABLE_HISTORY",
        "timing_observations": 0,
        "timing_previous_line": np.nan,
        "timing_previous_odds": np.nan,
        "timing_line_move_pts": np.nan,
        "timing_price_move_pp": np.nan,
        "timing_last_observed_at": None,
        "timing_quote_age_minutes": np.nan,
        "timing_status": "RESEARCH_ONLY",
    }
    if priority not in {"ROBUST_CORE", "CORE"}:
        result.update(timing_action="PASS", timing_reason="NOT_PRIORITY_CORE")
        return result
    if pd.isna(cutoff) or cutoff <= as_of:
        result.update(timing_action="PASS", timing_reason="GAME_STARTED_OR_BAD_KICKOFF")
        return result
    if pd.isna(quote) or not math.isfinite(line) or not math.isfinite(price) or price == 0:
        result.update(timing_action="PASS", timing_reason="MISSING_EXECUTABLE_QUOTE")
        return result
    age = (as_of - quote).total_seconds() / 60
    result["timing_quote_age_minutes"] = round(age, 2)
    if age > MAX_QUOTE_AGE_MINUTES or age < -CLOCK_TOLERANCE_MINUTES:
        result.update(timing_action="PASS", timing_reason="STALE_OR_FUTURE_QUOTE")
        return result
    if str(row.get("market")) == "spread" and (not math.isfinite(bet_to) or line < bet_to - 1e-9):
        result.update(timing_action="PASS", timing_reason="OUTSIDE_BET_TO_LINE")
        return result
    if not math.isfinite(_num(row.get("conservative_price_margin"))) or _num(row.get("conservative_price_margin")) <= 0:
        result.update(timing_action="PASS", timing_reason="NO_CONSERVATIVE_PRICE_CUSHION")
        return result
    same = history[
        (history["game_id"] == str(row.get("game_id")))
        & (history["market"] == str(row.get("market")))
        & (history["side"] == str(row.get("side")))
        & (history["book"] == str(row.get("book")))
        & (history["_seen"] < as_of)
    ].sort_values("_seen")
    # Historic quotes must precede the current quote by at least ten minutes;
    # different prices at a different book are never called line movement.
    same = same[same["_seen"] <= quote - pd.Timedelta(minutes=MIN_MOVEMENT_MINUTES)]
    result["timing_observations"] = int(len(same))
    if same.empty:
        return result
    prev = same.iloc[-1]
    result["timing_previous_line"] = float(prev["line"])
    result["timing_previous_odds"] = float(prev["odds"])
    result["timing_last_observed_at"] = prev["_seen"].isoformat()
    line_move = line - float(prev["line"])  # selected-side spread: higher is better
    price_move = (_implied(price) - _implied(prev["odds"])) * 100
    result["timing_line_move_pts"] = round(line_move, 3)
    result["timing_price_move_pp"] = round(price_move, 3) if math.isfinite(price_move) else np.nan
    worse = line_move <= -0.5 or (abs(line_move) < 0.5 and price_move >= 1)
    better = line_move >= 0.5 or (abs(line_move) < 0.5 and price_move <= -1)
    if worse:
        result.update(timing_action="BET_NOW_RESEARCH", timing_reason="OBSERVED_SAME_BOOK_DETERIORATION")
    elif better:
        result.update(timing_action="WAIT_MONITOR", timing_reason="OBSERVED_SAME_BOOK_IMPROVEMENT")
    else:
        result.update(timing_action="NO_TIMING_SIGNAL", timing_reason="FLAT_SAME_BOOK_MARKET")
    return result


def enrich_edge_timing(frame, history_path="history/edge_candidate_snapshots.csv", as_of=None):
    """Annotate existing ranked edges. No alteration to priority, EV, or stake."""
    if frame is None:
        return pd.DataFrame()
    out = frame.copy()
    if out.empty:
        for col in ("timing_action", "timing_reason", "timing_status"):
            out[col] = pd.Series(dtype=str)
        return out
    as_of = _ts(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if pd.isna(as_of):
        raise ValueError("as_of must be a valid UTC timestamp")
    path = Path(history_path)
    try:
        raw = pd.read_csv(path, low_memory=False) if path.exists() and path.stat().st_size else pd.DataFrame()
    except (ValueError, pd.errors.ParserError):
        raw = pd.DataFrame()
    history = _observations(raw, as_of)
    details = pd.DataFrame([_status(row, history, as_of) for _, row in out.iterrows()], index=out.index)
    for col in details.columns:
        out[col] = details[col]
    return out


def summarize_observed_survival(history_path="history/edge_candidate_snapshots.csv", as_of=None):
    """Descriptive first-to-last *observed* price/line survival for finished games.

    No timing optimizer, no historical strategy performance claim, no fabricated close.
    A game's last observed quote is NOT necessarily the closing market.
    """
    as_of = _ts(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    path = Path(history_path)
    raw = pd.read_csv(path, low_memory=False) if path.exists() and path.stat().st_size else pd.DataFrame()
    clean = _observations(raw, as_of)
    if clean.empty:
        return {"schema_version": TIMING_SCHEMA, "status": "PENDING_OBSERVATIONS",
                "comparable_series": 0, "survived_bet_to": 0,
                "survival_rate": None, "closing_line_claim": False}
    finished = clean[clean["_date"] <= as_of]
    series = 0
    survived = 0
    for _, group in finished.groupby(["game_id", "market", "side", "book"]):
        if len(group) < 2:
            continue
        group = group.sort_values("_seen")
        first, last = group.iloc[0], group.iloc[-1]
        if (last["_seen"] - first["_seen"]).total_seconds() < 600:
            continue
        # The historical candidate's supported lower edge bound must be known.
        if str(first["market"]) != "spread":
            continue
        band = str(first.get("regime_band") or "")
        try:
            minimum = float(band.split("-", 1)[0])
            starting_edge = float(first["edge"])
            bet_to = math.ceil((float(first["line"]) - max(0, starting_edge - minimum)) * 2 - 1e-12) / 2
        except (ValueError, TypeError):
            continue
        series += 1
        survived += int(float(last["line"]) >= bet_to)
    return {
        "schema_version": TIMING_SCHEMA,
        "status": "DESCRIPTIVE_ONLY" if series else "PENDING_OBSERVATIONS",
        "comparable_series": series,
        "survived_bet_to": survived,
        "survival_rate": survived / series if series else None,
        "closing_line_claim": False,
        "line_stage": "last_captured_pre_kickoff_not_official_close",
        "betting_authorized": False,
    }


def write_timing_report(frame, history_path, destination, as_of):
    report = summarize_observed_survival(history_path, as_of)
    report["current_rows"] = int(len(frame))
    report["current_actions"] = (
        {str(k): int(v) for k, v in frame["timing_action"].value_counts().items()}
        if "timing_action" in frame else {}
    )
    Path(destination).write_text(json.dumps(report, indent=2))
    return report
