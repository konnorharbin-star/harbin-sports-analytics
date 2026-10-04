from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math

import pandas as pd


SIGNATURE_FIELDS = (
    "quant_market",
    "quant_side",
    "quant_book",
    "quant_price",
    "quant_odds",
    "quant_quote_at",
    "portfolio_candidate_units",
    "portfolio_stake_units",
    "performance_multiplier",
    "performance_feedback_reason",
    "execution_ready",
    "portfolio_action",
)

LEDGER_FIELDS = (
    "game_id",
    "season",
    "week",
    "date",
    "away_team",
    "home_team",
    "model_margin_home",
    "model_total",
    "calibrated_home_probability",
    "quant_signal",
    "quant_market",
    "quant_side",
    "quant_book",
    "quant_price",
    "quant_odds",
    "quant_quote_at",
    "quant_probability",
    "quant_ev",
    "quant_edge",
    "market_book_count",
    "market_consensus_quality",
    "data_quality_score",
    "context_risk",
    "risk_multiplier",
    "market_disagreement",
    "performance_multiplier",
    "performance_feedback_reason",
    "performance_adjusted_units",
    "portfolio_candidate_units",
    "portfolio_stake_units",
    "execution_ready",
    "portfolio_action",
    "portfolio_limit_reason",
)


def _finite(v):
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _signature(row) -> str:
    values = {}
    for field in SIGNATURE_FIELDS:
        value = row.get(field)
        if _finite(value):
            value = round(float(value), 8)
        elif pd.isna(value) if not isinstance(value, (dict, list)) else False:
            value = None
        values[field] = value
    return json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))


def append_portfolio_decisions(
    pred: pd.DataFrame,
    path="history/portfolio_decisions_v1.csv",
    decision_at: str | None = None,
) -> dict:
    """Persist cap-constrained portfolio decisions for independent forward grading.

    The ledger records PAPER/SHADOW/PRODUCTION portfolio decisions after Stage 5 has
    applied execution and concentration controls.  Repeated model runs append only when
    a game's executable portfolio state changes.  Grading can therefore use the first
    timestamp-valid decision as the simulated entry without treating raw model signals
    as wagers.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(pred, pd.DataFrame) or pred.empty:
        return {"path": str(p), "eligible_rows": 0, "appended_rows": 0}

    units = pd.to_numeric(pred.get("portfolio_candidate_units", 0), errors="coerce").fillna(0.0)
    action = pred.get("portfolio_action", pd.Series("PASS", index=pred.index)).fillna("PASS").astype(str).str.upper()
    eligible = pred[(units > 0) & action.isin({"PAPER", "SHADOW", "BET"})].copy()
    if eligible.empty:
        return {"path": str(p), "eligible_rows": 0, "appended_rows": 0}

    stamp = decision_at or datetime.now(timezone.utc).isoformat()
    cols = [c for c in LEDGER_FIELDS if c in eligible.columns]
    rows = eligible[cols].copy()
    rows.insert(0, "decision_at", stamp)
    rows["decision_signature"] = rows.apply(_signature, axis=1)

    if p.exists() and p.stat().st_size > 0:
        try:
            old = pd.read_csv(p, low_memory=False)
        except Exception:
            old = pd.DataFrame()
    else:
        old = pd.DataFrame()

    if not old.empty and {"game_id", "decision_signature"}.issubset(old.columns):
        old_ts = pd.to_datetime(old.get("decision_at"), utc=True, errors="coerce")
        old = old.assign(_decision_ts=old_ts).sort_values(["_decision_ts"], kind="mergesort")
        latest = old.groupby(old["game_id"].astype(str), sort=False).tail(1)
        last_sig = dict(zip(latest["game_id"].astype(str), latest["decision_signature"].astype(str)))
        keep = rows.apply(
            lambda r: str(r.get("decision_signature")) != last_sig.get(str(r.get("game_id"))),
            axis=1,
        )
        rows = rows[keep].copy()
        old = old.drop(columns=["_decision_ts"], errors="ignore")

    if rows.empty:
        return {"path": str(p), "eligible_rows": int(len(eligible)), "appended_rows": 0}

    combined = pd.concat([old, rows], ignore_index=True, sort=False) if not old.empty else rows
    combined.to_csv(p, index=False)
    return {
        "path": str(p),
        "eligible_rows": int(len(eligible)),
        "appended_rows": int(len(rows)),
        "total_rows": int(len(combined)),
    }
