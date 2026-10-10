from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import json
import hashlib
import math
import csv
import os
import fcntl

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
    "paper_stake_units",
    "bankroll_adjusted_units",
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
    decisions,
    path="history/portfolio_decisions_v1.csv",
    decision_at: str | None = None,
) -> dict:
    """Append research decisions without rewriting any historical bytes.

    This is the legacy research ledger, not a receipt proving a user recommendation.
    Lock concurrent writers and deduplicate all prior signatures per game/market.
    Fail closed on unreadable or incompatible history rather than replacing it.
    """
    source = Path(path)
    source.parent.mkdir(parents=True, exist_ok=True)
    records = decisions.to_dict("records")
    rows = [r for r in records if float(r.get("portfolio_candidate_units") or 0) > 0
            and str(r.get("portfolio_action") or "PASS").upper() in {"PAPER", "SHADOW", "BET"}]
    if not rows:
        return {"path": str(source), "eligible_rows": 0, "appended_rows": 0}
    stamp = decision_at or datetime.now(UTC).isoformat()
    expected = ["decision_at", *LEDGER_FIELDS, "decision_signature"]
    with source.open("a+", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or expected
        if reader.fieldnames and not {"decision_at", "game_id", "quant_market", "decision_signature"}.issubset(fields):
            raise ValueError("Incompatible decision ledger schema; history preserved")
        old = list(reader)
        seen = {(str(r.get("game_id") or ""), str(r.get("quant_market") or ""),
                 str(r.get("decision_signature") or "")) for r in old}
        additions = []
        for row in rows:
            signature = _signature(row)
            key = (str(row.get("game_id") or ""), str(row.get("quant_market") or ""), signature)
            if not key[0] or not key[1]:
                raise ValueError("Decision requires game and market identity")
            if key in seen:
                continue
            additions.append({"decision_at": stamp, **{k: row.get(k) for k in LEDGER_FIELDS},
                              "decision_signature": signature})
            seen.add(key)
        # Legacy headers stay byte-for-byte intact. Freeze the complete modern
        # row in a sidecar so new context fields are never silently discarded.
        if set(LEDGER_FIELDS) - set(fields):
            events = source.with_suffix(".events")
            events.mkdir(exist_ok=True)
            for record in additions:
                identity = {k: record.get(k) for k in
                            ("game_id", "quant_market", "decision_signature")}
                event_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                event_path = events / (event_id + ".json")
                if not event_path.exists():
                    with event_path.open("x") as event:
                        json.dump(record, event, sort_keys=True, default=str)
                        event.flush()
                        os.fsync(event.fileno())
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerows(additions)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(source), "eligible_rows": len(rows),
            "appended_rows": len(additions), "total_rows": len(old) + len(additions)}
