"""Timestamp-safe forward ledger for historically supported CFB edge candidates.

Historical edge regimes are discovery evidence. This module creates an independent
2026+ forward record from the first clean pre-kickoff observation of each game/market.
It never changes projections, signals, selection, or stake sizing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .grading import (
    _clv_from_market_snapshot,
    _execution_clv_from_market_snapshot,
    _grade_row,
    _latest_pre_kickoff_market,
    _profit,
    _summary,
)

EDGE_FORWARD_SCHEMA_VERSION = 2
ELIGIBLE_RELIABILITY = {"SUPPORTED_SUBGROUP", "PERSISTENT_PARENT_ONLY"}


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def _present(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(str(value).strip())


def edge_forward_class(reliability_status, price_status):
    reliability = str(reliability_status or "").upper()
    price = str(price_status or "").upper()
    if reliability == "SUPPORTED_SUBGROUP":
        if price == "CONFIRMED":
            return "CORE_CONFIRMED"
        if price == "PLAUSIBLE":
            return "SUPPORTED_PLAUSIBLE"
        if price == "OVERPRICED":
            return "SUPPORTED_OVERPRICED"
        return "SUPPORTED_UNKNOWN_PRICE"
    if reliability == "PERSISTENT_PARENT_ONLY":
        if price == "CONFIRMED":
            return "PARENT_CONFIRMED"
        return "PARENT_UNCONFIRMED"
    return "OTHER"


def append_edge_candidate_snapshots(
    edges: pd.DataFrame,
    path="history/edge_candidate_snapshots.csv",
    snapshot_at=None,
) -> dict[str, object]:
    """Append this run's supported/persistent pregame edge observations.

    A row is an observation, not a bet authorization. Exact duplicate observations
    from a retried run are de-duplicated by snapshot/game/market/side/line/odds.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = pd.to_datetime(snapshot_at, utc=True, errors="coerce")
    if pd.isna(stamp):
        stamp = pd.Timestamp.now(tz="UTC")
    stamp_text = stamp.isoformat()

    if edges is None or edges.empty:
        return {
            "status": "NO_CANDIDATES",
            "appended": 0,
            "snapshot_at": stamp_text,
            "path": str(target),
        }

    frame = edges.copy()
    if "edge_reliability_status" in frame.columns:
        frame = frame[
            frame["edge_reliability_status"].fillna("").astype(str).isin(
                ELIGIBLE_RELIABILITY
            )
        ].copy()
    if frame.empty:
        return {
            "status": "NO_ELIGIBLE_CANDIDATES",
            "appended": 0,
            "snapshot_at": stamp_text,
            "path": str(target),
        }

    # Never publish a first-seen entry whose quote could not have existed
    # at its purported observation timestamp. This previously invalidated
    # every v2 forward row when the pipeline reused its run-start timestamp.
    if "quote_at" in frame:
        quote_times = pd.to_datetime(frame["quote_at"], utc=True, errors="coerce")
        future_quotes = quote_times.notna() & (quote_times > stamp)
        if future_quotes.any():
            raise ValueError(
                f"Forward edge snapshot precedes {int(future_quotes.sum())} "
                "captured quotes; use post-collection observation time"
            )
    if "date" in frame:
        kickoff_times = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        after_kickoff = kickoff_times.notna() & (kickoff_times <= stamp)
        if after_kickoff.any():
            raise ValueError(
                f"Forward edge snapshot contains {int(after_kickoff.sum())} "
                "started games; remove them before freezing"
            )

    frame["edge_forward_class"] = [
        edge_forward_class(reliability, price)
        for reliability, price in zip(
            frame.get("edge_reliability_status", pd.Series(index=frame.index, dtype=object)),
            frame.get("price_evidence_status", pd.Series(index=frame.index, dtype=object)),
        )
    ]
    frame.insert(0, "edge_snapshot_schema_version", EDGE_FORWARD_SCHEMA_VERSION)
    frame.insert(1, "snapshot_at", stamp_text)

    keep = [
        "edge_snapshot_schema_version",
        "snapshot_at",
        "game_id",
        "date",
        "away_team",
        "home_team",
        "market",
        "side",
        "line",
        "odds",
        "probability",
        "edge",
        "ev",
        "badge",
        "book",
        "quote_at",
        "regime_status",
        "edge_reliability_status",
        "edge_forward_class",
        "price_evidence_status",
        "price_evidence_sample",
        "historical_price_win_rate",
        "historical_price_wilson_lower",
        "current_break_even_probability",
        "historical_price_margin",
        "conservative_price_margin",
        "historical_fair_odds_lower_bound",
        "regime_band",
        "historical_bets",
        "historical_win_rate",
        "historical_roi",
        "historical_avg_clv",
        "profitable_seasons",
        "season_count",
        "subgroup_key",
        "subgroup_status",
        "subgroup_bets",
        "subgroup_win_rate",
        "subgroup_roi",
        "subgroup_avg_clv",
        "subgroup_profitable_seasons",
        "subgroup_season_count",
        "subgroup_validation_design",
        "subgroup_discovery_seasons",
        "subgroup_discovery_bets",
        "subgroup_discovery_roi",
        "subgroup_holdout_season",
        "subgroup_holdout_bets",
        "subgroup_holdout_roi",
        "subgroup_holdout_win_rate",
        "subgroup_holdout_confirmed",
        "currently_selected",
        "selected_quant_signal",
        "selected_quant_ev",
    ]
    for column in keep:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[keep]

    old = pd.read_csv(target, low_memory=False) if target.exists() else pd.DataFrame()
    combined = pd.concat([old, frame], ignore_index=True, sort=False)
    keys = ["snapshot_at", "game_id", "market", "side", "line", "odds"]
    available_keys = [column for column in keys if column in combined.columns]
    if available_keys:
        combined = combined.drop_duplicates(subset=available_keys, keep="last")
    appended = max(0, len(combined) - len(old))
    combined.to_csv(target, index=False)
    return {
        "status": "APPENDED" if appended else "UNCHANGED",
        "appended": int(appended),
        "total_rows": int(len(combined)),
        "snapshot_at": stamp_text,
        "path": str(target),
    }


def _clean_forward_entries(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history.copy()

    data = history.copy()
    data["_snapshot"] = pd.to_datetime(data.get("snapshot_at"), utc=True, errors="coerce")
    data["_kickoff"] = pd.to_datetime(data.get("date"), utc=True, errors="coerce")
    data["_quote"] = pd.to_datetime(data.get("quote_at"), utc=True, errors="coerce")
    data["_line"] = pd.to_numeric(data.get("line"), errors="coerce")
    data["_odds"] = pd.to_numeric(data.get("odds"), errors="coerce")
    data["_reliability"] = (
        data.get("edge_reliability_status", pd.Series(index=data.index, dtype=object))
        .fillna("")
        .astype(str)
    )
    data["_schema"] = pd.to_numeric(
        data.get("edge_snapshot_schema_version", pd.Series(index=data.index, dtype=float)),
        errors="coerce",
    )
    data["_edge_class"] = (
        data.get("edge_forward_class", pd.Series(index=data.index, dtype=object))
        .fillna("")
        .astype(str)
    )
    book_names = data.get(
        "book", pd.Series(index=data.index, dtype=object)
    ).fillna("").astype(str).str.strip()
    # A numeric source identifier, generic consensus, or opening-market
    # baseline is not a resolved sportsbook. Preserve raw observations but
    # never count them as clean forward betting evidence.
    unresolved = book_names.str.fullmatch(
        r"(?i)(?:actionnetwork\\s+book\\s+\\d+|book\\s*\\d+|"
        r"unknown|consensus|open|opening|unresolved|primary)",
        na=False,
    )
    data["_book_ok"] = book_names.map(_present) & ~unresolved

    clean = data[
        data["_snapshot"].notna()
        & data["_kickoff"].notna()
        & data["_quote"].notna()
        & (data["_snapshot"] < data["_kickoff"])
        & (data["_quote"] <= data["_snapshot"])
        & data["_line"].notna()
        & data["_odds"].notna()
        & data["_book_ok"]
        & data["_reliability"].isin(ELIGIBLE_RELIABILITY)
        & (data["_schema"] >= EDGE_FORWARD_SCHEMA_VERSION)
        & data["_edge_class"].map(_present)
    ].copy()
    if clean.empty:
        return clean

    clean = clean.sort_values(["_snapshot", "game_id", "market"])
    # The first clean edge observation is the forward entry for that game/market.
    clean = clean.drop_duplicates(subset=["game_id", "market"], keep="first")
    return clean


def _forward_status(summary: dict[str, object]) -> str:
    bets = int(summary.get("graded_bets", 0) or 0)
    roi = summary.get("roi")
    clv = summary.get("avg_execution_clv")
    ci = summary.get("roi_ci_95") or [None, None]
    low = ci[0] if len(ci) > 0 else None
    high = ci[1] if len(ci) > 1 else None

    if bets == 0:
        return "PENDING_FORWARD"
    if bets < 25:
        return "EARLY_SAMPLE"
    if bets < 50:
        return "DEVELOPING"
    if (
        bets >= 100
        and low is not None
        and float(low) > 0
        and clv is not None
        and float(clv) > 0
    ):
        return "FORWARD_VALIDATED"
    if (
        bets >= 50
        and high is not None
        and float(high) < 0
        and clv is not None
        and float(clv) < 0
    ):
        return "UNDERPERFORMING"
    if (
        roi is not None
        and float(roi) > 0
        and clv is not None
        and float(clv) >= 0
    ):
        return "PROMISING"
    return "UNVALIDATED"


def _group_summaries(frame: pd.DataFrame, column: str) -> dict[str, object]:
    if frame.empty or column not in frame.columns:
        return {}
    out = {}
    for key, rows in frame.groupby(column, dropna=False):
        label = str(key)
        summary = _summary(rows)
        summary["status"] = _forward_status(summary)
        out[label] = summary
    return out


def grade_edge_forward_history(
    client,
    history_path="history/edge_candidate_snapshots.csv",
    market_history_path="history/market_snapshots.csv",
    reports_dir="reports",
) -> dict[str, object]:
    """Grade first clean supported-edge observations after games become final."""

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    history_file = Path(history_path)
    market_file = Path(market_history_path)
    graded_path = reports / "edge_forward_graded.csv"
    report_path = reports / "edge_forward_performance.json"

    if not history_file.exists():
        empty = pd.DataFrame()
        empty.to_csv(graded_path, index=False)
        report = {
            "schema_version": EDGE_FORWARD_SCHEMA_VERSION,
            "status": "PENDING_FORWARD",
            "observed_rows": 0,
            "clean_entries": 0,
            "graded_bets": 0,
            "note": "No forward edge candidate history exists yet.",
        }
        report_path.write_text(json.dumps(report, indent=2))
        return report

    raw = pd.read_csv(history_file, low_memory=False)
    clean = _clean_forward_entries(raw)
    markets = pd.read_csv(market_file, low_memory=False) if market_file.exists() else pd.DataFrame()

    finals = {}
    seasons = sorted(
        {
            int(value)
            for value in pd.to_numeric(raw.get("season"), errors="coerce")
            .dropna()
            .unique()
        }
    ) if "season" in raw.columns else []
    # Current edge-board rows do not always carry season explicitly. Infer from kickoff.
    if not seasons and "date" in raw.columns:
        years = pd.to_datetime(raw["date"], utc=True, errors="coerce").dt.year.dropna()
        seasons = sorted({int(value) for value in years.unique()})

    for season in seasons:
        try:
            season_frame = client.season_frame(season)
        except Exception:
            continue
        completed = season_frame[
            season_frame["completed"].astype(str).str.lower().isin(
                {"true", "1", "t", "yes"}
            )
        ]
        for _, row in completed.iterrows():
            try:
                finals[str(row.game_id)] = (
                    float(row.home_points),
                    float(row.away_points),
                )
            except Exception:
                continue

    rows = []
    for _, entry in clean.iterrows():
        game_id = str(entry.get("game_id"))
        if game_id not in finals:
            continue
        home_points, away_points = finals[game_id]
        actual_margin = home_points - away_points
        actual_total = home_points + away_points

        market = str(entry.get("market") or "")
        side = str(entry.get("side") or "")
        pseudo = pd.Series(
            {
                **entry.to_dict(),
                "quant_market": market,
                "quant_side": side,
                "quant_price": entry.get("line"),
                "quant_odds": entry.get("odds"),
                "quant_book": entry.get("book"),
                "home_team": entry.get("home_team"),
                "away_team": entry.get("away_team"),
            }
        )
        result = _grade_row(pseudo, actual_margin, actual_total)
        profit = _profit(result, entry.get("odds"), market)
        kickoff = pd.to_datetime(entry.get("date"), utc=True, errors="coerce")
        close = _latest_pre_kickoff_market(markets, game_id, kickoff)
        clv, clv_source = _clv_from_market_snapshot(pseudo, close)
        execution_clv = _execution_clv_from_market_snapshot(pseudo, close)

        rows.append(
            {
                "game_id": game_id,
                "date": entry.get("date"),
                "away_team": entry.get("away_team"),
                "home_team": entry.get("home_team"),
                "market": market,
                "side": side,
                "line": entry.get("line"),
                "odds": entry.get("odds"),
                "book": entry.get("book"),
                "probability": entry.get("probability"),
                "edge": entry.get("edge"),
                "ev": entry.get("ev"),
                "badge": entry.get("badge"),
                "edge_reliability_status": entry.get("edge_reliability_status"),
                "edge_forward_class": entry.get("edge_forward_class"),
                "price_evidence_status": entry.get("price_evidence_status"),
                "price_evidence_sample": entry.get("price_evidence_sample"),
                "historical_price_win_rate": entry.get("historical_price_win_rate"),
                "historical_price_wilson_lower": entry.get("historical_price_wilson_lower"),
                "current_break_even_probability": entry.get("current_break_even_probability"),
                "historical_price_margin": entry.get("historical_price_margin"),
                "conservative_price_margin": entry.get("conservative_price_margin"),
                "historical_fair_odds_lower_bound": entry.get("historical_fair_odds_lower_bound"),
                "regime_band": entry.get("regime_band"),
                "subgroup_key": entry.get("subgroup_key"),
                "subgroup_status": entry.get("subgroup_status"),
                "historical_bets": entry.get("historical_bets"),
                "historical_roi": entry.get("historical_roi"),
                "subgroup_bets": entry.get("subgroup_bets"),
                "subgroup_roi": entry.get("subgroup_roi"),
                "subgroup_discovery_roi": entry.get("subgroup_discovery_roi"),
                "subgroup_holdout_season": entry.get("subgroup_holdout_season"),
                "subgroup_holdout_bets": entry.get("subgroup_holdout_bets"),
                "subgroup_holdout_roi": entry.get("subgroup_holdout_roi"),
                "entry_snapshot": entry.get("snapshot_at"),
                "entry_quote_at": entry.get("quote_at"),
                "kickoff": entry.get("date"),
                "result": result,
                "profit": profit,
                "clv_proxy": clv,
                "execution_clv": execution_clv,
                "clv_source": clv_source,
                "actual_margin_home": actual_margin,
                "actual_total": actual_total,
            }
        )

    graded = pd.DataFrame(rows)
    graded.to_csv(graded_path, index=False)
    overall = _summary(graded)
    status = _forward_status(overall)
    report = {
        "schema_version": EDGE_FORWARD_SCHEMA_VERSION,
        "status": status,
        "observed_rows": int(len(raw)),
        "clean_entries": int(len(clean)),
        **overall,
        "by_edge_class": _group_summaries(graded, "edge_forward_class"),
        "by_price_evidence": _group_summaries(graded, "price_evidence_status"),
        "by_reliability": _group_summaries(graded, "edge_reliability_status"),
        "by_subgroup": _group_summaries(graded, "subgroup_key"),
        "by_market": _group_summaries(graded, "market"),
        "methodology": (
            "First provenance-complete pre-kickoff observation per game/market; "
            "schema v2 preserves CORE/PLAUSIBLE/parent-only price class at entry; "
            "flat 1u risk at the observed executable price; results only after final; "
            "CLV uses the latest timestamp-valid pre-kickoff market snapshot. "
            "Historical regime statistics never alter forward P/L. "
            "FORWARD_VALIDATED requires >=100 graded bets, positive 95% ROI lower "
            "bound, and positive average execution CLV."
        ),
        "note": (
            "Forward edge validation is independent evidence. It does not authorize "
            "production staking by itself."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2))
    return report
