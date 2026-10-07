from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime, timezone
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


def _finite(value):
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def append_edge_shadow_candidates(board, path="history/edge_shadow_snapshots.csv", captured_at=None):
    """Persist the first verified pre-kickoff observation of each hypothesis regime.

    A shadow row is immutable evidence: once a game/market/regime band has an entry,
    later price movement does not replace it. This prevents hindsight line selection.
    """

    frame = board.copy() if isinstance(board, pd.DataFrame) else pd.DataFrame(board)
    if frame.empty:
        return 0

    captured = pd.Timestamp(captured_at or datetime.now(timezone.utc))
    if captured.tzinfo is None:
        captured = captured.tz_localize("UTC")
    else:
        captured = captured.tz_convert("UTC")

    rows = []
    for _, row in frame.iterrows():
        kickoff = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
        quote_at = pd.to_datetime(row.get("quote_at"), utc=True, errors="coerce")
        market = _text(row.get("market")).lower()
        regime_band = _text(row.get("regime_band"))
        status = _text(row.get("regime_status"))
        if (
            market not in {"moneyline", "spread", "total"}
            or not regime_band
            or status not in {"HISTORICAL_HYPOTHESIS", "PERSISTENT_CANDIDATE"}
            or pd.isna(kickoff)
            or pd.isna(quote_at)
            or not (quote_at <= captured < kickoff)
            or not _text(row.get("book"))
            or not _finite(row.get("line"))
            or not _finite(row.get("odds"))
            or not _finite(row.get("probability"))
            or not _finite(row.get("edge"))
            or not _finite(row.get("ev"))
            or float(row.get("ev")) <= 0
        ):
            continue
        rows.append(
            {
                "captured_at": captured.isoformat(),
                "game_id": str(row.get("game_id")),
                "kickoff": kickoff.isoformat(),
                "away_team": row.get("away_team"),
                "home_team": row.get("home_team"),
                "market": market,
                "side": row.get("side"),
                "line": float(row.get("line")),
                "odds": float(row.get("odds")),
                "probability": float(row.get("probability")),
                "edge": abs(float(row.get("edge"))),
                "ev": float(row.get("ev")),
                "badge": row.get("badge"),
                "book": row.get("book"),
                "quote_at": quote_at.isoformat(),
                "regime_status_at_entry": status,
                "regime_band": regime_band,
                "historical_bets": row.get("historical_bets"),
                "historical_win_rate": row.get("historical_win_rate"),
                "historical_roi": row.get("historical_roi"),
                "historical_avg_clv": row.get("historical_avg_clv"),
                "profitable_seasons": row.get("profitable_seasons"),
                "season_count": row.get("season_count"),
                "historical_verified_entry_bets": row.get("verified_entry_bets"),
                "historical_verified_entry_rate": row.get("verified_entry_rate"),
            }
        )

    if not rows:
        return 0

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    old = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()

    if old.empty:
        keep = new
    else:
        old_keys = set(
            zip(
                old.get("game_id", pd.Series(dtype=str)).astype(str),
                old.get("market", pd.Series(dtype=str)).astype(str),
                old.get("regime_band", pd.Series(dtype=str)).astype(str),
            )
        )
        mask = [
            (str(row.game_id), str(row.market), str(row.regime_band)) not in old_keys
            for _, row in new.iterrows()
        ]
        keep = new[pd.Series(mask, index=new.index)]

    if keep.empty:
        return 0

    combined = pd.concat([old, keep], ignore_index=True, sort=False)
    combined.to_csv(path, index=False)
    return int(len(keep))


def _forward_status(summary):
    n = int(summary.get("graded_bets", 0) or 0)
    roi = summary.get("roi")
    avg_clv = summary.get("avg_execution_clv")
    ci = summary.get("roi_ci_95") or [None, None]
    low = ci[0] if len(ci) > 0 else None
    high = ci[1] if len(ci) > 1 else None

    if (
        n >= 100
        and low is not None
        and float(low) > 0
        and avg_clv is not None
        and float(avg_clv) > 0
    ):
        return "FORWARD_VALIDATED"
    if (
        n >= 30
        and high is not None
        and float(high) < 0
        and avg_clv is not None
        and float(avg_clv) < 0
    ):
        return "UNDERPERFORMING"
    if (
        n >= 30
        and roi is not None
        and float(roi) > 0
        and (avg_clv is None or float(avg_clv) >= 0)
    ):
        return "PROMISING"
    if n >= 10:
        return "DEVELOPING"
    return "EARLY_SAMPLE"


def _empty_report():
    return {
        "schema_version": 1,
        "status": "NO_GRADED_SAMPLE",
        "graded_bets": 0,
        "regimes": [],
        "validated_regimes": 0,
        "methodology": _methodology(),
    }


def _methodology():
    return (
        "First verified pre-kickoff shadow entry per game/market/regime band. "
        "Entries require a real sportsbook, executable American price, quote timestamp "
        "at or before capture time, and capture time strictly before kickoff. Flat 1u "
        "risk is used for edge proof. FORWARD_VALIDATED requires >=100 graded bets, a "
        "positive 95% bootstrap ROI lower bound, and positive average execution CLV. "
        "Shadow evidence never changes the fair line or model probability."
    )


def grade_edge_shadow_history(client, history_dir="history", reports_dir="reports"):
    hdir = Path(history_dir)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    hp = hdir / "edge_shadow_snapshots.csv"
    mp = hdir / "market_snapshots.csv"

    if not hp.exists():
        report = _empty_report()
        (reports / "edge_forward_performance.json").write_text(json.dumps(report, indent=2))
        pd.DataFrame().to_csv(reports / "edge_forward_graded.csv", index=False)
        return report

    hist = pd.read_csv(hp, low_memory=False)
    if hist.empty:
        report = _empty_report()
        (reports / "edge_forward_performance.json").write_text(json.dumps(report, indent=2))
        pd.DataFrame().to_csv(reports / "edge_forward_graded.csv", index=False)
        return report

    hist["_ts"] = pd.to_datetime(hist.get("captured_at"), utc=True, errors="coerce")
    hist["_quote"] = pd.to_datetime(hist.get("quote_at"), utc=True, errors="coerce")
    hist["_kick"] = pd.to_datetime(hist.get("kickoff"), utc=True, errors="coerce")
    hist = hist[
        hist["_ts"].notna()
        & hist["_quote"].notna()
        & hist["_kick"].notna()
        & (hist["_quote"] <= hist["_ts"])
        & (hist["_ts"] < hist["_kick"])
    ].copy()

    markets = pd.read_csv(mp, low_memory=False) if mp.exists() else pd.DataFrame()

    finals = {}
    seasons = sorted(
        {
            int(x)
            for x in pd.to_numeric(
                hist.get("kickoff", pd.Series(dtype=object)).astype(str).str[:4],
                errors="coerce",
            )
            .dropna()
            .unique()
        }
    )
    for season in seasons:
        try:
            sf = client.season_frame(season)
        except Exception:
            continue
        done = sf[sf["completed"].astype(str).str.lower().isin({"true", "1", "t", "yes"})]
        for _, row in done.iterrows():
            try:
                finals[str(row.game_id)] = (float(row.home_points), float(row.away_points))
            except Exception:
                pass

    rows = []
    group_cols = ["game_id", "market", "regime_band"]
    for _, group in hist.sort_values("_ts").groupby(group_cols, sort=False):
        entry = group.iloc[0]
        gid = str(entry.game_id)
        if gid not in finals:
            continue
        home_points, away_points = finals[gid]
        actual_margin = home_points - away_points
        actual_total = home_points + away_points
        market = str(entry.market)
        normalized = pd.Series(
            {
                "home_team": entry.get("home_team"),
                "away_team": entry.get("away_team"),
                "quant_market": market,
                "quant_side": entry.get("side"),
                "quant_price": entry.get("line"),
                "quant_odds": entry.get("odds"),
            }
        )
        result = _grade_row(normalized, actual_margin, actual_total)
        profit = _profit(result, entry.get("odds"), market)
        kickoff = pd.to_datetime(entry.get("kickoff"), utc=True, errors="coerce")
        close = _latest_pre_kickoff_market(markets, gid, kickoff)
        clv, clv_source = _clv_from_market_snapshot(normalized, close)
        execution_clv = _execution_clv_from_market_snapshot(normalized, close)
        rows.append(
            {
                "game_id": gid,
                "kickoff": entry.get("kickoff"),
                "away_team": entry.get("away_team"),
                "home_team": entry.get("home_team"),
                "market": market,
                "side": entry.get("side"),
                "line": entry.get("line"),
                "odds": entry.get("odds"),
                "probability": entry.get("probability"),
                "edge": entry.get("edge"),
                "ev": entry.get("ev"),
                "book": entry.get("book"),
                "entry_snapshot": entry.get("captured_at"),
                "entry_quote_at": entry.get("quote_at"),
                "regime_band": entry.get("regime_band"),
                "regime_status_at_entry": entry.get("regime_status_at_entry"),
                "historical_bets": entry.get("historical_bets"),
                "historical_roi": entry.get("historical_roi"),
                "actual_margin_home": actual_margin,
                "actual_total": actual_total,
                "result": result,
                "profit": profit,
                "clv_proxy": clv,
                "execution_clv": execution_clv,
                "clv_source": clv_source,
                "close_snapshot_at": close.get("captured_at") if close is not None else None,
            }
        )

    graded = pd.DataFrame(rows)
    graded.to_csv(reports / "edge_forward_graded.csv", index=False)

    regimes = []
    if not graded.empty:
        for (market, band), frame in graded.groupby(["market", "regime_band"], sort=True):
            summary = _summary(frame)
            status = _forward_status(summary)
            regimes.append(
                {
                    "market": str(market),
                    "edge_band": str(band),
                    "status": status,
                    **summary,
                }
            )

    report = {
        "schema_version": 1,
        "status": "TRACKING" if len(graded) else "NO_GRADED_SAMPLE",
        "graded_bets": int(len(graded)),
        "regimes": regimes,
        "validated_regimes": int(sum(r["status"] == "FORWARD_VALIDATED" for r in regimes)),
        "methodology": _methodology(),
    }
    (reports / "edge_forward_performance.json").write_text(json.dumps(report, indent=2))
    return report


def load_edge_forward_report(reports_dir="reports"):
    path = Path(reports_dir) / "edge_forward_performance.json"
    if not path.exists():
        return _empty_report()
    try:
        return json.loads(path.read_text())
    except Exception:
        return _empty_report()


def apply_forward_validation(historical_report, forward_report):
    """Promote a historical hypothesis only after verified forward validation."""

    report = deepcopy(historical_report or {})
    forward_map = {
        (str(row.get("market") or "").lower(), str(row.get("edge_band") or "")): row
        for row in (forward_report or {}).get("regimes") or []
    }
    for regime in report.get("regimes") or []:
        key = (str(regime.get("market") or "").lower(), str(regime.get("edge_band") or ""))
        forward = forward_map.get(key)
        regime["forward_status"] = (forward or {}).get("status", "EARLY_SAMPLE")
        regime["forward_bets"] = int((forward or {}).get("graded_bets", 0) or 0)
        regime["forward_roi"] = (forward or {}).get("roi")
        regime["forward_roi_ci_95"] = (forward or {}).get("roi_ci_95", [None, None])
        regime["forward_avg_execution_clv"] = (forward or {}).get("avg_execution_clv")
        if (
            regime.get("status") == "HISTORICAL_HYPOTHESIS"
            and regime["forward_status"] == "FORWARD_VALIDATED"
        ):
            regime["status"] = "PERSISTENT_CANDIDATE"
            regime["promotion_basis"] = "verified_forward_validation"
        else:
            regime["promotion_basis"] = (
                "verified_historical_entries"
                if regime.get("status") == "PERSISTENT_CANDIDATE"
                else "not_promoted"
            )

    regimes = report.get("regimes") or []
    report["persistent_regimes"] = int(
        sum(row.get("status") == "PERSISTENT_CANDIDATE" for row in regimes)
    )
    report["historical_hypothesis_regimes"] = int(
        sum(row.get("status") == "HISTORICAL_HYPOTHESIS" for row in regimes)
    )
    report["forward_validated_regimes"] = int(
        sum(row.get("forward_status") == "FORWARD_VALIDATED" for row in regimes)
    )
    report["forward_validation"] = {
        "graded_bets": int((forward_report or {}).get("graded_bets", 0) or 0),
        "validated_regimes": int((forward_report or {}).get("validated_regimes", 0) or 0),
        "status": (forward_report or {}).get("status", "NO_GRADED_SAMPLE"),
    }
    return report
