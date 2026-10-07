from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .market import roi


EDGE_BINS = (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 999.0)
PERSISTENT_MIN_BETS = 180
PERSISTENT_MIN_SEASON_BETS = 50
WATCH_MIN_BETS = 100


def _finite(v):
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def edge_band(edge):
    try:
        x = abs(float(edge))
    except Exception:
        return None
    for lo, hi in zip(EDGE_BINS[:-1], EDGE_BINS[1:]):
        if lo <= x < hi:
            return f"{lo:g}-{hi:g}"
    return None


def _segment_stats(frame):
    if frame.empty:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": None,
            "units": 0.0,
            "roi": None,
            "avg_clv": None,
            "positive_clv_rate": None,
            "verified_entry_bets": 0,
            "verified_entry_rate": 0.0,
            "entry_quote_sources": {},
        }
    result = pd.to_numeric(frame.get("result"), errors="coerce")
    profit = pd.to_numeric(frame.get("profit"), errors="coerce")
    clv = pd.to_numeric(frame.get("clv"), errors="coerce").dropna()
    wins = int((result > 0).sum())
    losses = int((result < 0).sum())
    pushes = int((result == 0).sum())
    units = float(profit.fillna(0).sum())
    verified = (
        frame.get("entry_quote_verified", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "t", "yes"})
    )
    source_series = frame.get(
        "entry_quote_source",
        pd.Series("unknown", index=frame.index),
    ).fillna("unknown").astype(str)
    sources = {
        str(key): int(value)
        for key, value in source_series.value_counts(dropna=False).to_dict().items()
    }
    return {
        "bets": int(len(frame)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": float(wins / max(1, wins + losses)),
        "units": units,
        "roi": float(units / max(1, len(frame))),
        "avg_clv": float(clv.mean()) if len(clv) else None,
        "positive_clv_rate": float((clv > 0).mean()) if len(clv) else None,
        "verified_entry_bets": int(verified.sum()),
        "verified_entry_rate": float(verified.mean()) if len(frame) else 0.0,
        "entry_quote_sources": sources,
    }


def _status(overall, seasons):
    season_rows = list(seasons.values())
    season_count = len(season_rows)
    profitable = sum(
        row.get("roi") is not None and float(row["roi"]) > 0
        for row in season_rows
    )
    min_season_bets = min(
        (int(row.get("bets", 0)) for row in season_rows),
        default=0,
    )
    roi_value = overall.get("roi")
    avg_clv = overall.get("avg_clv")
    verified_rate = float(overall.get("verified_entry_rate", 0.0) or 0.0)
    verified_bets = int(overall.get("verified_entry_bets", 0) or 0)

    persistent_shape = (
        int(overall.get("bets", 0)) >= PERSISTENT_MIN_BETS
        and season_count >= 3
        and min_season_bets >= PERSISTENT_MIN_SEASON_BETS
        and profitable == season_count
        and roi_value is not None
        and float(roi_value) >= 0.03
        and avg_clv is not None
        and float(avg_clv) > 0
    )
    if persistent_shape and verified_rate >= 0.90 and verified_bets >= 150:
        return "PERSISTENT_CANDIDATE"
    if persistent_shape:
        return "HISTORICAL_HYPOTHESIS"

    watch_shape = (
        int(overall.get("bets", 0)) >= WATCH_MIN_BETS
        and season_count >= 2
        and profitable >= max(2, math.ceil(season_count * 2 / 3))
        and roi_value is not None
        and float(roi_value) > 0
        and (avg_clv is None or float(avg_clv) >= 0)
    )
    if watch_shape and verified_rate >= 0.90:
        return "WATCH"
    if watch_shape:
        return "HISTORICAL_WATCH"
    return "UNSUPPORTED"


def build_edge_regime_report(bets):
    """Find historically persistent model-market disagreement regimes.

    The edge bands are fixed before reading outcomes. Archive-only historical results
    can generate hypotheses but cannot certify a live edge. PERSISTENT_CANDIDATE also
    requires high coverage from timestamp-verified entry quotes. This report never
    changes the fair projection.
    """

    data = bets.copy() if isinstance(bets, pd.DataFrame) else pd.DataFrame(bets)
    if data.empty:
        return {
            "schema_version": 1,
            "status": "NO_SAMPLE",
            "regimes": [],
            "persistent_regimes": 0,
            "historical_hypothesis_regimes": 0,
            "watch_regimes": 0,
            "historical_watch_regimes": 0,
            "methodology": _methodology(),
        }

    data["market"] = data.get("market", "").astype(str).str.lower()
    data["edge"] = pd.to_numeric(data.get("edge"), errors="coerce")
    data = data[data["edge"].notna()].copy()
    data["edge_band"] = data["edge"].map(edge_band)

    regimes = []
    for market in ("spread", "total", "moneyline"):
        market_rows = data[data["market"] == market]
        for lo, hi in zip(EDGE_BINS[:-1], EDGE_BINS[1:]):
            band = f"{lo:g}-{hi:g}"
            segment = market_rows[
                (market_rows["edge"].abs() >= lo)
                & (market_rows["edge"].abs() < hi)
            ]
            if segment.empty:
                continue
            overall = _segment_stats(segment)
            by_season = {}
            if "season" in segment.columns:
                for season, season_rows in segment.groupby("season"):
                    by_season[str(int(float(season)))] = _segment_stats(season_rows)
            status = _status(overall, by_season)
            profitable_seasons = sum(
                row.get("roi") is not None and float(row["roi"]) > 0
                for row in by_season.values()
            )
            regimes.append(
                {
                    "market": market,
                    "edge_band": band,
                    "min_edge": lo,
                    "max_edge": hi,
                    "status": status,
                    **overall,
                    "season_count": int(len(by_season)),
                    "profitable_seasons": int(profitable_seasons),
                    "min_season_bets": min(
                        (int(row.get("bets", 0)) for row in by_season.values()),
                        default=0,
                    ),
                    "by_season": by_season,
                }
            )

    order = {
        "PERSISTENT_CANDIDATE": 0,
        "HISTORICAL_HYPOTHESIS": 1,
        "WATCH": 2,
        "HISTORICAL_WATCH": 3,
        "UNSUPPORTED": 4,
    }
    regimes.sort(
        key=lambda row: (
            order.get(row["status"], 9),
            row["market"],
            float(row["min_edge"]),
        )
    )
    return {
        "schema_version": 1,
        "status": "TRACKING",
        "regimes": regimes,
        "persistent_regimes": int(
            sum(row["status"] == "PERSISTENT_CANDIDATE" for row in regimes)
        ),
        "historical_hypothesis_regimes": int(
            sum(row["status"] == "HISTORICAL_HYPOTHESIS" for row in regimes)
        ),
        "watch_regimes": int(sum(row["status"] == "WATCH" for row in regimes)),
        "historical_watch_regimes": int(
            sum(row["status"] == "HISTORICAL_WATCH" for row in regimes)
        ),
        "methodology": _methodology(),
    }


def _methodology():
    return (
        "Fixed edge bands; market-specific walk-forward outcomes. Archive/opening-line "
        "results can identify HISTORICAL_HYPOTHESIS regimes but cannot steer live "
        "selection. PERSISTENT_CANDIDATE additionally requires >=90% timestamp-verified "
        "entry quotes and >=150 verified entries, plus >=180 total bets, >=3 seasons, "
        ">=50 bets in every season, positive ROI in every season, >=3% aggregate ROI, "
        "and positive average CLV. WATCH also requires >=90% verified entry coverage. "
        "Historical evidence never changes the fair line or model probability."
    )


def write_edge_regime_report(bets, json_path, csv_path=None):
    report = build_edge_regime_report(bets)
    Path(json_path).write_text(json.dumps(report, indent=2))
    if csv_path is not None:
        flat = []
        for row in report["regimes"]:
            flat.append({k: v for k, v in row.items() if k != "by_season"})
        pd.DataFrame(flat).to_csv(csv_path, index=False)
    return report


def load_or_build_edge_regime_report(reports_dir="reports"):
    reports = Path(reports_dir)
    path = reports / "edge_regimes.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    bets_path = reports / "backtest_bets.csv"
    if bets_path.exists():
        try:
            return build_edge_regime_report(pd.read_csv(bets_path, low_memory=False))
        except Exception:
            pass
    return build_edge_regime_report(pd.DataFrame())


def match_edge_regime(report, market, edge):
    if not _finite(edge):
        return None
    market = str(market or "").lower()
    x = abs(float(edge))
    for row in (report or {}).get("regimes") or []:
        if str(row.get("market") or "").lower() != market:
            continue
        lo = float(row.get("min_edge", -1))
        hi = float(row.get("max_edge", -1))
        if lo <= x < hi:
            return row
    return None


def annotate_selected_regimes(frame, report):
    out = frame.copy()
    fields = {
        "edge_regime_status": "UNSUPPORTED",
        "edge_regime_band": "",
        "edge_regime_bets": 0,
        "edge_regime_roi": np.nan,
        "edge_regime_profitable_seasons": 0,
        "edge_regime_season_count": 0,
        "edge_regime_verified_entry_bets": 0,
        "edge_regime_verified_entry_rate": 0.0,
        "edge_regime_candidate": False,
    }
    for key, default in fields.items():
        if key not in out.columns:
            out[key] = default

    for idx, row in out.iterrows():
        regime = match_edge_regime(
            report,
            row.get("quant_market"),
            row.get("quant_edge"),
        )
        if not regime:
            continue
        out.at[idx, "edge_regime_status"] = regime.get("status", "UNSUPPORTED")
        out.at[idx, "edge_regime_band"] = regime.get("edge_band", "")
        out.at[idx, "edge_regime_bets"] = int(regime.get("bets", 0))
        out.at[idx, "edge_regime_roi"] = regime.get("roi")
        out.at[idx, "edge_regime_profitable_seasons"] = int(
            regime.get("profitable_seasons", 0)
        )
        out.at[idx, "edge_regime_season_count"] = int(regime.get("season_count", 0))
        out.at[idx, "edge_regime_verified_entry_bets"] = int(
            regime.get("verified_entry_bets", 0) or 0
        )
        out.at[idx, "edge_regime_verified_entry_rate"] = float(
            regime.get("verified_entry_rate", 0.0) or 0.0
        )
        out.at[idx, "edge_regime_candidate"] = (
            regime.get("status") == "PERSISTENT_CANDIDATE"
        )
    return out


def _market_candidate(row, market):
    home = row.get("home_team")
    away = row.get("away_team")
    if market == "spread":
        side = row.get("spread_team")
        edge = row.get("spread_edge_pts")
        probability = row.get("cover_probability")
        line = row.get("spread_line")
        odds = row.get("spread_odds")
        book = row.get("spread_book")
        quote_at = row.get("spread_quote_at")
        badge = row.get("spread_badge")
    elif market == "total":
        side = row.get("total_dir")
        edge = row.get("total_edge_pts")
        probability = row.get("total_probability")
        line = row.get("market_total")
        odds = row.get("total_odds")
        book = row.get("total_book")
        quote_at = row.get("total_quote_at")
        badge = row.get("total_badge")
    else:
        side = row.get("quant_best_ml_side")
        edge = row.get("quant_best_ml_edge_pp")
        if side == home:
            probability = row.get("calibrated_home_probability")
            odds = row.get("best_home_ml")
            book = row.get("best_home_ml_book")
            quote_at = row.get("best_home_ml_quote_at")
        elif side == away and _finite(row.get("calibrated_home_probability")):
            probability = 1.0 - float(row.get("calibrated_home_probability"))
            odds = row.get("best_away_ml")
            book = row.get("best_away_ml_book")
            quote_at = row.get("best_away_ml_quote_at")
        else:
            probability = odds = book = quote_at = None
        line = odds
        badge = row.get("ml_badge")

    if not (
        _finite(edge)
        and _finite(probability)
        and _finite(line)
        and _finite(odds)
        and str(book or "").strip()
        and str(quote_at or "").strip()
    ):
        return None
    p = float(probability)
    price = float(odds)
    return {
        "market": market,
        "side": side,
        "line": float(line),
        "odds": price,
        "probability": p,
        "edge": abs(float(edge)),
        "ev": float(roi(p, price)),
        "badge": str(badge or ""),
        "book": book,
        "quote_at": quote_at,
    }


def current_edge_board(frame, report, statuses=None):
    """Return all executable current market candidates in selected evidence regimes."""

    statuses = set(statuses or {"PERSISTENT_CANDIDATE"})
    rows = []
    for _, game in frame.iterrows():
        for market in ("spread", "total", "moneyline"):
            candidate = _market_candidate(game, market)
            if not candidate:
                continue
            regime = match_edge_regime(report, market, candidate["edge"])
            if not regime or regime.get("status") not in statuses:
                continue
            rows.append(
                {
                    "game_id": game.get("game_id"),
                    "date": game.get("date"),
                    "away_team": game.get("away_team"),
                    "home_team": game.get("home_team"),
                    **candidate,
                    "regime_status": regime.get("status"),
                    "regime_band": regime.get("edge_band"),
                    "historical_bets": regime.get("bets"),
                    "historical_win_rate": regime.get("win_rate"),
                    "historical_roi": regime.get("roi"),
                    "historical_avg_clv": regime.get("avg_clv"),
                    "profitable_seasons": regime.get("profitable_seasons"),
                    "season_count": regime.get("season_count"),
                    "verified_entry_bets": regime.get("verified_entry_bets"),
                    "verified_entry_rate": regime.get("verified_entry_rate"),
                    "entry_quote_sources": json.dumps(
                        regime.get("entry_quote_sources") or {},
                        sort_keys=True,
                    ),
                    "currently_selected": (
                        str(game.get("quant_market") or "").lower() == market
                        and str(game.get("quant_side") or "") == str(candidate["side"])
                    ),
                    "selected_quant_signal": game.get("quant_signal"),
                    "selected_quant_ev": game.get("quant_ev"),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["regime_status", "historical_roi", "ev"],
            ascending=[True, False, False],
        ).reset_index(drop=True)
    return out
