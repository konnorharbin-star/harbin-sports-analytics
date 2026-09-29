from __future__ import annotations

"""Production hardening for the historical market backtest.

This module fixes nullable archive IDs, normalizes closing-line value onto a
comparable probability scale, uses week-block bootstrap confidence intervals,
and adds market-role / conference diagnostics without changing the core score
model.  The backtest still bets only prices that were available in the archive
and trains each evaluation week using prior games only.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as _bt
from .data import SportsDataVerseClient
from .market import norm_cdf


def canonical_game_id(value) -> str:
    try:
        if value is None:
            return ""
        s = str(value).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return ""
        return str(int(float(s)))
    except Exception:
        return str(value).strip()


class CanonicalArchiveMarketStore(_bt.ArchiveMarketStore):
    def __init__(self, cache_dir="cache/backtest"):
        super().__init__(cache_dir=cache_dir)
        if len(self.df) and "game_id" in self.df.columns:
            self.df["_gid"] = self.df["game_id"].map(canonical_game_id)
            self.df = self.df[self.df["_gid"] != ""].copy()
            self.by_id = {k: v.copy() for k, v in self.df.groupby("_gid", sort=False)}


def normalize_clv(raw_clv, market: str, sigma_margin=16.0, sigma_total=16.0):
    """Return CLV as probability-equivalent improvement for cross-market use.

    Moneyline CLV is already represented by the underlying backtester as the
    change in no-vig win probability. Spread / total CLV is stored in points,
    so convert it to probability-equivalent distance from a 50/50 close.
    Positive is always favorable to the bettor.
    """
    try:
        raw = float(raw_clv)
        if not math.isfinite(raw):
            return np.nan
    except Exception:
        return np.nan
    market = str(market).lower()
    if market == "moneyline":
        return raw
    sigma = float(sigma_margin if market == "spread" else sigma_total)
    return float(norm_cdf(raw / max(6.0, sigma)) - 0.5)


def _market_role(bet, game):
    market = str(bet.get("market") or "")
    side = str(bet.get("side") or "")
    if market == "total":
        return "over" if side == "O" else "under" if side == "U" else "total"
    try:
        price_or_line = float(bet.get("line"))
    except Exception:
        return "unknown"
    if market == "spread":
        return "favorite" if price_or_line < 0 else "underdog" if price_or_line > 0 else "pickem"
    if market == "moneyline":
        return "favorite" if price_or_line < 0 else "underdog" if price_or_line > 0 else "pickem"
    return "unknown"


def _side_location(bet, game):
    side = str(bet.get("side") or "")
    if side == str(game.home_team):
        return "home"
    if side == str(game.away_team):
        return "away"
    if side == "O":
        return "over"
    if side == "U":
        return "under"
    return "other"


def _closing_line(bet, game, q):
    market = str(bet.get("market") or "")
    side = str(bet.get("side") or "")
    if market == "spread":
        h = q.get("final_home_spread", np.nan)
        if pd.isna(h):
            return np.nan
        return float(h) if side == game.home_team else -float(h)
    if market == "total":
        return q.get("final_total", np.nan)
    if market == "moneyline":
        return q.get("final_home_ml", np.nan) if side == game.home_team else q.get("final_away_ml", np.nan)
    return np.nan


_orig_market_bets = _bt._market_bets


def _rigorous_market_bets(game, margin, total, p_home, sigma_m, sigma_t, q):
    bets = _orig_market_bets(game, margin, total, p_home, sigma_m, sigma_t, q)
    for b in bets:
        raw = b.get("clv", np.nan)
        b["clv_raw"] = raw
        b["clv"] = normalize_clv(raw, b.get("market"), sigma_m, sigma_t)
        b["market_role"] = _market_role(b, game)
        b["side_location"] = _side_location(b, game)
        b["closing_line"] = _closing_line(b, game, q)
        b["used_distinct_open"] = bool(q.get("has_distinct_open", False))
    return bets


def _block_bootstrap_roi_ci(df: pd.DataFrame, seed=26, n_boot=3000):
    if df.empty or len(df) < 30:
        return [None, None]
    if not {"season", "week"}.issubset(df.columns):
        x = pd.to_numeric(df.profit, errors="coerce").dropna().to_numpy(float)
        if len(x) < 30:
            return [None, None]
        rng = np.random.default_rng(seed)
        vals = [float(rng.choice(x, size=len(x), replace=True).mean()) for _ in range(n_boot)]
        return [float(v) for v in np.quantile(vals, [.025, .975])]
    blocks = []
    for _, g in df.groupby(["season", "week"], sort=False):
        p = pd.to_numeric(g.profit, errors="coerce").dropna().to_numpy(float)
        if len(p):
            blocks.append((float(p.sum()), int(len(p))))
    if len(blocks) < 8:
        return _block_bootstrap_roi_ci(df.drop(columns=[c for c in ("season", "week") if c in df.columns]), seed, n_boot)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(blocks)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        units = sum(blocks[i][0] for i in idx)
        bets = sum(blocks[i][1] for i in idx)
        vals.append(units / max(1, bets))
    return [float(v) for v in np.quantile(vals, [.025, .975])]


def _max_drawdown(profits):
    x = np.asarray(profits, dtype=float)
    if not len(x):
        return 0.0
    curve = np.cumsum(x)
    peaks = np.maximum.accumulate(np.r_[0.0, curve])[:-1]
    return float(np.max(peaks - curve))


def _rigorous_group_summary(df: pd.DataFrame):
    if df.empty:
        return {"bets": 0, "wins": 0, "losses": 0, "pushes": 0, "win_rate": None, "units": 0.0, "roi": None, "max_drawdown": 0.0, "avg_clv": None, "avg_clv_raw": None, "positive_clv_rate": None, "clv_samples": 0, "roi_ci_95": [None, None]}
    work = df.copy()
    wins = int((work.result > 0).sum()); losses = int((work.result < 0).sum()); pushes = int((work.result == 0).sum())
    profits = pd.to_numeric(work.profit, errors="coerce").fillna(0).to_numpy(float)
    clv = pd.to_numeric(work.get("clv"), errors="coerce").dropna()
    raw = pd.to_numeric(work.get("clv_raw"), errors="coerce").dropna()
    weekly = work.groupby(["season", "week"], dropna=False).profit.sum() if {"season", "week"}.issubset(work.columns) else pd.Series(dtype=float)
    return {
        "bets": int(len(work)), "wins": wins, "losses": losses, "pushes": pushes,
        "win_rate": float(wins / max(1, wins + losses)), "units": float(profits.sum()),
        "roi": float(profits.mean()), "max_drawdown": _max_drawdown(profits),
        "avg_clv": float(clv.mean()) if len(clv) else None,
        "avg_clv_raw": float(raw.mean()) if len(raw) else None,
        "positive_clv_rate": float((clv > 0).mean()) if len(clv) else None,
        "clv_samples": int(len(clv)), "roi_ci_95": _block_bootstrap_roi_ci(work),
        "profitable_week_rate": float((weekly > 0).mean()) if len(weekly) else None,
    }


def _conference_map(start_season: int, end_season: int):
    client = SportsDataVerseClient(); out = {}
    for season in range(int(start_season), int(end_season) + 1):
        try:
            d = client.season_frame(season)
        except Exception:
            continue
        hc = next((c for c in ("home_conference", "home_conference_name", "home_conf") if c in d.columns), None)
        ac = next((c for c in ("away_conference", "away_conference_name", "away_conf") if c in d.columns), None)
        if not hc and not ac:
            continue
        for _, r in d.iterrows():
            gid = canonical_game_id(r.get("game_id"))
            if gid:
                out[gid] = (str(r.get(hc, "Unknown")) if hc else "Unknown", str(r.get(ac, "Unknown")) if ac else "Unknown")
    return out


def _attach_segments(bdf: pd.DataFrame, start_season: int, end_season: int):
    if bdf.empty:
        return bdf
    cmap = _conference_map(start_season, end_season)
    out = bdf.copy(); homes = []; aways = []; chosen = []
    for _, r in out.iterrows():
        h, a = cmap.get(canonical_game_id(r.get("game_id")), ("Unknown", "Unknown")); homes.append(h); aways.append(a)
        side = str(r.get("side") or "")
        chosen.append(h if side == str(r.get("home_team")) else a if side == str(r.get("away_team")) else "TOTAL")
    out["home_conference"] = homes; out["away_conference"] = aways; out["bet_conference"] = chosen
    return out


# Patch dynamic globals looked up by the original run_backtest implementation.
_bt.ArchiveMarketStore = CanonicalArchiveMarketStore
_bt._market_bets = _rigorous_market_bets
_bt._group_summary = _rigorous_group_summary
_orig_run_backtest = _bt.run_backtest


def run_backtest(start_season=2023, end_season=2025, history_start=2018, reports_dir="reports"):
    bdf, summary = _orig_run_backtest(start_season, end_season, history_start, reports_dir)
    reports = Path(reports_dir); reports.mkdir(parents=True, exist_ok=True)
    bdf = _attach_segments(bdf, start_season, end_season)
    bdf.to_csv(reports / "backtest_bets.csv", index=False)
    summary["overall"] = _rigorous_group_summary(bdf)
    for col, key in (("market", "by_market"), ("signal", "by_signal"), ("season", "by_season"), ("week", "by_week"), ("market_role", "by_role"), ("side_location", "by_side_location"), ("bet_conference", "by_conference")):
        summary[key] = {str(k): _rigorous_group_summary(v) for k, v in bdf.groupby(col, dropna=False)} if len(bdf) and col in bdf.columns else {}
    summary["methodology"] = {
        "bet_timestamp": "archived opening line/price when present; final quote used only when archive lacks a distinct opening value",
        "clv": "moneyline: no-vig probability movement; spread/total: line movement converted to probability-equivalent CLV using model residual sigma",
        "confidence_interval": "95% week-block bootstrap to preserve within-week wager correlation",
        "modeling": "weekly expanding-window fit; nested chronological tuning/calibration; advanced features are pregame as-of only",
    }
    (reports / "backtest_summary.json").write_text(json.dumps(summary, indent=2))
    for col, name in (("market_role", "backtest_by_role.csv"), ("bet_conference", "backtest_by_conference.csv"), ("season", "backtest_by_season.csv"), ("week", "backtest_by_week.csv")):
        rows=[]
        if len(bdf) and col in bdf.columns:
            for k, g in bdf.groupby(col, dropna=False): rows.append({col: k, **_rigorous_group_summary(g)})
        pd.DataFrame(rows).to_csv(reports / name, index=False)
    return bdf, summary
