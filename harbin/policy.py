from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .entry_provenance import verified_entry_mask

DEFAULT_POLICY = {
    "version": 4,
    "deployment_mode": "paper",
    "markets": {
        "moneyline": {"enabled": True, "excluded_weeks": [], "lean": {"min_ev": .02, "min_edge": 1.5, "min_prob": .52}, "bet": {"min_ev": .04, "min_edge": 2.5, "min_prob": .54}, "strong": {"min_ev": .07, "min_edge": 4.0, "min_prob": .56}},
        "spread": {"enabled": True, "excluded_weeks": [], "lean": {"min_ev": .02, "min_edge": 2.0, "min_prob": .52}, "bet": {"min_ev": .04, "min_edge": 3.0, "min_prob": .54}, "strong": {"min_ev": .07, "min_edge": 5.0, "min_prob": .57}},
        "total": {"enabled": True, "excluded_weeks": [], "lean": {"min_ev": .02, "min_edge": 2.5, "min_prob": .52}, "bet": {"min_ev": .04, "min_edge": 4.0, "min_prob": .54}, "strong": {"min_ev": .07, "min_edge": 6.0, "min_prob": .57}},
    },
    "portfolio": {
        "max_slate_units": 5.0,
        "max_game_units": 1.0,
        "max_team_units": 1.5,
        "max_market_units": 2.5,
        "max_kickoff_window_units": 2.0,
        "kickoff_window_hours": 3,
        "max_book_units": 2.0,
        "max_bets": 20,
        "min_allocation_units": .05,
        "kelly_fraction": .20,
        "drawdown_soft_stop_units": 8.0,
        "drawdown_hard_stop_units": 15.0,
        "drawdown_floor_multiplier": .25,
        "trailing_window_bets": 50,
        "min_trailing_bets_for_throttle": 30,
        "trailing_roi_throttle": -.10,
        "trailing_clv_throttle": 0.0,
        "adverse_run_multiplier": .50,
        "enable_performance_feedback": True,
        "feedback_min_segment_bets": 20,
        "feedback_min_clv_coverage": .60,
        "feedback_weak_multiplier": .75,
        "feedback_severe_multiplier": .50,
        "feedback_severe_min_bets": 40,
        "feedback_severe_roi": -.05,
        "feedback_severe_positive_clv_rate": .45,
        "require_executable_book": True,
        "min_market_book_count_for_execution": 1,
        "require_quote_timestamp_for_execution": True,
        "max_quote_age_minutes": 60,
        "require_live_history_for_production": True,
    },
    "source": "conservative defaults",
}


def _deepcopy_default():
    return json.loads(json.dumps(DEFAULT_POLICY))


def load_policy(path="reports/production_policy.json"):
    p = Path(path)
    if not p.exists():
        return _deepcopy_default()
    try:
        data = json.loads(p.read_text())
    except Exception:
        return _deepcopy_default()
    base = _deepcopy_default()
    base.update({k: v for k, v in data.items() if k not in {"markets", "portfolio"}})
    incoming_portfolio = data.get("portfolio") or {}
    if isinstance(incoming_portfolio, dict):
        base["portfolio"].update(incoming_portfolio)
    for market in base["markets"]:
        incoming = (data.get("markets") or {}).get(market, {})
        base["markets"][market].update({k: v for k, v in incoming.items() if k not in {"lean", "bet", "strong"}})
        for tier in ("lean", "bet", "strong"):
            base["markets"][market][tier].update(incoming.get(tier, {}))
    return base


def thresholds_for(market: str, path="reports/production_policy.json"):
    return load_policy(path).get("markets", {}).get(market, DEFAULT_POLICY["markets"][market])


def market_allowed(market: str, week=None, path="reports/production_policy.json") -> tuple[bool, str]:
    cfg = thresholds_for(market, path)
    if not bool(cfg.get("enabled", True)):
        return False, str(cfg.get("disabled_reason") or "market did not pass untouched evaluation")
    if week is not None:
        try:
            w = int(week)
            if w in {int(x) for x in cfg.get("excluded_weeks", [])}:
                return False, f"week {w} failed repeated development/tuning validation"
        except Exception:
            pass
    return True, "validated nested policy path"


def signal_from_policy(ev: float, edge: float, probability: float, market: str, path="reports/production_policy.json", week=None) -> str:
    allowed, _ = market_allowed(market, week=week, path=path)
    if not allowed:
        return "PASS"
    try:
        ev, edge, probability = float(ev), abs(float(edge)), float(probability)
    except Exception:
        return "PASS"
    if ev <= 0:
        return "PASS"
    t = thresholds_for(market, path)
    for label in ("strong", "bet", "lean"):
        q = t[label]
        if ev >= float(q["min_ev"]) and edge >= float(q["min_edge"]) and probability >= float(q["min_prob"]):
            return label.upper()
    return "PASS"


def _profit_stats(df: pd.DataFrame):
    if df.empty:
        return {"n": 0, "roi": None, "lcb": None, "avg_clv": None}
    p = pd.to_numeric(df["profit"], errors="coerce").dropna().to_numpy(float)
    if not len(p):
        return {"n": 0, "roi": None, "lcb": None, "avg_clv": None}
    roi = float(p.mean())
    se = float(p.std(ddof=1) / math.sqrt(len(p))) if len(p) > 1 else 1.0
    clv = pd.to_numeric(df.get("clv", pd.Series(dtype=float)), errors="coerce").dropna()
    return {"n": int(len(p)), "roi": roi, "lcb": float(roi - 1.28 * se), "avg_clv": float(clv.mean()) if len(clv) else None}


def _promotion_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Only explicitly verified entries may calibrate a betting policy.

    Archive opening-field presence alone (used_distinct_open) is not verification.
    """
    flag = verified_entry_mask(df)
    return df.loc[flag].copy()


def _chronological_blocks(df: pd.DataFrame):
    cols = [c for c in ("season", "week", "game_id") if c in df.columns]
    d = df.sort_values(cols, kind="mergesort").reset_index(drop=True) if cols else df.reset_index(drop=True)
    if {"season", "week"}.issubset(d.columns):
        blocks = list(d.groupby(["season", "week"], sort=True, dropna=False))
        return d, blocks
    return d, []


def _nested_split(df: pd.DataFrame):
    """Development -> tune -> untouched evaluation. Evaluation never selects policy."""
    d, blocks = _chronological_blocks(df)
    seasons = sorted(pd.to_numeric(d.get("season"), errors="coerce").dropna().astype(int).unique()) if "season" in d else []
    if len(seasons) >= 3:
        tune_season, eval_season = seasons[-2], seasons[-1]
        dev = d[pd.to_numeric(d.season, errors="coerce") < tune_season].copy()
        tune = d[pd.to_numeric(d.season, errors="coerce") == tune_season].copy()
        evaluation = d[pd.to_numeric(d.season, errors="coerce") == eval_season].copy()
        return dev, tune, evaluation, f"season<{tune_season} / tune={tune_season} / untouched={eval_season}"
    if len(blocks) >= 8:
        n = len(blocks)
        i1 = max(1, int(round(.50 * n)))
        i2 = max(i1 + 1, int(round(.75 * n)))
        i2 = min(i2, n - 1)
        dev_keys = {k for k, _ in blocks[:i1]}
        tune_keys = {k for k, _ in blocks[i1:i2]}
        keys = list(zip(pd.to_numeric(d.season, errors="coerce"), pd.to_numeric(d.week, errors="coerce")))
        dev = d[[k in dev_keys for k in keys]].copy()
        tune = d[[k in tune_keys for k in keys]].copy()
        evaluation = d[[k not in dev_keys and k not in tune_keys for k in keys]].copy()
        return dev, tune, evaluation, "whole-week 50/25/25 nested chronology"
    n = len(d); a = max(1, int(.50 * n)); b = min(n - 1, max(a + 1, int(.75 * n)))
    return d.iloc[:a].copy(), d.iloc[a:b].copy(), d.iloc[b:].copy(), "chronological 50/25/25 fallback"


def _filtered(df, ev, edge, prob):
    return df[
        (pd.to_numeric(df.ev, errors="coerce") >= ev)
        & (pd.to_numeric(df.edge, errors="coerce").abs() >= edge)
        & (pd.to_numeric(df.probability, errors="coerce") >= prob)
    ].copy()


def _repeated_weak_weeks(dev: pd.DataFrame, tune: pd.DataFrame, ev: float, edge: float, prob: float) -> list[int]:
    """Week exclusions are selected before the untouched evaluation period."""
    if "week" not in dev.columns or "week" not in tune.columns:
        return []
    tr = _filtered(dev, ev, edge, prob); va = _filtered(tune, ev, edge, prob); out = []
    weeks = sorted(set(pd.to_numeric(tr.week, errors="coerce").dropna().astype(int)) & set(pd.to_numeric(va.week, errors="coerce").dropna().astype(int)))
    for w in weeks:
        a = _profit_stats(tr[pd.to_numeric(tr.week, errors="coerce") == w])
        b = _profit_stats(va[pd.to_numeric(va.week, errors="coerce") == w])
        if a["n"] >= 40 and b["n"] >= 15 and a["roi"] is not None and b["roi"] is not None and a["roi"] < 0 and b["roi"] < 0:
            out.append(int(w))
    return out


def _candidate_grid(market):
    if market == "moneyline": return ([.02,.03,.04,.05,.06,.07,.08,.10],[1.5,2,2.5,3,4,5,6],[.52,.54,.56,.58,.60])
    if market == "spread": return ([.02,.03,.04,.05,.06,.07,.08,.10],[2,2.5,3,3.5,4,5,6],[.52,.54,.56,.58,.60])
    return ([.02,.03,.04,.05,.06,.07,.08,.10],[2.5,3,3.5,4,5,6,7],[.52,.54,.56,.58,.60])


def derive_production_policy(
    bets_path="reports/backtest_bets.csv",
    summary_path="reports/backtest_summary.json",
    out_path="reports/production_policy.json",
    evidence_path="reports/evidence_report.json",
):
    policy = _deepcopy_default(); bp = Path(bets_path)
    if not bp.exists():
        Path(out_path).write_text(json.dumps(policy, indent=2)); return policy
    raw = pd.read_csv(bp, low_memory=False)
    bets = _promotion_sample(raw)
    required = {"market", "ev", "edge", "probability", "profit"}
    if bets.empty or not required.issubset(bets.columns):
        policy["source"] = "paper only; no verified archived opening-entry sample"
        Path(out_path).write_text(json.dumps(policy, indent=2)); return policy
    bets, _ = _chronological_blocks(bets)
    dev, tune, evaluation, split_desc = _nested_split(bets)
    diagnostics = {"selection_uses_evaluation": False, "promotion_rows": int(len(bets)), "raw_archive_rows": int(len(raw))}

    passed_markets = 0
    for market in ("moneyline", "spread", "total"):
        tr = dev[dev.market.astype(str).str.lower() == market].copy()
        va = tune[tune.market.astype(str).str.lower() == market].copy()
        evl = evaluation[evaluation.market.astype(str).str.lower() == market].copy()
        candidates = []
        evs, edges, probs = _candidate_grid(market)
        for ev in evs:
            for edge in edges:
                for prob in probs:
                    st = _profit_stats(_filtered(tr, ev, edge, prob))
                    if st["n"] >= 40 and st["roi"] is not None:
                        candidates.append((st["lcb"], st["roi"], st["n"], ev, edge, prob))
        candidates.sort(reverse=True)
        chosen = None
        for _, _, _, ev, edge, prob in candidates[:25]:
            hs = _profit_stats(_filtered(va, ev, edge, prob))
            if hs["n"] >= 20 and hs["roi"] is not None and hs["roi"] >= 0 and hs["avg_clv"] is not None and hs["avg_clv"] >= 0:
                chosen = (ev, edge, prob, hs); break

        cfg = policy["markets"][market]
        diag = {"development_bets": int(len(tr)), "tune_bets": int(len(va)), "evaluation_bets": int(len(evl)), "selected": chosen}
        if not chosen:
            cfg["enabled"] = False; cfg["disabled_reason"] = "no threshold passed pre-evaluation development/tuning validation"; cfg["excluded_weeks"] = []
            diag["evaluation"] = None; diag["evaluation_passed"] = False; diagnostics[market] = diag; continue

        ev, edge, prob, _ = chosen; d = DEFAULT_POLICY["markets"][market]
        floor = {"min_ev": max(float(d["lean"]["min_ev"]), ev), "min_edge": max(float(d["lean"]["min_edge"]), edge), "min_prob": max(float(d["lean"]["min_prob"]), prob)}
        cfg["lean"] = floor
        cfg["bet"] = {"min_ev": max(float(d["bet"]["min_ev"]), floor["min_ev"] + .015), "min_edge": max(float(d["bet"]["min_edge"]), floor["min_edge"] + .75), "min_prob": max(float(d["bet"]["min_prob"]), floor["min_prob"] + .015)}
        cfg["strong"] = {"min_ev": max(float(d["strong"]["min_ev"]), floor["min_ev"] + .04), "min_edge": max(float(d["strong"]["min_edge"]), floor["min_edge"] + 2.0), "min_prob": max(float(d["strong"]["min_prob"]), floor["min_prob"] + .03)}
        cfg["excluded_weeks"] = _repeated_weak_weeks(tr, va, floor["min_ev"], floor["min_edge"], floor["min_prob"])
        eval_filtered = _filtered(evl, floor["min_ev"], floor["min_edge"], floor["min_prob"])
        if cfg["excluded_weeks"] and "week" in eval_filtered:
            eval_filtered = eval_filtered[~pd.to_numeric(eval_filtered.week, errors="coerce").isin(cfg["excluded_weeks"])]
        es = _profit_stats(eval_filtered)
        passed = es["n"] >= 20 and es["roi"] is not None and es["roi"] >= 0 and es["avg_clv"] is not None and es["avg_clv"] >= 0
        cfg["enabled"] = bool(passed)
        if not passed: cfg["disabled_reason"] = "frozen threshold failed untouched chronological evaluation"
        else: cfg.pop("disabled_reason", None); passed_markets += 1
        diag.update({"excluded_weeks": cfg["excluded_weeks"], "evaluation": es, "evaluation_passed": bool(passed)})
        diagnostics[market] = diag

    try: evidence = json.loads(Path(evidence_path).read_text())
    except Exception: evidence = {}
    robust = str(evidence.get("status") or "").upper() == "ROBUST" and bool((evidence.get("promotion_sample") or {}).get("entry_quote_verified", False))
    policy["deployment_mode"] = "production" if robust and passed_markets >= 2 else "paper"
    policy["source"] = "nested chronological policy calibration on verified archived opening entries; untouched evaluation is release-only; Stage 5/7 execution defaults fail closed"
    policy["split"] = split_desc
    policy["diagnostics"] = diagnostics
    Path(out_path).parent.mkdir(parents=True, exist_ok=True); Path(out_path).write_text(json.dumps(policy, indent=2)); return policy
