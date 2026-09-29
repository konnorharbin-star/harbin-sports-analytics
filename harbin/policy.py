from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_POLICY = {
    "version": 3,
    "deployment_mode": "paper",
    "markets": {
        "moneyline": {"enabled": False, "evidence_tier": "UNVALIDATED", "lean": {"min_ev": .02, "min_edge": 1.5, "min_prob": .52}, "bet": {"min_ev": .04, "min_edge": 2.5, "min_prob": .54}, "strong": {"min_ev": .07, "min_edge": 4.0, "min_prob": .56}},
        "spread": {"enabled": False, "evidence_tier": "UNVALIDATED", "lean": {"min_ev": .02, "min_edge": 2.0, "min_prob": .52}, "bet": {"min_ev": .04, "min_edge": 3.0, "min_prob": .54}, "strong": {"min_ev": .07, "min_edge": 5.0, "min_prob": .57}},
        "total": {"enabled": False, "evidence_tier": "UNVALIDATED", "lean": {"min_ev": .02, "min_edge": 2.5, "min_prob": .52}, "bet": {"min_ev": .04, "min_edge": 4.0, "min_prob": .54}, "strong": {"min_ev": .07, "min_edge": 6.0, "min_prob": .57}},
    },
    "regime_filters": {"blocked_weeks": [], "reason": {}},
    "portfolio": {"max_slate_units": 5.0, "max_game_units": 1.0, "max_team_units": 1.5, "max_market_units": 2.5, "kelly_fraction": .20},
    "source": "safe defaults; betting signals disabled until validated",
}


def _deepcopy_default():
    return json.loads(json.dumps(DEFAULT_POLICY))


def _normalize_policy(data: dict) -> dict:
    base = _deepcopy_default()
    base.update({k: v for k, v in (data or {}).items() if k != "markets"})
    for market in base["markets"]:
        incoming = ((data or {}).get("markets") or {}).get(market, {})
        base["markets"][market].update(incoming)
        if "enabled" not in incoming:
            sel = (((data or {}).get("diagnostics") or {}).get(market) or {}).get("selected")
            ok = False
            if isinstance(sel, (list, tuple)) and len(sel) >= 4 and isinstance(sel[3], dict):
                st = sel[3]
                ok = int(st.get("n", 0) or 0) >= 50 and float(st.get("roi", -1) or -1) > 0 and float(st.get("lcb", -1) or -1) > 0
            base["markets"][market]["enabled"] = bool(ok)
            base["markets"][market]["evidence_tier"] = "VALIDATED" if ok else "UNVALIDATED"
    base.setdefault("regime_filters", {"blocked_weeks": [], "reason": {}})
    base["regime_filters"].setdefault("blocked_weeks", [])
    base["regime_filters"].setdefault("reason", {})
    return base


def load_policy(path="reports/production_policy.json"):
    p = Path(path)
    if not p.exists():
        return _deepcopy_default()
    try:
        return _normalize_policy(json.loads(p.read_text()))
    except Exception:
        return _deepcopy_default()


def thresholds_for(market: str, path="reports/production_policy.json"):
    return load_policy(path).get("markets", {}).get(market, DEFAULT_POLICY["markets"][market])


def market_enabled(market: str, path="reports/production_policy.json") -> bool:
    return bool(thresholds_for(market, path).get("enabled", False))


def signal_from_policy(ev: float, edge: float, probability: float, market: str, week=None, path="reports/production_policy.json") -> str:
    policy = load_policy(path)
    m = (policy.get("markets") or {}).get(market, {})
    if not bool(m.get("enabled", False)):
        return "PASS"
    try:
        if week is not None and int(week) in {int(x) for x in (policy.get("regime_filters") or {}).get("blocked_weeks", [])}:
            return "PASS"
        ev, edge, probability = float(ev), abs(float(edge)), float(probability)
    except Exception:
        return "PASS"
    if ev <= 0:
        return "PASS"
    for label in ("strong", "bet", "lean"):
        q = m[label]
        if ev >= float(q["min_ev"]) and edge >= float(q["min_edge"]) and probability >= float(q["min_prob"]):
            return label.upper()
    return "PASS"


def _cluster_bootstrap_roi(df: pd.DataFrame, draws=2000, seed=20260929):
    if df.empty:
        return [None, None]
    x = df.copy()
    x["profit"] = pd.to_numeric(x["profit"], errors="coerce")
    x = x.dropna(subset=["profit"])
    if x.empty:
        return [None, None]
    if {"season", "week"}.issubset(x.columns):
        blocks = [g["profit"].to_numpy(float) for _, g in x.groupby(["season", "week"], sort=False)]
    else:
        blocks = [x["profit"].to_numpy(float)]
    if len(blocks) < 3:
        p = x["profit"].to_numpy(float)
        se = p.std(ddof=1) / math.sqrt(len(p)) if len(p) > 1 else 1.0
        mu = float(p.mean())
        return [mu - 1.96 * se, mu + 1.96 * se]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(draws)):
        picks = rng.integers(0, len(blocks), size=len(blocks))
        sample = np.concatenate([blocks[i] for i in picks])
        vals.append(float(sample.mean()))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def _profit_stats(df: pd.DataFrame):
    if df.empty:
        return {"n": 0, "roi": None, "lcb": None, "ucb": None, "avg_clv": None}
    p = pd.to_numeric(df["profit"], errors="coerce").dropna()
    if not len(p):
        return {"n": 0, "roi": None, "lcb": None, "ucb": None, "avg_clv": None}
    ci = _cluster_bootstrap_roi(df)
    clv = pd.to_numeric(df.get("clv", pd.Series(dtype=float)), errors="coerce").dropna()
    return {"n": int(len(p)), "roi": float(p.mean()), "lcb": ci[0], "ucb": ci[1], "avg_clv": float(clv.mean()) if len(clv) else None}


def _split_three_way(df: pd.DataFrame):
    """Return development, tuning, untouched-final-test splits in time order."""
    if "season" in df.columns:
        seasons = sorted(int(x) for x in pd.to_numeric(df["season"], errors="coerce").dropna().unique())
        if len(seasons) >= 3:
            tune_season, test_season = seasons[-2], seasons[-1]
            dev = df[pd.to_numeric(df["season"], errors="coerce") < tune_season].copy()
            tune = df[pd.to_numeric(df["season"], errors="coerce") == tune_season].copy()
            test = df[pd.to_numeric(df["season"], errors="coerce") == test_season].copy()
            return dev, tune, test, f"development<{tune_season} / tune={tune_season} / untouched_test={test_season}"
    n = len(df)
    a, b = max(1, int(n * .55)), max(2, int(n * .78))
    b = min(b, n - 1)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy(), "chronological 55/23/22"


def _validated_candidate(st: dict, min_n=75):
    return st.get("n", 0) >= min_n and st.get("roi") is not None and st.get("lcb") is not None and float(st["lcb"]) > 0 and (st.get("avg_clv") is None or float(st["avg_clv"]) >= 0)


def _derive_blocked_weeks(summary: dict):
    blocked, reason = [], {}
    for wk, st in (summary.get("by_week") or {}).items():
        ci = st.get("roi_ci_95") or [None, None]
        n = int(st.get("bets", 0) or 0)
        if n >= 150 and ci[1] is not None and float(ci[1]) < 0:
            blocked.append(int(wk))
            reason[str(wk)] = {"bets": n, "roi": st.get("roi"), "roi_ci_95": ci, "rule": "historically negative with 95% upper bound below zero"}
    return sorted(set(blocked)), reason


def derive_production_policy(bets_path="reports/backtest_bets.csv", summary_path="reports/backtest_summary.json", out_path="reports/production_policy.json"):
    policy = _deepcopy_default()
    bp = Path(bets_path)
    if not bp.exists():
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(policy, indent=2))
        return policy
    bets = pd.read_csv(bp, low_memory=False)
    if bets.empty or not {"market", "ev", "edge", "probability", "profit"}.issubset(bets.columns):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(policy, indent=2))
        return policy
    sort_cols = [c for c in ("season", "week", "game_id") if c in bets.columns]
    bets = bets.sort_values(sort_cols).reset_index(drop=True)
    dev, tune, untouched_test, split_desc = _split_three_way(bets)
    diagnostics = {}
    grids = {
        "moneyline": ([.02,.03,.04,.05,.06,.07,.08,.10], [1.5,2,2.5,3,4,5,6], [.52,.54,.56,.58,.60,.62]),
        "spread": ([.02,.03,.04,.05,.06,.07,.08,.10,.12], [2,2.5,3,3.5,4,5,6,7], [.52,.54,.56,.58,.60,.62,.64]),
        "total": ([.02,.03,.04,.05,.06,.07,.08,.10,.12], [2.5,3,3.5,4,5,6,7,8], [.52,.54,.56,.58,.60,.62]),
    }
    for market, (evs, edges, probs) in grids.items():
        tr = dev[dev.market.astype(str).str.lower() == market].copy()
        va = tune[tune.market.astype(str).str.lower() == market].copy()
        te = untouched_test[untouched_test.market.astype(str).str.lower() == market].copy()
        candidates = []
        for ev in evs:
            for edge in edges:
                for prob in probs:
                    s = tr[(pd.to_numeric(tr.ev, errors="coerce") >= ev) & (pd.to_numeric(tr.edge, errors="coerce").abs() >= edge) & (pd.to_numeric(tr.probability, errors="coerce") >= prob)]
                    st = _profit_stats(s)
                    if st["n"] >= 50 and st["roi"] is not None:
                        candidates.append((st["lcb"] if st["lcb"] is not None else -999, st["roi"], st["n"], ev, edge, prob, st))
        candidates.sort(reverse=True)
        chosen = None
        for _, _, _, ev, edge, prob, trst in candidates[:50]:
            hv = va[(pd.to_numeric(va.ev, errors="coerce") >= ev) & (pd.to_numeric(va.edge, errors="coerce").abs() >= edge) & (pd.to_numeric(va.probability, errors="coerce") >= prob)]
            hs = _profit_stats(hv)
            if _validated_candidate(hs, 75):
                chosen = (ev, edge, prob, trst, hs)
                break
        diagnostics[market] = {"development_bets": int(len(tr)), "tuning_bets": int(len(va)), "untouched_test_bets": int(len(te)), "selected": chosen}
        if chosen:
            ev, edge, prob, trst, hs = chosen
            d = DEFAULT_POLICY["markets"][market]
            floor = {"min_ev": max(float(d["lean"]["min_ev"]), ev), "min_edge": max(float(d["lean"]["min_edge"]), edge), "min_prob": max(float(d["lean"]["min_prob"]), prob)}
            policy["markets"][market]["enabled"] = True
            policy["markets"][market]["evidence_tier"] = "TUNING_VALIDATED_AWAITING_FINAL_TEST"
            policy["markets"][market]["lean"] = floor
            policy["markets"][market]["bet"] = {"min_ev": max(float(d["bet"]["min_ev"]), floor["min_ev"] + .015), "min_edge": max(float(d["bet"]["min_edge"]), floor["min_edge"] + .75), "min_prob": max(float(d["bet"]["min_prob"]), floor["min_prob"] + .015)}
            policy["markets"][market]["strong"] = {"min_ev": max(float(d["strong"]["min_ev"]), floor["min_ev"] + .04), "min_edge": max(float(d["strong"]["min_edge"]), floor["min_edge"] + 2.0), "min_prob": max(float(d["strong"]["min_prob"]), floor["min_prob"] + .03)}
        else:
            policy["markets"][market]["enabled"] = False
            policy["markets"][market]["evidence_tier"] = "UNVALIDATED"
    try:
        summary = json.loads(Path(summary_path).read_text())
    except Exception:
        summary = {}
    blocked, reason = _derive_blocked_weeks(summary)
    policy["regime_filters"] = {"blocked_weeks": blocked, "reason": reason}
    enabled = [m for m, v in policy["markets"].items() if v.get("enabled")]
    policy["deployment_mode"] = "paper"
    policy["source"] = "three-way chronological threshold development/tuning with untouched final test; week-cluster bootstrap gates"
    policy["split"] = split_desc
    policy["untouched_test_seasons"] = sorted(int(x) for x in pd.to_numeric(untouched_test.get("season", pd.Series(dtype=float)), errors="coerce").dropna().unique())
    policy["diagnostics"] = diagnostics
    policy["validated_markets"] = enabled
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(policy, indent=2))
    return policy
