from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_POLICY = {
    "version": 2,
    "deployment_mode": "paper",
    "markets": {
        "moneyline": {"enabled": True, "excluded_weeks": [], "lean": {"min_ev": .02, "min_edge": 1.5, "min_prob": .52}, "bet": {"min_ev": .04, "min_edge": 2.5, "min_prob": .54}, "strong": {"min_ev": .07, "min_edge": 4.0, "min_prob": .56}},
        "spread": {"enabled": True, "excluded_weeks": [], "lean": {"min_ev": .02, "min_edge": 2.0, "min_prob": .52}, "bet": {"min_ev": .04, "min_edge": 3.0, "min_prob": .54}, "strong": {"min_ev": .07, "min_edge": 5.0, "min_prob": .57}},
        "total": {"enabled": True, "excluded_weeks": [], "lean": {"min_ev": .02, "min_edge": 2.5, "min_prob": .52}, "bet": {"min_ev": .04, "min_edge": 4.0, "min_prob": .54}, "strong": {"min_ev": .07, "min_edge": 6.0, "min_prob": .57}},
    },
    "portfolio": {"max_slate_units": 5.0, "max_game_units": 1.0, "max_team_units": 1.5, "max_market_units": 2.5, "kelly_fraction": .20},
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
    base.update({k: v for k, v in data.items() if k != "markets"})
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
        return False, str(cfg.get("disabled_reason") or "market did not pass out-of-sample policy validation")
    if week is not None:
        try:
            w = int(week)
            if w in {int(x) for x in cfg.get("excluded_weeks", [])}:
                return False, f"week {w} failed repeated prior/holdout validation"
        except Exception:
            pass
    return True, "validated policy path"


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


def _split_time(df: pd.DataFrame):
    if "season" in df.columns and df["season"].nunique() >= 2:
        last = int(pd.to_numeric(df["season"], errors="coerce").dropna().max())
        return df[df["season"] < last].copy(), df[df["season"] == last].copy(), f"season<{last} / season={last}"
    n = len(df)
    cut = max(1, int(n * .7))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy(), "chronological 70/30"


def _filtered(df, ev, edge, prob):
    return df[
        (pd.to_numeric(df.ev, errors="coerce") >= ev)
        & (pd.to_numeric(df.edge, errors="coerce").abs() >= edge)
        & (pd.to_numeric(df.probability, errors="coerce") >= prob)
    ].copy()


def _repeated_weak_weeks(train: pd.DataFrame, hold: pd.DataFrame, ev: float, edge: float, prob: float) -> list[int]:
    """Exclude only weeks that are negative in both prior seasons and untouched holdout.

    This is deliberately conservative: a week needs adequate sample in each side of the
    time split and must be negative twice. It prevents one historically bad week from
    being hard-coded after looking at the complete sample.
    """
    if "week" not in train.columns or "week" not in hold.columns:
        return []
    tr = _filtered(train, ev, edge, prob)
    va = _filtered(hold, ev, edge, prob)
    out = []
    weeks = sorted(set(pd.to_numeric(tr.week, errors="coerce").dropna().astype(int)) & set(pd.to_numeric(va.week, errors="coerce").dropna().astype(int)))
    for w in weeks:
        a = _profit_stats(tr[pd.to_numeric(tr.week, errors="coerce") == w])
        b = _profit_stats(va[pd.to_numeric(va.week, errors="coerce") == w])
        if a["n"] >= 40 and b["n"] >= 15 and a["roi"] is not None and b["roi"] is not None and a["roi"] < 0 and b["roi"] < 0:
            out.append(int(w))
    return out


def derive_production_policy(bets_path="reports/backtest_bets.csv", summary_path="reports/backtest_summary.json", out_path="reports/production_policy.json"):
    policy = _deepcopy_default()
    bp = Path(bets_path)
    if not bp.exists():
        Path(out_path).write_text(json.dumps(policy, indent=2))
        return policy
    bets = pd.read_csv(bp, low_memory=False)
    if bets.empty or not {"market", "ev", "edge", "probability", "profit"}.issubset(bets.columns):
        Path(out_path).write_text(json.dumps(policy, indent=2))
        return policy
    bets = bets.sort_values([c for c in ("season", "week", "game_id") if c in bets.columns]).reset_index(drop=True)
    train, hold, split_desc = _split_time(bets)
    diagnostics = {}
    grids = {
        "moneyline": ([.02, .03, .04, .05, .06, .07, .08, .10], [1.5, 2, 2.5, 3, 4, 5, 6], [.52, .54, .56, .58, .60]),
        "spread": ([.02, .03, .04, .05, .06, .07, .08, .10], [2, 2.5, 3, 3.5, 4, 5, 6], [.52, .54, .56, .58, .60]),
        "total": ([.02, .03, .04, .05, .06, .07, .08, .10], [2.5, 3, 3.5, 4, 5, 6, 7], [.52, .54, .56, .58, .60]),
    }

    for market, (evs, edges, probs) in grids.items():
        tr = train[train.market.astype(str).str.lower() == market].copy()
        va = hold[hold.market.astype(str).str.lower() == market].copy()
        candidates = []
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
            if hs["n"] >= 20 and (hs["roi"] or -1) >= 0 and (hs["avg_clv"] is None or hs["avg_clv"] >= 0):
                chosen = (ev, edge, prob, hs)
                break

        diagnostics[market] = {"train_bets": int(len(tr)), "holdout_bets": int(len(va)), "selected": chosen}
        cfg = policy["markets"][market]
        if not chosen:
            # Critical fail-closed behavior: an unvalidated market cannot emit live
            # LEAN/BET/STRONG labels merely because conservative defaults exist.
            cfg["enabled"] = False
            cfg["disabled_reason"] = "no threshold passed the independent time holdout with non-negative ROI and CLV"
            cfg["excluded_weeks"] = []
            continue

        ev, edge, prob, hs = chosen
        d = DEFAULT_POLICY["markets"][market]
        floor = {
            "min_ev": max(float(d["lean"]["min_ev"]), ev),
            "min_edge": max(float(d["lean"]["min_edge"]), edge),
            "min_prob": max(float(d["lean"]["min_prob"]), prob),
        }
        cfg["enabled"] = True
        cfg.pop("disabled_reason", None)
        cfg["lean"] = floor
        cfg["bet"] = {
            "min_ev": max(float(d["bet"]["min_ev"]), floor["min_ev"] + .015),
            "min_edge": max(float(d["bet"]["min_edge"]), floor["min_edge"] + .75),
            "min_prob": max(float(d["bet"]["min_prob"]), floor["min_prob"] + .015),
        }
        cfg["strong"] = {
            "min_ev": max(float(d["strong"]["min_ev"]), floor["min_ev"] + .04),
            "min_edge": max(float(d["strong"]["min_edge"]), floor["min_edge"] + 2.0),
            "min_prob": max(float(d["strong"]["min_prob"]), floor["min_prob"] + .03),
        }
        cfg["excluded_weeks"] = _repeated_weak_weeks(tr, va, floor["min_ev"], floor["min_edge"], floor["min_prob"])
        diagnostics[market]["excluded_weeks"] = cfg["excluded_weeks"]

    summary = {}
    try:
        summary = json.loads(Path(summary_path).read_text())
    except Exception:
        pass
    overall = summary.get("overall") or {}
    ci = overall.get("roi_ci_95") or [None, None]
    robust = (
        int(overall.get("bets", 0) or 0) >= 500
        and ci[0] is not None and float(ci[0]) > 0
        and overall.get("avg_clv") is not None and float(overall["avg_clv"]) > 0
    )
    policy["deployment_mode"] = "production" if robust else "paper"
    policy["source"] = "time-split backtest policy calibration with fail-closed market and repeated-week gates"
    policy["split"] = split_desc
    policy["diagnostics"] = diagnostics
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(policy, indent=2))
    return policy
