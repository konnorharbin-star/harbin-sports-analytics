from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .policy import load_policy, _profit_stats


def _final_test_frame(df: pd.DataFrame, policy: dict):
    seasons = [int(x) for x in (policy.get("untouched_test_seasons") or [])]
    if seasons and "season" in df.columns:
        s = pd.to_numeric(df["season"], errors="coerce")
        return df[s.isin(seasons)].copy(), seasons
    if "season" in df.columns:
        s = pd.to_numeric(df["season"], errors="coerce")
        valid = s.dropna()
        if len(valid):
            last = int(valid.max())
            return df[s == last].copy(), [last]
    cut = max(1, int(len(df) * .78))
    return df.iloc[cut:].copy(), []


def _select_by_threshold(frame: pd.DataFrame, cfg: dict):
    if frame.empty:
        return frame.copy()
    q = cfg.get("lean") or {}
    ev = pd.to_numeric(frame.get("ev"), errors="coerce")
    edge = pd.to_numeric(frame.get("edge"), errors="coerce").abs()
    prob = pd.to_numeric(frame.get("probability"), errors="coerce")
    return frame[
        (ev >= float(q.get("min_ev", 1)))
        & (edge >= float(q.get("min_edge", 999)))
        & (prob >= float(q.get("min_prob", 1)))
    ].copy()


def _passes_final_test(stats: dict, min_n=100):
    return (
        int(stats.get("n", 0) or 0) >= int(min_n)
        and stats.get("roi") is not None
        and float(stats["roi"]) > 0
        and stats.get("lcb") is not None
        and float(stats["lcb"]) > 0
        and (stats.get("avg_clv") is None or float(stats["avg_clv"]) >= 0)
    )


def validate_policy_against_backtest(
    bets_path="reports/backtest_bets.csv",
    policy_path="reports/production_policy.json",
    out_path="reports/policy_validation.json",
):
    """Evaluate tuning-selected candidates exactly once on the untouched final block.

    This function is the only automatic promotion path from candidate to enabled.
    Thresholds were selected on earlier data by ``derive_production_policy``;
    this step must not modify those thresholds based on final-test outcomes.
    """
    p = load_policy(policy_path)
    bp = Path(bets_path)
    out = {
        "policy_version": p.get("version"),
        "deployment_mode": "paper",
        "markets": {},
        "blocked_weeks": (p.get("regime_filters") or {}).get("blocked_weeks", []),
        "status": "UNPROVEN",
        "methodology": "candidate thresholds fixed on development+tuning data; untouched final chronological block used only for accept/reject",
    }
    if not bp.exists():
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
        return out
    try:
        df = pd.read_csv(bp, low_memory=False)
    except pd.errors.EmptyDataError:
        # pandas writes a truly empty DataFrame as a zero-column CSV. Treat that
        # as absence of evidence and fail closed rather than crashing CI.
        df = pd.DataFrame()
    if df.empty:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(out, indent=2))
        return out

    hold, seasons = _final_test_frame(df, p)
    out["untouched_test_seasons"] = seasons
    blocked = {int(x) for x in out["blocked_weeks"]}
    if "week" in hold.columns and blocked:
        wk = pd.to_numeric(hold["week"], errors="coerce").fillna(-1).astype(int)
        hold = hold[~wk.isin(blocked)].copy()

    validated = []
    all_final_profit = []
    for market, cfg in (p.get("markets") or {}).items():
        m = hold[hold.market.astype(str).str.lower() == market].copy() if "market" in hold.columns else hold.iloc[0:0]
        candidate = bool(cfg.get("candidate", False))
        if not candidate:
            cfg["enabled"] = False
            if cfg.get("evidence_tier") not in {"FINAL_TEST_VALIDATED"}:
                cfg["evidence_tier"] = "UNVALIDATED"
            out["markets"][market] = {
                "candidate": False,
                "enabled": False,
                "bets": 0,
                "roi": None,
                "roi_ci_95": [None, None],
                "avg_clv": None,
                "reason": "did not pass development+tuning candidate gate",
            }
            continue

        selected = _select_by_threshold(m, cfg)
        st = _profit_stats(selected)
        passed = _passes_final_test(st, 100)
        cfg["enabled"] = bool(passed)
        cfg["evidence_tier"] = "FINAL_TEST_VALIDATED" if passed else "FINAL_TEST_FAILED"
        if passed:
            validated.append(market)
            all_final_profit.extend(pd.to_numeric(selected["profit"], errors="coerce").dropna().tolist())
        out["markets"][market] = {
            "candidate": True,
            "enabled": bool(passed),
            "bets": int(st.get("n", 0) or 0),
            "roi": st.get("roi"),
            "roi_ci_95": [st.get("lcb"), st.get("ucb")],
            "avg_clv": st.get("avg_clv"),
            "untouched_test_seasons": seasons,
            "evidence_tier": cfg["evidence_tier"],
            "pass_rule": ">=100 bets, ROI>0, week-cluster bootstrap 95% lower bound>0, non-negative CLV when available",
        }

    arr = np.asarray(all_final_profit, float)
    out["overall_validated_policy"] = {
        "bets": int(len(arr)),
        "units": float(arr.sum()) if len(arr) else 0.0,
        "roi": float(arr.mean()) if len(arr) else None,
    }
    out["validated_markets_in_final_test"] = validated
    out["status"] = "FORWARD_PAPER_READY" if validated else "RESEARCH_ONLY"

    # Promotion happens only after the untouched test. Real-money mode remains
    # impossible here; this only permits forward PAPER signals.
    p["validated_markets"] = validated
    p["deployment_mode"] = "paper"
    p["final_test_status"] = out["status"]
    Path(policy_path).parent.mkdir(parents=True, exist_ok=True)
    Path(policy_path).write_text(json.dumps(p, indent=2))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    return out


def build_upgrade_audit(meta_path="outputs/system_health.json", backtest_path="reports/backtest_summary.json", policy_validation_path="reports/policy_validation.json", out_path="reports/v7_audit.json"):
    def read(path):
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return {}

    health, bt, pv = read(meta_path), read(backtest_path), read(policy_validation_path)
    comp = health.get("components") or {}
    findings = []
    if comp.get("market_intelligence", 0) < 90:
        findings.append("Multi-book consensus remains an external-data limitation unless a second independent live book source is configured.")
    if comp.get("live_monitoring", 0) < 90:
        findings.append("Live monitoring cannot be considered mature until enough forward snapshots accumulate for drift and CLV stability.")
    if (bt.get("overall") or {}).get("roi", 0) <= 0:
        findings.append("Raw historical signal set is not profitable overall; production gates must remain selective and paper-only.")
    if (bt.get("advanced_features") or {}).get("dynamic_coverage", 0) < .8:
        findings.append("Historical dynamic advanced-feature coverage is below target; identity matching/source coverage needs verification.")
    if pv.get("status") != "FORWARD_PAPER_READY":
        findings.append("No candidate market passed the untouched final-test release gate; live quant selections must remain disabled for failed markets.")

    report = {
        "engineering_readiness": health.get("system_health_score"),
        "betting_proof_status": health.get("betting_proof_status"),
        "policy_validation": pv,
        "raw_backtest_roi": (bt.get("overall") or {}).get("roi"),
        "raw_backtest_bets": (bt.get("overall") or {}).get("bets"),
        "findings": findings,
        "release_gate": {
            "real_money_allowed": False,
            "why": "A high-quality research stack can be complete before a market edge is statistically proven. Real-money promotion requires positive forward evidence, not a software score.",
            "requirements": [
                "positive untouched historical test and forward-paper ROI with uncertainty bounds",
                "positive CLV across a meaningful sample",
                "stable probability calibration",
                "multi-book or independently verified market prices",
                "no unresolved data-quality alerts",
            ],
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2))
    return report
