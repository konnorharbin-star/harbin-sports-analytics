from __future__ import annotations

import json
from pathlib import Path


def _read(path):
    p=Path(path)
    if not p.exists(): return {}
    try: return json.loads(p.read_text())
    except Exception: return {}
def _check(name,passed,value,requirement,detail=""):
    return {"name":name,"passed":bool(passed),"value":value,"requirement":requirement,"detail":detail}
def build_release_gate(meta: dict, monitor: dict | None=None, data_quality: dict | None=None,
                       evidence_path="reports/evidence_report.json", live_path="reports/live_performance.json") -> dict:
    """Hard release states: RESEARCH -> PAPER -> SHADOW -> PRODUCTION.

    No weighted average can override a failed hard gate. Historical profitability and
    live evidence are deliberately separate from engineering readiness.
    """
    monitor=monitor or {}; data_quality=data_quality or {}; metrics=meta.get("metrics") or {}
    market=meta.get("market_coverage") or {}; games=max(1,int(market.get("games",0) or 0))
    adv=meta.get("advanced_features") or {}; ctx=meta.get("current_context") or {}; intel=meta.get("market_intelligence") or {}
    evidence=_read(evidence_path); live=_read(live_path)
    eo=evidence.get("overall") or {}; by_market=evidence.get("by_market") or {}; by_season=evidence.get("by_season") or {}
    ci=eo.get("roi_ci_95") or [None,None]
    positive_markets=sum(1 for v in by_market.values() if (v.get("roi") is not None and float(v.get("roi",0))>0 and v.get("avg_clv") is not None and float(v.get("avg_clv",0))>0))
    positive_seasons=sum(1 for v in by_season.values() if (v.get("roi") is not None and float(v.get("roi",0))>0 and v.get("avg_clv") is not None and float(v.get("avg_clv",0))>0))

    complete=min(int(market.get("moneyline",0) or 0),int(market.get("spread",0) or 0),int(market.get("total",0) or 0))/games
    broad_adv_cov=float(adv.get("live_coverage",adv.get("coverage",0)) or 0)
    # Compatibility for historical fixtures that predate the explicit dynamic field;
    # current model metadata always writes dynamic_coverage and therefore cannot hide 0%.
    dynamic_adv_cov=float(adv.get("dynamic_coverage",broad_adv_cov) or 0)
    context_cov=float(ctx.get("coverage",1.0 if ctx.get("sources") else 0.0) or 0)
    multi=float(intel.get("multi_book_coverage",0) or 0)
    brier=metrics.get("win_brier"); ece=metrics.get("win_ece")
    margin_guard=bool(metrics.get("margin_release_guard_passed",metrics.get("margin_mae",999)<=metrics.get("margin_baseline_mae",-999)))
    total_weight=float(metrics.get("total_release_weight",metrics.get("total_blend_weight",0)) or 0)
    total_guard=bool(metrics.get("total_release_guard_passed",False) or total_weight==0)
    monitor_score=float(monitor.get("live_readiness_score",0) or 0)

    hist_n=int(eo.get("bets",0) or 0); hist_clv=eo.get("avg_clv")
    historical_ready=(hist_n>=1000 and ci[0] is not None and float(ci[0])>0 and hist_clv is not None and float(hist_clv)>0 and positive_markets>=2 and positive_seasons>=2)
    live_n=int(live.get("graded_bets",live.get("bets",0)) or 0); live_roi=live.get("roi"); live_clv=live.get("avg_clv_proxy",live.get("avg_clv"))
    live_ready=(live_n>=300 and live_roi is not None and float(live_roi)>=0 and live_clv is not None and float(live_clv)>0)

    checks=[
        _check("data_contracts",data_quality.get("status")!="FAIL",data_quality.get("status","UNKNOWN"),"no ERROR-level data-contract violations"),
        _check("complete_market_coverage",complete>=.95,round(complete,4),">= 95% games with ML + spread + total"),
        _check("advanced_feature_coverage",broad_adv_cov>=.95,round(broad_adv_cov,4),">= 95% live advanced/static feature coverage"),
        _check("dynamic_advanced_feature_coverage",dynamic_adv_cov>=.90,round(dynamic_adv_cov,4),">= 90% live opponent-adjusted dynamic efficiency coverage"),
        _check("context_coverage",context_cov>=.90,round(context_cov,4),">= 90% current context coverage"),
        _check("margin_release_guard",margin_guard,metrics.get("margin_release_weight",metrics.get("margin_blend_weight")),"learned margin correction passes release guard or safely falls back"),
        _check("total_release_safety",total_guard,total_weight,"total correction passes release guard or automatically falls back to baseline",metrics.get("total_release_guard_reason","")),
        _check("probability_brier",brier is not None and float(brier)<=.18,brier,"Brier <= 0.18 on chronological release holdout"),
        _check("probability_ece",ece is not None and float(ece)<=.05,ece,"ECE <= 0.05 on chronological release holdout"),
        _check("live_monitoring",monitor_score>=90,monitor_score,"live readiness >= 90/100"),
        _check("multi_book_consensus",multi>=.75,round(multi,4),">= 75% of games covered by 2+ books","Configure THE_ODDS_API_KEY if ESPN exposes only one provider."),
        _check("historical_market_edge",historical_ready,{"bets":hist_n,"roi_ci_95":ci,"avg_clv":hist_clv,"positive_markets":positive_markets,"positive_seasons":positive_seasons},">=1000 bets, ROI 95% CI lower bound >0, positive CLV, >=2 positive markets and >=2 positive seasons"),
        _check("live_shadow_evidence",live_ready,{"graded_bets":live_n,"roi":live_roi,"avg_clv":live_clv},">=300 graded live bets, non-negative ROI and positive pre-kickoff CLV proxy"),
    ]
    engineering_names={"data_contracts","complete_market_coverage","advanced_feature_coverage","dynamic_advanced_feature_coverage","context_coverage","margin_release_guard","total_release_safety","probability_brier","probability_ece","live_monitoring"}
    engineering_ready=all(c["passed"] for c in checks if c["name"] in engineering_names)
    if not engineering_ready:
        state="RESEARCH"
    elif historical_ready and multi>=.75 and live_ready:
        state="PRODUCTION"
    elif historical_ready:
        state="SHADOW"
    else:
        state="PAPER"
    blockers=[f"{c['name']}: {c['requirement']}" for c in checks if not c["passed"]]
    next_steps=[]
    if dynamic_adv_cov<.90: next_steps.append("Restore dynamic pregame efficiency coverage; static priors do not substitute for current team efficiency.")
    if multi<.75: next_steps.append("Add a verified multi-book source; THE_ODDS_API_KEY is already supported as a GitHub secret.")
    if not historical_ready: next_steps.append("Keep paper testing until the walk-forward ROI confidence interval clears zero with positive CLV across markets and seasons.")
    if not live_ready: next_steps.append("Accumulate graded live/shadow bets and pre-kickoff line snapshots; do not infer live profitability from historical backtests.")
    return {
        "release_state":state,"production_eligible":state=="PRODUCTION","engineering_ready":engineering_ready,
        "historical_edge_ready":historical_ready,"live_evidence_ready":live_ready,"checks":checks,"blockers":blockers,
        "next_requirements":next_steps,
        "meaning":"Hard deployment gate. PRODUCTION requires engineering quality, dynamic advanced data, multi-book pricing, robust historical edge and independent live evidence; it is not a guarantee of future profit.",
    }
def write_release_gate(meta, monitor=None, data_quality=None, output_path="outputs/release_gate.json",
                       report_path="reports/release_gate.json"):
    gate=build_release_gate(meta,monitor,data_quality)
    for path in (output_path,report_path):
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(gate,indent=2))
    return gate
