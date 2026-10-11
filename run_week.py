#!/usr/bin/env python3
import argparse, json, shutil
from pathlib import Path

import harbin.pipeline as pipeline
from harbin.grading_audit import grade_prediction_history as audited_grade_prediction_history
from harbin.portfolio_audit import write_portfolio_outputs
from harbin.decision_ledger import append_portfolio_decisions
from harbin.monitoring import write_live_monitoring
from harbin.model_card import write_model_card
from harbin.contracts import write_data_quality
from harbin.release_gate import write_release_gate
from harbin.reporting import write_reporting_bundle

PLATFORM_VERSION="7.4.0"

# Stage 7 makes the forward evidence path portfolio-aware without changing the core
# score model. The pipeline's live grading hook is replaced before this run starts.
pipeline.grade_prediction_history = audited_grade_prediction_history

p=argparse.ArgumentParser(description="Run Harbin Sports Analytics CFB model")
p.add_argument("--season",type=int)
p.add_argument("--week",type=int)
p.add_argument("--history-start",type=int)
a=p.parse_args()
pred,meta=pipeline.run_week(a.season,a.week,a.history_start)
meta["platform_version"]=PLATFORM_VERSION
# Show exactly which FBS-involved matchups received model forecasts.
scope_counts = pred["fbs_matchup_scope"].value_counts().to_dict() if "fbs_matchup_scope" in pred else {}
meta["fbs_slate"] = {
    "inclusion_policy": "ALL_FBS_INVOLVED_FOOTBALL_GAMES",
    "pregame_games_projected": len(pred),
    "fbs_vs_fbs": int(scope_counts.get("FBS_VS_FBS", 0)),
    "fbs_vs_non_fbs_unvalidated": int(scope_counts.get(
        "FBS_VS_NON_FBS_UNVALIDATED", 0
    )),
    "non_fbs_opponent_betting_authorized": False,
}

# Transparent prediction intervals for downstream analysis.
sm=float(meta.get("calibrated_margin_sigma",meta.get("metrics",{}).get("margin_rmse",15)) or 15)
st=float(meta.get("calibrated_total_sigma",meta.get("metrics",{}).get("total_rmse",15)) or 15)
for z,label in ((1.2816,"80"),(1.96,"95")):
    if "model_margin_home" in pred:
        pred[f"margin_pi{label}_low"]=pred.model_margin_home-z*sm; pred[f"margin_pi{label}_high"]=pred.model_margin_home+z*sm
    if "model_total" in pred:
        pred[f"total_pi{label}_low"]=pred.model_total-z*st; pred[f"total_pi{label}_high"]=pred.model_total+z*st

# Fail closed on malformed outputs before any staking or publication decisions.
data_quality=write_data_quality(pred,meta,fail_on_error=True)
monitor=write_live_monitoring(pred,meta)
release_gate=write_release_gate(meta,monitor,data_quality)
meta["data_quality"]=data_quality; meta["release_gate"]=release_gate; meta["live_monitoring"]=monitor

# Stage 5/7 portfolio approval requires release + policy + portfolio-verified live ledger
# + executable quote provenance/freshness + concentration and drawdown controls.
pred,portfolio=write_portfolio_outputs(
    pred,
    release_gate_path="outputs/release_gate.json",
    live_bets_path="reports/live_graded_bets.csv",
)
meta["portfolio"]=portfolio
meta["decision_ledger"]=append_portfolio_decisions(pred)
write_model_card(meta,monitor,portfolio)

base=f"cfb_model_{meta['season']}_week{meta['week']}"; out=Path("outputs"); docs=Path("docs"); docs.mkdir(exist_ok=True)
pred.to_csv(out/f"{base}.csv",index=False); (out/f"{base}.json").write_text(pred.to_json(orient="records",indent=2)); (out/f"{base}_metadata.json").write_text(json.dumps(meta,indent=2))
for src,dst in (
    (out/"live_monitoring.json",docs/"live_monitoring.json"),(out/"portfolio_summary.json",docs/"portfolio_summary.json"),
    (out/"release_gate.json",docs/"release_gate.json"),(out/"data_quality.json",docs/"data_quality.json"),
    (out/f"{base}_metadata.json",docs/"metadata.json"),
):
    if src.exists(): shutil.copyfile(src,dst)

# Stage 6 publishes one reconciled audit snapshot and fails closed on cross-file drift.
audit_snapshot,publication_validation=write_reporting_bundle(pred,meta)

readme=out/"README.md"
with readme.open("a") as f:
    f.write("\n## v7.4 final audit / evidence integrity layer\n- [Portfolio card](portfolio_card.csv)\n- [Portfolio summary](portfolio_summary.json)\n- [Live monitoring](live_monitoring.json)\n- [Canonical audit snapshot](audit_snapshot.json)\n- [Run report](RUN_REPORT.md)\n- [Publication validation](publication_validation.json)\n- [Release gate](release_gate.json)\n- [Data-quality contracts](data_quality.json)\n- [Model card](MODEL_CARD.md)\n- [System audit dashboard](../docs/audit.html)\n- Forward evidence is now based on execution-ready cap-constrained portfolio decisions with strict pre-kickoff timestamps.\n- Real approved stake remains **0** unless every hard PRODUCTION gate and audited execution check passes.\n")
print(f"Harbin CFB platform v{PLATFORM_VERSION} / core v{meta['model_version']} complete: {meta['season']} Week {meta['week']} — {len(pred)} games")
print("Validation:",meta["validation"])
print("Margin MAE:",meta["metrics"].get("margin_mae"),"| Total MAE:",meta["metrics"].get("total_mae"))
print("Release weights — margin:",meta["metrics"].get("margin_release_weight"),"total:",meta["metrics"].get("total_release_weight"))
print("Calibration Brier:",meta["metrics"].get("win_brier"),"| ECE:",meta["metrics"].get("win_ece"))
print("Data contracts:",data_quality["status"],"| Release state:",release_gate["release_state"])
print("Monitoring:",monitor["live_readiness_score"],"/100 | drift stability:",monitor.get("scores",{}).get("distribution_stability"))
br=portfolio.get("bankroll_risk") or {}
print("Portfolio mode:",portfolio["mode"],"| proposed units:",portfolio["proposed_units"],"| approved units:",portfolio["approved_units"],"| unit-risk multiplier:",br.get("risk_multiplier",1.0))
print("Decision ledger appended:",meta["decision_ledger"].get("appended_rows",0))
print("Audit publication:",audit_snapshot.get("status"),"| reconciliation:",publication_validation.get("status"))
print("Open outputs/README.md for the latest result links.")

