#!/usr/bin/env python3
import argparse, json, shutil
from pathlib import Path
from harbin.pipeline import run_week
from harbin.portfolio import write_portfolio_outputs
from harbin.monitoring import write_live_monitoring
from harbin.model_card import write_model_card
from harbin.contracts import write_data_quality
from harbin.release_gate import write_release_gate

PLATFORM_VERSION="7.0.0"

p=argparse.ArgumentParser(description="Run Harbin Sports Analytics CFB model")
p.add_argument("--season",type=int)
p.add_argument("--week",type=int)
p.add_argument("--history-start",type=int)
a=p.parse_args()
pred,meta=run_week(a.season,a.week,a.history_start)
meta["platform_version"]=PLATFORM_VERSION

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

# Portfolio controls now require BOTH a historically validated policy and a PRODUCTION release gate.
pred,portfolio=write_portfolio_outputs(pred,release_gate_path="outputs/release_gate.json")
meta["portfolio"]=portfolio
write_model_card(meta,monitor,portfolio)

base=f"cfb_model_{meta['season']}_week{meta['week']}"; out=Path("outputs"); docs=Path("docs"); docs.mkdir(exist_ok=True)
pred.to_csv(out/f"{base}.csv",index=False); (out/f"{base}.json").write_text(pred.to_json(orient="records",indent=2)); (out/f"{base}_metadata.json").write_text(json.dumps(meta,indent=2))
for src,dst in (
    (out/"live_monitoring.json",docs/"live_monitoring.json"),(out/"portfolio_summary.json",docs/"portfolio_summary.json"),
    (out/"release_gate.json",docs/"release_gate.json"),(out/"data_quality.json",docs/"data_quality.json"),
    (out/f"{base}_metadata.json",docs/"metadata.json"),
):
    if src.exists(): shutil.copyfile(src,dst)
readme=out/"README.md"
with readme.open("a") as f:
    f.write("\n## v7 release / evidence layer\n- [Portfolio card](portfolio_card.csv)\n- [Portfolio summary](portfolio_summary.json)\n- [Live monitoring](live_monitoring.json)\n- [Release gate](release_gate.json)\n- [Data-quality contracts](data_quality.json)\n- [Model card](MODEL_CARD.md)\n- [System audit dashboard](../docs/audit.html)\n- Real approved stake remains **0** unless every hard PRODUCTION gate is satisfied.\n")
print(f"Harbin CFB platform v7 / core v{meta['model_version']} complete: {meta['season']} Week {meta['week']} — {len(pred)} games")
print("Validation:",meta["validation"])
print("Margin MAE:",meta["metrics"].get("margin_mae"),"| Total MAE:",meta["metrics"].get("total_mae"))
print("Release weights — margin:",meta["metrics"].get("margin_release_weight"),"total:",meta["metrics"].get("total_release_weight"))
print("Calibration Brier:",meta["metrics"].get("win_brier"),"| ECE:",meta["metrics"].get("win_ece"))
print("Data contracts:",data_quality["status"],"| Release state:",release_gate["release_state"])
print("Monitoring:",monitor["live_readiness_score"],"/100")
print("Portfolio mode:",portfolio["mode"],"| proposed units:",portfolio["proposed_units"],"| approved units:",portfolio["approved_units"])
print("Open outputs/README.md for the latest result links.")
