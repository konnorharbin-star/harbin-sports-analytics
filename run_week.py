#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
from harbin.pipeline import run_week
from harbin.portfolio import write_portfolio_outputs
from harbin.monitoring import write_live_monitoring
from harbin.model_card import write_model_card

p=argparse.ArgumentParser(description="Run Harbin Sports Analytics CFB model")
p.add_argument("--season",type=int)
p.add_argument("--week",type=int)
p.add_argument("--history-start",type=int)
a=p.parse_args()
pred,meta=run_week(a.season,a.week,a.history_start)
# Add transparent uncertainty bands to the machine-readable outputs.
sm=float(meta.get("calibrated_margin_sigma",meta.get("metrics",{}).get("margin_rmse",15)) or 15); st=float(meta.get("calibrated_total_sigma",meta.get("metrics",{}).get("total_rmse",15)) or 15)
for z,label in ((1.2816,"80"),(1.96,"95")):
    if "model_margin_home" in pred: pred[f"margin_pi{label}_low"]=pred.model_margin_home-z*sm; pred[f"margin_pi{label}_high"]=pred.model_margin_home+z*sm
    if "model_total" in pred: pred[f"total_pi{label}_low"]=pred.model_total-z*st; pred[f"total_pi{label}_high"]=pred.model_total+z*st
pred,portfolio=write_portfolio_outputs(pred)
monitor=write_live_monitoring(pred,meta)
write_model_card(meta,monitor,portfolio)
base=f"cfb_model_{meta['season']}_week{meta['week']}"; out=Path("outputs")
pred.to_csv(out/f"{base}.csv",index=False); (out/f"{base}.json").write_text(pred.to_json(orient="records",indent=2))
meta["platform_version"]="5.0.0"; meta["portfolio"]=portfolio; meta["live_monitoring"]=monitor; (out/f"{base}_metadata.json").write_text(json.dumps(meta,indent=2))
readme=out/"README.md"
with readme.open("a") as f:
    f.write("\n## v5 risk / proof layer\n- [Portfolio card](portfolio_card.csv)\n- [Portfolio summary](portfolio_summary.json)\n- [Live monitoring](live_monitoring.json)\n- [Model card](MODEL_CARD.md)\n- Policy is backtest-calibrated when `reports/production_policy.json` exists. Until evidence is validated, portfolio mode remains PAPER.\n")
print(f"Harbin CFB platform v5 / core v{meta['model_version']} complete: {meta['season']} Week {meta['week']} — {len(pred)} games")
print("Validation:",meta["validation"])
print("Margin MAE:",meta["metrics"].get("margin_mae"),"| Total MAE:",meta["metrics"].get("total_mae"))
print("Calibration Brier:",meta["metrics"].get("win_brier"),"| ECE:",meta["metrics"].get("win_ece"))
print("Monitoring:",monitor["live_readiness_score"],"/100")
print("Portfolio mode:",portfolio["mode"],"| proposed units:",portfolio["proposed_units"],"| approved units:",portfolio["approved_units"])
print("Open outputs/README.md for the latest result links.")
