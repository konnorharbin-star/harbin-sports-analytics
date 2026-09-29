from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED_COLUMNS={
    "game_id","season","week","date","away_team","home_team","model_margin_home","model_total",
    "winner","calibrated_home_probability","away_ml","home_ml","market_spread_home","market_total",
}

def _finite(v):
    try: return math.isfinite(float(v))
    except Exception: return False

def _issue(code,message,severity="ERROR",count=1):
    return {"code":code,"severity":severity,"count":int(count),"message":message}

def validate_run(pred: pd.DataFrame, meta: dict | None=None) -> dict:
    """Machine-enforced contracts for every published slate.

    ERRORs are conditions under which the slate must not be treated as production quality.
    WARNINGs are surfaced but do not stop research/paper output.
    """
    meta=meta or {}; issues=[]
    missing=sorted(REQUIRED_COLUMNS-set(pred.columns))
    if missing: issues.append(_issue("schema.missing_columns",f"Missing required output columns: {', '.join(missing)}"))
    if pred.empty:
        issues.append(_issue("slate.empty","No games were produced."))
        return {"status":"FAIL","errors":1,"warnings":0,"issues":issues,"checks":{},"rows":0}

    if "game_id" in pred:
        dup=int(pred["game_id"].astype(str).duplicated().sum())
        if dup: issues.append(_issue("identity.duplicate_game_id","Duplicate game IDs in current slate.",count=dup))
    for c in ("home_team","away_team"):
        if c in pred:
            bad=int(pred[c].fillna("").astype(str).str.strip().eq("").sum())
            if bad: issues.append(_issue(f"identity.{c}",f"Blank {c} values.",count=bad))

    numeric_ranges={
        "model_margin_home":(-60,60),"model_total":(10,150),"away_score_exact":(-1,110),"home_score_exact":(-1,110),
        "calibrated_home_probability":(.001,.999),"win_probability":(.001,.999),
        "market_spread_home":(-80,80),"market_total":(10,150),"away_ml":(-100000,100000),"home_ml":(-100000,100000),
    }
    for c,(lo,hi) in numeric_ranges.items():
        if c not in pred: continue
        vals=pd.to_numeric(pred[c],errors="coerce")
        required=c in {"model_margin_home","model_total","calibrated_home_probability"}
        if required:
            bad_nonfinite=int(vals.isna().sum())
            if bad_nonfinite: issues.append(_issue(f"numeric.{c}.nonfinite",f"Non-finite required values in {c}.",count=bad_nonfinite))
        valid=vals.dropna(); bad=int(((valid<lo)|(valid>hi)).sum())
        if bad: issues.append(_issue(f"numeric.{c}.range",f"{c} outside sanity range [{lo}, {hi}].",count=bad))

    if {"home_team","away_team","winner"}.issubset(pred.columns):
        bad=int((~pred.apply(lambda r:str(r.winner) in {str(r.home_team),str(r.away_team)},axis=1)).sum())
        if bad: issues.append(_issue("prediction.invalid_winner","Winner is not one of the two teams.",count=bad))

    if "date" in pred:
        dates=pd.to_datetime(pred.date,utc=True,errors="coerce")
        bad=int(dates.isna().sum())
        if bad: issues.append(_issue("schedule.invalid_date","Invalid kickoff timestamps.",count=bad))

    market_checks={}
    for c in ("away_ml","home_ml","market_spread_home","market_total"):
        if c in pred: market_checks[c]=int(pd.to_numeric(pred[c],errors="coerce").notna().sum())
    games=max(1,len(pred)); complete=sum(
        pd.to_numeric(pred.get(c,pd.Series(index=pred.index,dtype=float)),errors="coerce").notna().astype(int)
        for c in ("away_ml","home_ml","market_spread_home","market_total")
    ) if len(pred) else pd.Series(dtype=int)
    complete_games=int((complete==4).sum()) if len(complete) else 0
    complete_rate=complete_games/games
    if complete_rate<.90:
        issues.append(_issue("market.coverage",f"Only {complete_rate:.1%} of games have complete ML/spread/total markets.","WARNING",games-complete_games))

    metrics=meta.get("metrics") or {}
    for name in ("margin","total"):
        if f"{name}_release_weight" in metrics and not _finite(metrics[f"{name}_release_weight"]):
            issues.append(_issue(f"model.{name}.release_weight","Release weight is non-finite."))
    errors=sum(i["severity"]=="ERROR" for i in issues); warnings=sum(i["severity"]=="WARNING" for i in issues)
    checks={
        "unique_game_ids": not any(i["code"]=="identity.duplicate_game_id" for i in issues),
        "required_schema": not any(i["code"]=="schema.missing_columns" for i in issues),
        "finite_core_predictions": not any(i["code"].endswith(".nonfinite") for i in issues),
        "sanity_ranges": not any(i["code"].endswith(".range") for i in issues),
        "complete_market_rate":round(complete_rate,6),
    }
    return {"status":"FAIL" if errors else "WARN" if warnings else "PASS","errors":errors,"warnings":warnings,"issues":issues,"checks":checks,"rows":int(len(pred))}

def write_data_quality(pred: pd.DataFrame, meta: dict, output_path="outputs/data_quality.json", fail_on_error=True):
    report=validate_run(pred,meta)
    p=Path(output_path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,indent=2))
    if fail_on_error and report["errors"]:
        details="; ".join(i["message"] for i in report["issues"] if i["severity"]=="ERROR")
        raise RuntimeError(f"Data contract failure: {details}")
    return report
