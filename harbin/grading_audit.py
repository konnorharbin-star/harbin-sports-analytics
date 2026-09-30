from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import grading as _g

_GRADED_COLUMNS = [
    "game_id","season","week","away_team","home_team","projected_margin_home","actual_margin_home",
    "projected_total","actual_total","quant_signal","quant_market","quant_side","quant_book","quant_price",
    "quant_odds","portfolio_candidate_units","portfolio_action","result","profit","clv_proxy","execution_clv",
    "clv_source","close_market_book_count","close_snapshot_at","decision_at","kickoff",
]


def _strict_pre_kickoff(df: pd.DataFrame, ts_col: str, kickoff_col: str):
    if df.empty or ts_col not in df.columns or kickoff_col not in df.columns:
        return df.iloc[0:0].copy(), int(len(df))
    x = df.copy()
    x["_decision_ts"] = pd.to_datetime(x[ts_col], utc=True, errors="coerce")
    x["_kick"] = pd.to_datetime(x[kickoff_col], utc=True, errors="coerce")
    ok = x["_decision_ts"].notna() & x["_kick"].notna() & (x["_decision_ts"] < x["_kick"])
    return x[ok].copy(), int((~ok).sum())


def _load_source(history_dir):
    hdir = Path(history_dir)
    decisions = hdir / "portfolio_decisions_v1.csv"
    if decisions.exists() and decisions.stat().st_size > 0:
        try:
            df = pd.read_csv(decisions, low_memory=False)
        except Exception:
            df = pd.DataFrame()
        if not df.empty:
            strict, excluded = _strict_pre_kickoff(df, "decision_at", "date")
            units = pd.to_numeric(strict.get("portfolio_candidate_units", 0), errors="coerce").fillna(0.0)
            action = strict.get("portfolio_action", pd.Series("PASS", index=strict.index)).fillna("PASS").astype(str).str.upper()
            execution = strict.get("execution_ready", pd.Series(False, index=strict.index)).fillna(False).astype(bool)
            eligible = strict[(units > 0) & action.isin({"PAPER","SHADOW","BET"}) & execution].copy()
            return eligible, True, "portfolio_decisions_v1", excluded

    legacy = hdir / "prediction_snapshots_v4.csv"
    if not legacy.exists() or legacy.stat().st_size <= 0:
        return pd.DataFrame(), False, "none", 0
    try: df = pd.read_csv(legacy, low_memory=False)
    except Exception: return pd.DataFrame(), False, "none", 0
    strict, excluded = _strict_pre_kickoff(df, "snapshot_at", "date")
    signal = strict.get("quant_signal", pd.Series("PASS", index=strict.index)).fillna("PASS").astype(str).str.upper()
    eligible = strict[(signal != "PASS") & strict.get("quant_market", pd.Series(index=strict.index, dtype=object)).notna()].copy()
    return eligible, False, "legacy_raw_quant_signals", excluded


def grade_prediction_history(client, history_dir="history", reports_dir="reports"):
    """Grade only strict pre-kickoff decisions; production evidence requires portfolio ledger."""
    reports = Path(reports_dir); reports.mkdir(parents=True, exist_ok=True); hdir = Path(history_dir)
    entries, portfolio_verified, evidence_source, timing_excluded = _load_source(hdir)
    markets_path = hdir / "market_snapshots.csv"
    markets = pd.read_csv(markets_path, low_memory=False) if markets_path.exists() and markets_path.stat().st_size > 0 else pd.DataFrame()
    method = "first execution-ready cap-constrained portfolio decision strictly before kickoff; close is latest timestamp-valid market snapshot strictly before kickoff"

    if entries.empty:
        empty = pd.DataFrame(columns=_GRADED_COLUMNS); empty.to_csv(reports/"live_graded_predictions.csv", index=False); empty.to_csv(reports/"live_graded_bets.csv", index=False)
        report = {"graded_games":0, **_g._summary(empty), "portfolio_verified":bool(portfolio_verified), "evidence_source":evidence_source, "eligible_decisions":0, "timing_excluded_rows":timing_excluded, "status":"live/shadow evidence only; not historical backtest evidence", "clv_method":method}
        (reports/"live_performance.json").write_text(json.dumps(report, indent=2)); return report

    finals = {}
    for season in sorted({int(x) for x in pd.to_numeric(entries.get("season"), errors="coerce").dropna().unique()}):
        try: sf = client.season_frame(season)
        except Exception: continue
        done = sf[sf["completed"].astype(str).str.lower().isin({"true","1","t","yes"})]
        for _, r in done.iterrows():
            try: finals[str(r.game_id)] = (float(r.home_points), float(r.away_points))
            except Exception: pass

    rows = []
    ts_col = "decision_at" if portfolio_verified else "snapshot_at"
    for gid, group in entries.groupby(entries.game_id.astype(str), sort=False):
        if gid not in finals: continue
        group = group.sort_values("_decision_ts" if "_decision_ts" in group.columns else ts_col, kind="mergesort")
        entry = group.iloc[0]; kickoff = pd.to_datetime(entry.get("date"), utc=True, errors="coerce")
        odds = _g._safe(entry.get("quant_odds")); market = str(entry.get("quant_market") or "")
        if not math.isfinite(odds) or odds == 0: continue
        if market in {"spread","total"} and not math.isfinite(_g._safe(entry.get("quant_price"))): continue
        if not str(entry.get("quant_book") or "").strip(): continue

        hpnt, apnt = finals[gid]; actual_margin = hpnt - apnt; actual_total = hpnt + apnt
        result = _g._grade_row(entry, actual_margin, actual_total); profit = _g._profit(result, odds, market)
        close = _g._latest_pre_kickoff_market(markets, gid, kickoff)
        clv, clv_source = _g._clv_from_market_snapshot(entry, close); execution_clv = _g._execution_clv_from_market_snapshot(entry, close)
        rows.append({
            "game_id":gid,"season":entry.get("season"),"week":entry.get("week"),"away_team":entry.get("away_team"),"home_team":entry.get("home_team"),
            "projected_margin_home":entry.get("model_margin_home"),"actual_margin_home":actual_margin,"projected_total":entry.get("model_total"),"actual_total":actual_total,
            "quant_signal":entry.get("quant_signal"),"quant_market":market,"quant_side":entry.get("quant_side"),"quant_book":entry.get("quant_book"),"quant_price":entry.get("quant_price"),"quant_odds":odds,
            "portfolio_candidate_units":entry.get("portfolio_candidate_units", np.nan),"portfolio_action":entry.get("portfolio_action"),"result":result,"profit":profit,"clv_proxy":clv,"execution_clv":execution_clv,"clv_source":clv_source,
            "close_market_book_count":close.get("market_book_count", np.nan) if close is not None else np.nan,"close_snapshot_at":close.get("captured_at") if close is not None else None,
            "decision_at":entry.get(ts_col),"kickoff":entry.get("date"),
        })

    df = pd.DataFrame(rows, columns=_GRADED_COLUMNS); df.to_csv(reports/"live_graded_predictions.csv", index=False)
    bets = df[df.result.notna()].copy() if len(df) else pd.DataFrame(columns=_GRADED_COLUMNS); bets.to_csv(reports/"live_graded_bets.csv", index=False)
    overall = _g._summary(bets); by_market = {str(k):_g._summary(v) for k,v in bets.groupby("quant_market")} if len(bets) else {}; by_signal = {str(k):_g._summary(v) for k,v in bets.groupby("quant_signal")} if len(bets) else {}; by_season = {str(k):_g._summary(v) for k,v in bets.groupby("season")} if len(bets) else {}
    verified_close = int(pd.to_numeric(bets.get("clv_proxy"), errors="coerce").notna().sum()) if len(bets) else 0
    report = {"graded_games":int(len(df)), **overall, "by_market":by_market,"by_signal":by_signal,"by_season":by_season,"clv_method":method,"verified_close_clv_samples":verified_close,"portfolio_verified":bool(portfolio_verified),"evidence_source":evidence_source,"eligible_decisions":int(len(entries)),"timing_excluded_rows":int(timing_excluded),"status":"live/shadow evidence only; not historical backtest evidence"}
    (reports/"live_performance.json").write_text(json.dumps(report, indent=2)); return report
