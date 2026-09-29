from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import SportsDataVerseClient
from .ratings import OpponentAdjustedRatings
from .models import train_models, predict_models
from .market import (
    norm_cdf,
    replica_win_probability,
    replica_ml_label,
    replica_spread_label,
    replica_total_label,
    no_vig,
    fair_american,
    roi,
)
from .render import render_html, render_png

MODEL_VERSION = "3.0.0"


def _is_missing(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def build_predictions(games, frame, bundle):
    if frame.empty:
        return pd.DataFrame()

    margins, totals = predict_models(bundle, frame)
    by_id = {str(g.game_id): g for g in games}
    rows = []

    for i, r in frame.reset_index(drop=True).iterrows():
        g = by_id[str(r.game_id)]
        m, t = float(margins[i]), float(totals[i])
        hp, ap = (t + m) / 2, (t - m) / 2
        winner = g.home_team if m >= 0 else g.away_team

        replica_p = replica_win_probability(m)
        sigma_m = max(6.0, float(bundle["margin_sigma"]))
        sigma_t = max(6.0, float(bundle["total_sigma"]))
        calibrated_home_p = norm_cdf(m / sigma_m)
        calibrated_winner_p = calibrated_home_p if m >= 0 else 1 - calibrated_home_p

        ml_odds = g.home_ml if m >= 0 else g.away_ml
        ml_badge = ""
        ml_edge = np.nan
        ml_roi = np.nan
        fair = fair_american(calibrated_winner_p)
        if ml_odds is not None:
            ml_badge, ml_edge = replica_ml_label(replica_p, ml_odds)
            ml_roi = roi(calibrated_winner_p, ml_odds)

        spread_team = np.nan
        spread_line = np.nan
        spread_badge = ""
        spread_edge = np.nan
        cover_p = np.nan
        fair_spread_home = -m
        if g.home_spread is not None:
            spread_badge, edge_home = replica_spread_label(m, g.home_spread)
            spread_edge = abs(edge_home)
            if edge_home >= 0:
                spread_team, spread_line = g.home_team, float(g.home_spread)
            else:
                spread_team, spread_line = g.away_team, -float(g.home_spread)
            cover_p = norm_cdf(abs(edge_home) / sigma_m)

        total_dir = np.nan
        total_badge = ""
        total_edge = np.nan
        total_prob = np.nan
        if g.market_total is not None:
            total_badge, edge_total = replica_total_label(t, g.market_total)
            total_edge = abs(edge_total)
            total_dir = "O" if edge_total >= 0 else "U"
            total_prob = norm_cdf(abs(edge_total) / sigma_t)

        nv_winner = np.nan
        quant_ml_side = np.nan
        quant_ml_edge = np.nan
        quant_ml_roi = np.nan
        if g.away_ml is not None and g.home_ml is not None:
            pa, ph = no_vig(g.away_ml, g.home_ml)
            edge_h = calibrated_home_p - ph
            edge_a = (1 - calibrated_home_p) - pa
            if edge_h >= edge_a:
                quant_ml_side = g.home_team
                quant_ml_edge = 100 * edge_h
                quant_ml_roi = roi(calibrated_home_p, g.home_ml)
            else:
                quant_ml_side = g.away_team
                quant_ml_edge = 100 * edge_a
                quant_ml_roi = roi(1 - calibrated_home_p, g.away_ml)
            nv_winner = ph if m >= 0 else pa

        market_available = any(
            x is not None for x in (g.home_ml, g.away_ml, g.home_spread, g.market_total)
        )

        rows.append({
            "game_id": g.game_id, "season": g.season, "week": g.week, "date": g.date,
            "away_team": g.away_team, "home_team": g.home_team,
            "away_score": int(round(ap)), "home_score": int(round(hp)),
            "away_score_exact": ap, "home_score_exact": hp,
            "model_margin_home": m, "model_total": t, "fair_spread_home": fair_spread_home,
            "fair_total": t, "winner": winner, "win_probability": replica_p,
            "win_pct": int(round(100 * replica_p)), "calibrated_home_probability": calibrated_home_p,
            "calibrated_winner_probability": calibrated_winner_p, "provider": g.provider,
            "market_available": market_available, "away_ml": g.away_ml, "home_ml": g.home_ml,
            "ml_team": winner, "ml_odds": ml_odds, "ml_badge": ml_badge,
            "ml_edge_pp": ml_edge, "ml_est_roi": ml_roi, "ml_fair_odds": fair,
            "no_vig_market_prob_winner": nv_winner, "quant_best_ml_side": quant_ml_side,
            "quant_best_ml_edge_pp": quant_ml_edge, "quant_best_ml_roi": quant_ml_roi,
            "market_spread_home": g.home_spread, "spread_team": spread_team,
            "spread_line": spread_line, "spread_badge": spread_badge,
            "spread_edge_pts": spread_edge, "cover_probability": cover_p,
            "market_total": g.market_total, "total_dir": total_dir, "proj_total": int(round(t)),
            "total_badge": total_badge, "total_edge_pts": total_edge,
            "total_probability": total_prob, "baseline_margin": float(r.baseline_margin),
            "baseline_total": float(r.baseline_total), "elo_diff_home": float(r.elo_diff_home),
            "net_eff_diff_home": float(r.net_eff_diff_home),
        })

    return pd.DataFrame(rows).sort_values(["date", "game_id"]).reset_index(drop=True)


def _coverage(pred: pd.DataFrame) -> dict:
    if pred.empty:
        return {"games": 0, "any_market": 0, "moneyline": 0, "spread": 0, "total": 0}
    return {
        "games": int(len(pred)),
        "any_market": int(pred["market_available"].fillna(False).astype(bool).sum()),
        "moneyline": int((pred["away_ml"].notna() & pred["home_ml"].notna()).sum()),
        "spread": int(pred["market_spread_home"].notna().sum()),
        "total": int(pred["market_total"].notna().sum()),
    }


def _market_status(coverage: dict, source: str) -> str:
    games = coverage["games"]
    if games == 0:
        return "No upcoming games were returned for this season/week."
    if coverage["any_market"] == 0:
        return "PROJECTION-ONLY MODE — no live sportsbook market data was available. No ML/spread/total betting signals are shown."
    if coverage["any_market"] < games:
        return f"PARTIAL MARKET DATA — {coverage['any_market']}/{games} games have at least one market from {source or 'available sources'}."
    return f"LIVE MARKET DATA — all {games} games have market data from {source or 'available sources'}."


def _display_time(stamp_dt: datetime) -> str:
    central = stamp_dt.astimezone(ZoneInfo("America/Chicago"))
    return central.strftime("%b %-d, %Y · %-I:%M %p CT")


def _prediction_signature(pred: pd.DataFrame) -> str:
    if pred.empty:
        return ""
    cols = ["game_id", "model_margin_home", "model_total", "away_ml", "home_ml", "market_spread_home", "market_total", "provider"]
    temp = pred[[c for c in cols if c in pred.columns]].copy()
    for c in temp.columns:
        if pd.api.types.is_numeric_dtype(temp[c]):
            temp[c] = temp[c].round(6)
    return temp.to_json(orient="records")


def _append_snapshot_if_changed(pred: pd.DataFrame, hist_path: Path, stamp: str) -> bool:
    if pred.empty:
        return False
    sig = _prediction_signature(pred)
    sig_path = hist_path.with_suffix(".signature")
    previous = sig_path.read_text() if sig_path.exists() else None
    if previous == sig:
        return False
    snap = pred.copy()
    snap.insert(0, "snapshot_at", stamp)
    old = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    pd.concat([old, snap], ignore_index=True).to_csv(hist_path, index=False)
    sig_path.write_text(sig)
    return True


def _write_output_readme(out: Path, base: str, meta: dict, pages: int):
    pngs = "\n".join(f"- [Page {i}]({base}_page{i}.png)" for i in range(1, pages + 1))
    text = f"""# Latest CFB model output

**Model:** v{meta['model_version']}  
**Season / Week:** {meta['season']} / {meta['week']}  
**Updated:** {meta['updated_at_ct']}  
**Market status:** {meta['market_status']}  
**Odds source:** {meta['odds_source']}

## Open these
- [Interactive Cooper-style table]({base}.html)
- [Full CSV]({base}.csv)
- [Full JSON]({base}.json)
- [Metadata / diagnostics]({base}_metadata.json)
{pngs}

If a market says **NO LINE**, the model did not receive a verified sportsbook quote for that market. It does not invent one.
"""
    (out / "README.md").write_text(text)


def run_week(season=None, week=None, history_start=None, root="."):
    root = Path(root)
    out, docs, hist = root / "outputs", root / "docs", root / "history"
    for p in (out, docs, hist):
        p.mkdir(parents=True, exist_ok=True)

    client = SportsDataVerseClient()
    if season is None or week is None:
        ds, dw = client.detect()
        season = season or ds
        week = week or dw

    history_start = history_start or max(2018, season - 5)
    history = client.history(history_start, season, week)
    ratings = OpponentAdjustedRatings()
    train_df = ratings.training_frame(history)
    bundle = train_models(train_df)

    games = [g for g in client.week(season, week) if not g.completed]
    frame = ratings.upcoming_frame(games)
    pred = build_predictions(games, frame, bundle)

    now = datetime.now(timezone.utc)
    stamp = now.isoformat()
    display_time = _display_time(now)
    base = f"cfb_model_{season}_week{week}"
    coverage = _coverage(pred)
    status = _market_status(coverage, client.odds_source)

    meta = {
        "model_version": MODEL_VERSION, "season": int(season), "week": int(week),
        "history_start": int(history_start), "historical_games": int(len(history)),
        "training_rows": int(len(train_df)), "upcoming_games": int(len(games)),
        "validation": bundle.get("validation"), "metrics": dict(bundle["metrics"]),
        "model_selection": {"margin_blend_weight": float(bundle["margin_weight"]), "total_blend_weight": float(bundle["total_weight"])},
        "replica_margin_sigma": 16.41, "calibrated_margin_sigma": float(bundle["margin_sigma"]),
        "calibrated_total_sigma": float(bundle["total_sigma"]), "generated_at": stamp,
        "updated_at_ct": display_time, "schedule_source": "sportsdataverse/cfbfastR-data",
        "odds_source": client.odds_source, "odds_errors": client.odds_errors,
        "odds_rows_matched": int(client.odds_rows_matched), "market_coverage": coverage,
        "market_status": status,
    }

    pred.to_csv(out / f"{base}.csv", index=False)
    (out / f"{base}.json").write_text(pred.to_json(orient="records", indent=2))
    (out / f"{base}_metadata.json").write_text(json.dumps(meta, indent=2))

    pages = max(1, math.ceil(len(pred) / 14))
    render_html(pred, out / f"{base}.html", week, display_time, status)
    for p in range(1, pages + 1):
        render_png(pred, out / f"{base}_page{p}.png", p, week, display_time, status)

    render_html(pred, docs / "index.html", week, display_time, status)
    (docs / "latest.csv").write_text(pred.to_csv(index=False))
    (docs / "latest.json").write_text(pred.to_json(orient="records", indent=2))
    (docs / "metadata.json").write_text(json.dumps(meta, indent=2))
    (docs / ".nojekyll").write_text("")

    snapshot_changed = _append_snapshot_if_changed(pred, hist / "prediction_snapshots_v3.csv", stamp)
    meta["snapshot_appended"] = snapshot_changed
    (out / f"{base}_metadata.json").write_text(json.dumps(meta, indent=2))
    (docs / "metadata.json").write_text(json.dumps(meta, indent=2))

    mp = hist / "run_metadata_v3.jsonl"
    with mp.open("a") as f:
        f.write(json.dumps(meta, separators=(",", ":")) + "\n")

    _write_output_readme(out, base, meta, pages)
    return pred, meta
