from __future__ import annotations

"""Stage 7 audit wrapper for historical betting evidence.

The legacy archive remains useful for research diagnostics, but only rows with explicit,
non-null archived opening fields qualify as promotion evidence.  This wrapper preserves
all historical rows while marking their entry provenance market-by-market.
"""

import json
import math
from pathlib import Path

import pandas as pd

from . import backtest as _bt
from . import backtest_runtime as _rt


def _numeric_present(series) -> int:
    return int(pd.to_numeric(series, errors="coerce").notna().sum()) if series is not None else 0


class AuditedArchiveMarketStore(_rt.CanonicalArchiveMarketStore):
    def quote(self, game):
        q = super().quote(game)
        if not q:
            return q
        q.update({
            "open_moneyline_verified": False,
            "open_spread_verified": False,
            "open_total_verified": False,
        })
        rows = self.by_id.get(_rt.canonical_game_id(game.game_id))
        if rows is None or rows.empty or "book" not in rows.columns or "market_type" not in rows.columns:
            return q
        br = rows[rows["book"].astype(str) == str(q.get("book"))].copy()
        if br.empty:
            return q
        br["_market"] = br["market_type"].astype(str).str.lower().str.replace("-", "_", regex=False).str.replace(" ", "_", regex=False)

        ml = br[br["_market"].str.contains("money", na=False)]
        sp = br[br["_market"].str.contains("spread", na=False)]
        to = br[br["_market"].str.contains("total", na=False)]
        if "opening_odds" in br.columns:
            q["open_moneyline_verified"] = _numeric_present(ml["opening_odds"]) >= 2
        if {"opening_lines", "opening_odds"}.issubset(br.columns):
            q["open_spread_verified"] = int((pd.to_numeric(sp["opening_lines"], errors="coerce").notna() & pd.to_numeric(sp["opening_odds"], errors="coerce").notna()).sum()) >= 2
            q["open_total_verified"] = int((pd.to_numeric(to["opening_lines"], errors="coerce").notna() & pd.to_numeric(to["opening_odds"], errors="coerce").notna()).sum()) >= 2
        return q


def _audited_market_bets(game, margin, total, p_home, sigma_m, sigma_t, q):
    bets = _rt._rigorous_market_bets(game, margin, total, p_home, sigma_m, sigma_t, q)
    for b in bets:
        market = str(b.get("market") or "").lower()
        verified = bool(q.get(f"open_{market}_verified", False))
        if market in {"spread", "total"}:
            try: verified = verified and math.isfinite(float(b.get("odds")))
            except Exception: verified = False
        b["entry_quote_verified"] = bool(verified)
        b["entry_quote_source"] = "archive_opening_fields" if verified else "archive_final_or_unverified_fallback"
    return bets


def run_backtest(start_season=2023, end_season=2025, history_start=2018, reports_dir="reports"):
    # The original backtest resolves these globals when each run executes, so patching
    # here changes only the market archive/evidence layer, not the fair-score model.
    _bt.ArchiveMarketStore = AuditedArchiveMarketStore
    _bt._market_bets = _audited_market_bets
    _bt._group_summary = _rt._rigorous_group_summary
    bdf, summary = _rt.run_backtest(start_season, end_season, history_start, reports_dir)
    verified = int(bdf.get("entry_quote_verified", pd.Series(False, index=bdf.index)).fillna(False).astype(bool).sum()) if len(bdf) else 0
    summary["quote_integrity"] = {
        "all_archive_bets": int(len(bdf)),
        "verified_opening_entry_bets": verified,
        "unverified_or_final_fallback_bets": int(max(0, len(bdf) - verified)),
        "verified_opening_entry_rate": float(verified / len(bdf)) if len(bdf) else 0.0,
        "promotion_rule": "only entry_quote_verified=true rows may contribute to production evidence or policy calibration",
    }
    summary.setdefault("methodology", {})["promotion_entry"] = "explicit non-null archived opening fields for the selected sportsbook/market; final-quote fallbacks are diagnostics only"
    reports = Path(reports_dir); bdf.to_csv(reports / "backtest_bets.csv", index=False)
    (reports / "backtest_summary.json").write_text(json.dumps(summary, indent=2))
    return bdf, summary
