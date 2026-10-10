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

from .advanced import canon_team
from . import backtest as _bt
from . import backtest_runtime as _rt
from .entry_provenance import verified_entry_mask


def _valid_opening_side(rows: pd.DataFrame, side: str, fields: tuple[str, ...]) -> bool:
    """Require every opening field for the actual selected market side."""
    selected = rows[rows["_side"] == side]
    if selected.empty or any(field not in selected.columns for field in fields):
        return False
    # The base archive quote chooses the final matching row per side.
    last = selected.iloc[-1]
    for field in fields:
        try:
            value = float(last[field])
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
        if "odds" in field and (-100.0 < value < 100.0):
            return False
    return True


def _valid_opening_total_side(rows: pd.DataFrame, side: str, fields: tuple[str, ...]) -> bool:
    selected = rows[rows["_side"].str.contains(side, na=False)]
    if selected.empty or any(field not in selected.columns for field in fields):
        return False
    last = selected.iloc[-1]
    for field in fields:
        try:
            value = float(last[field])
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
        if "odds" in field and (-100.0 < value < 100.0):
            return False
    return True


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
        if rows is None or rows.empty or not {"book", "market_type", "abbr"}.issubset(rows.columns):
            return q
        br = rows[rows["book"].astype(str) == str(q.get("book"))].copy()
        if br.empty:
            return q
        br["_market"] = br["market_type"].astype(str).str.lower().str.replace("-", "_", regex=False).str.replace(" ", "_", regex=False)
        br["_side"] = br["abbr"].map(canon_team)
        home = canon_team(game.home_team)
        away = canon_team(game.away_team)
        ml = br[br["_market"].str.contains("money", na=False)]
        sp = br[br["_market"].str.contains("spread", na=False)]
        total = br[br["_market"].str.contains("total", na=False)]
        q["open_moneyline_verified"] = (
            _valid_opening_side(ml, home, ("opening_odds",))
            and _valid_opening_side(ml, away, ("opening_odds",))
        )
        q["open_spread_verified"] = (
            _valid_opening_side(sp, home, ("opening_lines", "opening_odds"))
            and _valid_opening_side(sp, away, ("opening_lines", "opening_odds"))
        )
        q["open_total_verified"] = (
            _valid_opening_total_side(total, "over", ("opening_lines", "opening_odds"))
            and _valid_opening_total_side(total, "under", ("opening_lines", "opening_odds"))
        )
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
    verified = int(verified_entry_mask(bdf).sum())
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
