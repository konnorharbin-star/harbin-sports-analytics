from __future__ import annotations

"""Runtime hardening for the historical market backtest.

The SportsDataverse betting CSV contains nullable historical game IDs. Pandas
therefore may materialize the column as floats (e.g. ``401520123.0``), while
our schedule game IDs are strings without the decimal suffix. The original
ArchiveMarketStore grouped on the raw string representation, causing modern
2020-2025 quotes to miss their games and silently producing zero wagers.

This adapter canonicalizes every archive game ID before the existing backtest
logic runs. It deliberately leaves the scoring/market logic unchanged.
"""

from . import backtest as _bt


def canonical_game_id(value) -> str:
    try:
        if value is None:
            return ""
        s = str(value).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return ""
        return str(int(float(s)))
    except Exception:
        return str(value).strip()


class CanonicalArchiveMarketStore(_bt.ArchiveMarketStore):
    def __init__(self, cache_dir="cache/backtest"):
        super().__init__(cache_dir=cache_dir)
        if len(self.df) and "game_id" in self.df.columns:
            self.df["_gid"] = self.df["game_id"].map(canonical_game_id)
            self.df = self.df[self.df["_gid"] != ""].copy()
            self.by_id = {k: v.copy() for k, v in self.df.groupby("_gid", sort=False)}


# Patch the class reference used inside the already-tested run_backtest function.
_bt.ArchiveMarketStore = CanonicalArchiveMarketStore
run_backtest = _bt.run_backtest
