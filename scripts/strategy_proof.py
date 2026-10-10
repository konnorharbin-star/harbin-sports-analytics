"""Audit a fixed, retrospectively discovered CFB spread hypothesis.

Hypothesis: home favorites with a 6-to-8-point model spread discrepancy.
This file NEVER selects a newly optimized threshold, authorizes a bet, or
converts unverifiable archived bookmaker prices into executable records.

Only independently time-valid forward entries can validate the hypothesis.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
import numpy as np
import pandas as pd

CANDIDATE = "spread|favorite|home|6-8"
FORWARD_MIN_BETS = 100


def _summary(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {"bets": 0, "wins": 0, "losses": 0, "pushes": 0,
                "roi": None, "verified_entry_quotes": 0}
    results = pd.to_numeric(rows["result"], errors="coerce")
    profit = pd.to_numeric(rows["profit"], errors="coerce")
    if results.isna().any() or profit.isna().any():
        raise ValueError("Graded bets contain invalid results or profit")
    if not set(results.unique()).issubset({-1, 0, 1}):
        raise ValueError("Unknown graded result")
    verified = rows["entry_quote_verified"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    return {
        "bets": int(len(rows)), "wins": int((results == 1).sum()),
        "losses": int((results == -1).sum()), "pushes": int((results == 0).sum()),
        "roi": round(float(profit.mean()), 6),
        "verified_entry_quotes": int(verified.sum()),
    }



def candidate_forward_evidence(graded: pd.DataFrame | None) -> dict:
    """Study only frozen prospective entries matching the exact fixed hypothesis.

    Observed offers are shadow-market prices, not claimed sportsbook fills.
    Week-clustered bootstrap protects against multiple selections per weekend.
    """
    pending = {"status": "PENDING_FORWARD", "bets": 0, "distinct_weeks": 0,
               "roi": None, "roi_ci_95": [None, None],
               "avg_execution_clv": None, "execution_clv_samples": 0,
               "invalid_rows": 0, "paper_validated": False}
    if graded is None or graded.empty:
        return pending
    required = {"game_id", "date", "market", "subgroup_key",
                "regime_band", "entry_snapshot", "entry_quote_at",
                "book", "odds", "result", "profit", "execution_clv"}
    if required - set(graded):
        raise ValueError(f"Forward graded schema missing: {sorted(required-set(graded))}")
    subset = graded.loc[
        graded.market.astype(str).eq("spread")
        & graded.subgroup_key.astype(str).eq("favorite|home")
        & graded.regime_band.astype(str).eq("6-8")
    ].copy()
    if subset.empty:
        return pending
    kickoff = pd.to_datetime(subset.date, utc=True, errors="coerce")
    frozen = pd.to_datetime(subset.entry_snapshot, utc=True, errors="coerce")
    quoted = pd.to_datetime(subset.entry_quote_at, utc=True, errors="coerce")
    prices = pd.to_numeric(subset.odds, errors="coerce")
    payout = pd.to_numeric(subset.profit, errors="coerce")
    named = ~subset.book.fillna("").astype(str).str.fullmatch(
        r"(?i)(?:actionnetwork\s+book\s+\d+|book\s*\d+|"
        r"unknown|consensus|open|opening|unresolved|primary|\s*)"
    )
    valid = (
        kickoff.notna() & frozen.notna() & quoted.notna()
        & (quoted <= frozen) & (frozen < kickoff)
        & named & prices.notna() & (prices.abs() >= 100) & payout.notna()
    )
    excluded = int((~valid).sum())
    subset = subset.loc[valid].copy()
    if subset.empty:
        return {**pending, "invalid_rows": excluded}
    subset["_frozen"] = pd.to_datetime(subset.entry_snapshot, utc=True)
    subset = subset.sort_values("_frozen").drop_duplicates(
        ["game_id", "market"], keep="first")
    subset["_week"] = pd.to_datetime(subset.date, utc=True).dt.strftime("%G-W%V")
    profit = pd.to_numeric(subset.profit).to_numpy(dtype=float)
    clv = pd.to_numeric(subset.execution_clv, errors="coerce")
    groups = [
        pd.to_numeric(group.profit).to_numpy(dtype=float)
        for _, group in subset.groupby("_week", sort=True)
    ]
    weeks = len(groups)
    lower, upper = None, None
    if weeks >= 2:
        rng = np.random.default_rng(20261009)
        boot = []
        for _ in range(2000):
            indices = rng.integers(0, weeks, size=weeks)
            sample = np.concatenate([groups[i] for i in indices])
            boot.append(float(sample.mean()))
        lower, upper = (float(x) for x in np.quantile(boot, [.025, .975]))
    n = len(subset)
    clv_n = int(clv.notna().sum())
    clv_mean = float(clv.dropna().mean()) if clv_n else None
    validated = bool(
        n >= FORWARD_MIN_BETS and weeks >= 8
        and lower is not None and lower > 0
        and clv_n >= math.ceil(.8 * n)
        and clv_mean is not None and clv_mean > 0
    )
    return {
        "status": "PAPER_VALIDATED" if validated else "EARLY_FORWARD",
        "bets": n, "distinct_weeks": weeks,
        "roi": float(profit.mean()), "roi_ci_95": [lower, upper],
        "avg_execution_clv": clv_mean, "execution_clv_samples": clv_n,
        "invalid_rows": excluded, "paper_validated": validated,
    }


def evaluate(backtest: pd.DataFrame, forward: dict, gate: dict, board: pd.DataFrame,
             graded: pd.DataFrame | None = None) -> dict:
    required = {"market", "market_role", "side_location", "edge_bucket",
                "season", "result", "profit", "entry_quote_verified"}
    if required - set(backtest):
        raise ValueError(f"Historical bet schema missing: {sorted(required - set(backtest))}")
    observed = backtest[
        (backtest["market"].astype(str) == "spread") &
        (backtest["market_role"].astype(str) == "favorite") &
        (backtest["side_location"].astype(str) == "home") &
        (backtest["edge_bucket"].astype(str) == "6-8")
    ].copy()
    sample = _summary(observed)
    specific_forward = candidate_forward_evidence(graded)
    discovery = _summary(observed[observed.season.astype(str).isin(["2023", "2024"])])
    retrospective_holdout = _summary(observed[observed.season.astype(str) == "2025"])
    other_seasons = sorted(set(observed.season.astype(str)) - {"2023", "2024", "2025"})
    if other_seasons:
        raise ValueError(f"Unexpected historical seasons: {other_seasons}")
    clean_forward = int(forward.get("clean_entries", 0))
    graded_forward = int(forward.get("graded_bets", 0))
    if clean_forward < 0 or graded_forward < 0:
        raise ValueError("Negative forward evidence counts")
    # An aggregate forward ledger cannot be assumed to support this particular
    # hypothesis. A separate frozen candidate-specific ledger must be graded.
    subgroup = (forward.get("by_subgroup") or {}).get("favorite|home") or {}
    subgroup_bets = int(subgroup.get("graded_bets", 0))
    subgroup_roi = subgroup.get("roi")
    subgroup_ci = subgroup.get("roi_ci_95")
    subgroup_clv = subgroup.get("avg_execution_clv")
    verified_positive_forward = (
        subgroup_bets >= FORWARD_MIN_BETS and
        isinstance(subgroup_ci, (list, tuple)) and len(subgroup_ci) == 2 and
        subgroup_ci[0] is not None and float(subgroup_ci[0]) > 0 and
        subgroup_clv is not None and float(subgroup_clv) > 0
    )
    # Subgroup aggregation across edge bands alone is not sufficient.
    # Never upgrade this strategy from the aggregate subgroup metrics.
    selected = board[board.get("evidence_tier", pd.Series(dtype=str)).eq("ROBUST_CORE")].copy() if not board.empty else board
    live_candidates = []
    for _, row in selected.iterrows():
        live_candidates.append({
            "game_id": str(row.get("game_id", "")),
            "matchup": f'{row.get("away_team", "")} @ {row.get("home_team", "")}',
            "side": row.get("side"), "line": row.get("line"),
            "odds": row.get("american_odds"), "decision": row.get("decision"),
            "reasons": row.get("reasons"),
            "quote_verified": str(row.get("reasons", "")).strip() == "" and
                               str(row.get("bet_approved", "")).lower() == "true",
        })
    return {
        "schema_version": 1,
        "hypothesis": CANDIDATE,
        "hypothesis_status": "RETROSPECTIVELY_SELECTED_NOT_PROVEN",
        "historical_sample": sample,
        "discovery_2023_2024": discovery,
        "retrospective_2025_holdout": retrospective_holdout,
        "historical_quote_eligibility": "UNVERIFIED" if sample["verified_entry_quotes"] == 0 else "PARTIAL",
        "multiple_testing_warning": (
            "The 6-8 band and favorite/home category were identified from retrospective "
            "searches across markets and subgroups. The 2025 subset is not proof of a "
            "fully preregistered independent test."
        ),
        "forward_overall_clean_entries": clean_forward,
        "forward_overall_graded_bets": graded_forward,
        "forward_subgroup_graded_bets": subgroup_bets,
        "forward_subgroup_roi": subgroup_roi,
        "forward_subgroup_roi_ci_95": subgroup_ci,
        "forward_subgroup_execution_clv": subgroup_clv,
        "forward_subgroup_preliminary_gate": bool(verified_positive_forward),
        "candidate_specific_independent_evidence": specific_forward,
        "profitability_proven": False,
        "production_eligible": bool(gate.get("production_eligible", False)),
        "bet_approved_by_this_report": False,
        "research_candidates": live_candidates,
        "next_validation": (
            "Freeze eligible games and their sportsbook quote identity, odds, "
            "line and source timestamp before kickoff; grade one first-seen "
            "entry per game after final; require candidate-specific ROI CI lower "
            "bound > 0, positive timestamp-valid CLV and enough weeks/games."
        ),
        "disclaimer": (
            "Archived fallback prices are not verified placed bets. "
            "No automated wager placement or claim of demonstrated profit."
        ),
    }


def main(root=Path(".")):
    root = Path(root)
    backtest = pd.read_csv(root / "reports" / "backtest_bets.csv", low_memory=False)
    forward = json.loads((root / "reports" / "edge_forward_performance.json").read_text())
    gate = json.loads((root / "reports" / "release_gate.json").read_text())
    board = pd.read_csv(root / "outputs" / "final_edge_board.csv", dtype={"game_id": str})
    graded_path = root / "reports" / "edge_forward_graded.csv"
    if graded_path.exists() and graded_path.stat().st_size:
        try:
            graded = pd.read_csv(graded_path, low_memory=False)
        except pd.errors.EmptyDataError:
            graded = pd.DataFrame()
    else:
        graded = pd.DataFrame()
    report = evaluate(backtest, forward, gate, board, graded)
    for folder in ("outputs", "reports", "docs"):
        path = root / folder / "strategy_proof.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "hypothesis": report["hypothesis"],
        "historical_bets": report["historical_sample"]["bets"],
        "verified_historical_quotes": report["historical_sample"]["verified_entry_quotes"],
        "forward_clean_entries": report["forward_overall_clean_entries"],
        "profitability_proven": report["profitability_proven"],
        "research_candidates": len(report["research_candidates"]),
    }))
    return report


if __name__ == "__main__":
    main()
