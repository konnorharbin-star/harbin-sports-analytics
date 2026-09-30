from __future__ import annotations

from pathlib import Path
import json
import math

import pandas as pd

from .policy import load_policy


def _finite(v):
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def _read_json(path, fallback):
    p = Path(path)
    if not p.exists():
        return dict(fallback)
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else dict(fallback)
    except Exception:
        return dict(fallback)


def _read_gate(path):
    return _read_json(
        path,
        {
            "release_state": "PAPER",
            "production_eligible": False,
            "blockers": ["release gate missing or unreadable"],
        },
    )


def _kickoff_bucket(v, hours=3):
    try:
        ts = pd.to_datetime(v, utc=True, errors="raise")
        return str(ts.floor(f"{max(1, int(hours))}h"))
    except Exception:
        return "unknown"


def _num(value, default):
    try:
        x = float(value)
        return x if math.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _clean_book(value):
    s = str(value or "").strip()
    return "" if s.lower() in {"", "nan", "none", "null", "unknown"} else s


def _row_game_key(row):
    gid = str(row.get("game_id") or "").strip()
    if gid and gid.lower() not in {"nan", "none"}:
        return gid
    return f"{row.get('away_team','')}@{row.get('home_team','')}|{row.get('date','')}"


def _team_exposure_keys(row):
    """Directional markets expose the selected team; totals expose both teams."""
    market = str(row.get("quant_market") or "").lower()
    side = str(row.get("quant_side") or "")
    home = str(row.get("home_team") or "")
    away = str(row.get("away_team") or "")
    if market in {"moneyline", "spread"} and side in {home, away}:
        return [side]
    if market == "total":
        return [x for x in (home, away) if x]
    return []


def _execution_issue(row, limits):
    """Return a reason when a proposed bet is not executable from stored state."""
    market = str(row.get("quant_market") or "").lower()
    side = str(row.get("quant_side") or "").strip()
    if market not in {"moneyline", "spread", "total"}:
        return "unsupported or missing quant market"
    if not side or side.lower() in {"nan", "none"}:
        return "missing quant side"
    if not _finite(row.get("quant_odds")) or float(row.get("quant_odds")) == 0:
        return "missing executable American odds"
    if market in {"spread", "total"} and not _finite(row.get("quant_price")):
        return "missing executable line"
    if bool(limits.get("require_executable_book", True)) and not _clean_book(row.get("quant_book")):
        return "missing executable sportsbook provenance"
    min_books = max(1, int(_num(limits.get("min_market_book_count_for_execution", 1), 1)))
    if _finite(row.get("market_book_count")) and int(float(row.get("market_book_count"))) < min_books:
        return f"market book count below {min_books}"
    return ""


def build_bankroll_risk_state(live_bets_path="reports/live_graded_bets.csv", limits=None):
    """Build a unit-based risk throttle from independent graded live/shadow bets.

    No dollar bankroll is invented. Profit and drawdown are measured in flat-stake
    betting units from the grading ledger. The release gate remains the authority on
    whether the sample is large enough for production.
    """
    limits = limits or {}
    soft = max(0.0, _num(limits.get("drawdown_soft_stop_units", 8.0), 8.0))
    hard = max(soft + 1e-9, _num(limits.get("drawdown_hard_stop_units", 15.0), 15.0))
    floor = max(0.0, min(1.0, _num(limits.get("drawdown_floor_multiplier", 0.25), 0.25)))
    tail_n = max(10, int(_num(limits.get("trailing_window_bets", 50), 50)))
    min_tail = max(10, int(_num(limits.get("min_trailing_bets_for_throttle", 30), 30)))
    roi_trigger = _num(limits.get("trailing_roi_throttle", -0.10), -0.10)
    clv_trigger = _num(limits.get("trailing_clv_throttle", 0.0), 0.0)
    adverse_mult = max(0.0, min(1.0, _num(limits.get("adverse_run_multiplier", 0.50), 0.50)))
    p = Path(live_bets_path)
    empty = {
        "history_available": False,
        "graded_bets": 0,
        "cumulative_units": 0.0,
        "current_drawdown_units": 0.0,
        "max_drawdown_units": 0.0,
        "trailing_window_bets": 0,
        "trailing_roi": None,
        "trailing_avg_clv": None,
        "risk_multiplier": 1.0,
        "hard_stop": False,
        "reason": "no independent graded live/shadow betting history; unit throttle is neutral outside production",
    }
    if not p.exists():
        return empty
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception as exc:
        out = dict(empty)
        out["reason"] = f"live betting ledger unreadable: {type(exc).__name__}"
        return out
    if df.empty or "profit" not in df.columns:
        return empty

    work = df.copy()
    sort_cols = [c for c in ("kickoff", "bet_entry_snapshot", "season", "week", "game_id") if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, kind="mergesort")
    work["_profit"] = pd.to_numeric(work["profit"], errors="coerce")
    work = work[work["_profit"].notna()].copy()
    if work.empty:
        return empty

    profit = work["_profit"].astype(float)
    curve = profit.cumsum()
    running_peak = curve.cummax().clip(lower=0.0)
    drawdowns = running_peak - curve
    current_dd = max(0.0, float(running_peak.iloc[-1] - curve.iloc[-1]))
    max_dd = max(0.0, float(drawdowns.max()))
    tail = work.tail(tail_n)
    trailing_roi = float(tail["_profit"].mean()) if len(tail) else None

    clv_col = None
    for candidate in ("execution_clv", "clv_proxy", "clv"):
        if candidate in tail.columns and pd.to_numeric(tail[candidate], errors="coerce").notna().any():
            clv_col = candidate
            break
    trailing_clv = None
    if clv_col:
        c = pd.to_numeric(tail[clv_col], errors="coerce").dropna()
        trailing_clv = float(c.mean()) if len(c) else None

    hard_stop = current_dd >= hard - 1e-12
    mult = 0.0 if hard_stop else 1.0
    reasons = []
    if hard_stop:
        reasons.append(f"current drawdown {current_dd:.2f}u reached hard stop {hard:.2f}u")
    elif current_dd > soft:
        frac = min(1.0, max(0.0, (current_dd - soft) / (hard - soft)))
        mult = 1.0 - frac * (1.0 - floor)
        reasons.append(f"drawdown throttle active above {soft:.2f}u soft stop")

    if (
        not hard_stop
        and len(tail) >= min_tail
        and trailing_roi is not None
        and trailing_roi <= roi_trigger
        and trailing_clv is not None
        and trailing_clv <= clv_trigger
    ):
        mult *= adverse_mult
        reasons.append("recent ROI and CLV are jointly adverse")

    return {
        "history_available": True,
        "graded_bets": int(len(work)),
        "cumulative_units": round(float(profit.sum()), 4),
        "current_drawdown_units": round(current_dd, 4),
        "max_drawdown_units": round(max_dd, 4),
        "trailing_window_bets": int(len(tail)),
        "trailing_roi": round(trailing_roi, 6) if trailing_roi is not None else None,
        "trailing_avg_clv": round(trailing_clv, 6) if trailing_clv is not None else None,
        "risk_multiplier": round(max(0.0, min(1.0, mult)), 4),
        "hard_stop": bool(hard_stop),
        "reason": "; ".join(reasons) if reasons else "no bankroll throttle active",
    }


def _effective_caps(limits, bankroll_multiplier):
    m = max(0.0, min(1.0, float(bankroll_multiplier)))
    names = {
        "max_slate_units": 5.0,
        "max_game_units": 1.0,
        "max_team_units": 1.5,
        "max_market_units": 2.5,
        "max_kickoff_window_units": 2.0,
        "max_book_units": 2.0,
    }
    return {name: max(0.0, _num(limits.get(name, default), default)) * m for name, default in names.items()}


def apply_portfolio_controls(
    pred: pd.DataFrame,
    policy_path="reports/production_policy.json",
    release_gate_path="outputs/release_gate.json",
    live_bets_path="reports/live_graded_bets.csv",
) -> tuple[pd.DataFrame, dict]:
    """Apply execution, bankroll and concentration controls to quant candidates."""
    policy = load_policy(policy_path)
    gate = _read_gate(release_gate_path)
    limits = policy.get("portfolio") or {}
    out = pred.copy()
    gate_state = str(gate.get("release_state", "PAPER")).upper()
    policy_mode = str(policy.get("deployment_mode", "paper")).lower()
    production_gate_open = bool(gate.get("production_eligible")) and policy_mode == "production"
    bankroll = build_bankroll_risk_state(live_bets_path, limits)
    require_live_history = bool(limits.get("require_live_history_for_production", True))
    production_history_ok = bankroll["history_available"] or not require_live_history
    production_allowed = production_gate_open and production_history_ok and not bankroll["hard_stop"]
    if production_gate_open and not production_allowed:
        effective_mode = "halted"
    elif production_allowed:
        effective_mode = "production"
    elif gate_state == "SHADOW":
        effective_mode = "shadow"
    else:
        effective_mode = "paper"

    base_summary = {
        "mode": effective_mode,
        "policy_mode": policy_mode,
        "release_state": gate_state,
        "production_gate_open": production_gate_open,
        "production_eligible": production_allowed,
        "bankroll_risk": bankroll,
    }
    if out.empty:
        return out, {
            **base_summary,
            "proposed_units": 0.0,
            "risk_adjusted_proposed_units": 0.0,
            "approved_units": 0.0,
            "paper_allocated_units": 0.0,
            "paper_or_shadow_allocated_units": 0.0,
            "bets": 0,
            "approved_bets": 0,
            "execution_blocked_bets": 0,
        }

    out["paper_stake_units"] = pd.to_numeric(out.get("stake_units", 0), errors="coerce").fillna(0.0).clip(lower=0.0)
    bankroll_mult = float(bankroll.get("risk_multiplier", 1.0) or 0.0)
    out["bankroll_adjusted_units"] = (out["paper_stake_units"] * bankroll_mult).round(6)
    out["portfolio_candidate_units"] = 0.0
    out["portfolio_stake_units"] = 0.0
    out["portfolio_action"] = "PASS"
    out["portfolio_limit_reason"] = ""
    out["execution_ready"] = False
    out["portfolio_rank_score"] = (
        pd.to_numeric(out.get("quant_ev", 0), errors="coerce").fillna(0.0)
        * pd.to_numeric(out.get("risk_multiplier", 1), errors="coerce").fillna(0.0)
    )

    order_frame = pd.DataFrame(
        {
            "idx": list(out.index),
            "score": out["portfolio_rank_score"].to_numpy(float),
            "ev": pd.to_numeric(out.get("quant_ev", 0), errors="coerce").fillna(0.0).to_numpy(float),
        }
    )
    order = list(order_frame.sort_values(["score", "ev", "idx"], ascending=[False, False, True], kind="mergesort")["idx"])

    caps = _effective_caps(limits, bankroll_mult)
    min_alloc = max(0.0, _num(limits.get("min_allocation_units", 0.05), 0.05))
    max_bets = max(1, int(_num(limits.get("max_bets", 20), 20)))
    window_hours = max(1, int(_num(limits.get("kickoff_window_hours", 3), 3)))

    slate = 0.0
    by_game = {}
    by_team = {}
    by_market = {}
    by_book = {}
    by_window = {}
    allocated_bets = 0
    execution_blocked = 0
    limit_hits = {}

    for i in order:
        r = out.loc[i]
        signal = str(r.get("quant_signal") or "PASS").upper()
        raw_proposed = max(0.0, _num(r.get("paper_stake_units"), 0.0))
        if signal == "PASS" or raw_proposed <= 0:
            continue
        if bankroll_mult <= 0:
            out.at[i, "portfolio_limit_reason"] = bankroll.get("reason") or "bankroll risk multiplier is zero"
            continue
        proposed = max(0.0, _num(r.get("bankroll_adjusted_units"), 0.0))

        issue = _execution_issue(r, limits)
        if issue:
            execution_blocked += 1
            out.at[i, "portfolio_limit_reason"] = issue
            continue
        out.at[i, "execution_ready"] = True

        if allocated_bets >= max_bets:
            out.at[i, "portfolio_limit_reason"] = "max bet count"
            limit_hits["max bet count"] = limit_hits.get("max bet count", 0) + 1
            continue

        game = _row_game_key(r)
        teams = _team_exposure_keys(r)
        market = str(r.get("quant_market") or "unknown").lower()
        book = _clean_book(r.get("quant_book")) or "unattributed"
        bucket = _kickoff_bucket(r.get("date"), window_hours)

        residuals = {
            "game cap": caps["max_game_units"] - by_game.get(game, 0.0),
            "slate cap": caps["max_slate_units"] - slate,
            "market cap": caps["max_market_units"] - by_market.get(market, 0.0),
            "book cap": caps["max_book_units"] - by_book.get(book, 0.0),
            "kickoff-cluster cap": caps["max_kickoff_window_units"] - by_window.get(bucket, 0.0),
        }
        for team in teams:
            residuals[f"team cap:{team}"] = caps["max_team_units"] - by_team.get(team, 0.0)

        remaining = min(residuals.values()) if residuals else proposed
        units = max(0.0, min(proposed, remaining))
        if units + 1e-12 < min_alloc:
            binding = [name for name, value in residuals.items() if value <= min_alloc + 1e-12]
            reason = "; ".join(binding) if binding else "minimum allocation"
            out.at[i, "portfolio_limit_reason"] = reason
            for name in binding or ["minimum allocation"]:
                limit_hits[name] = limit_hits.get(name, 0) + 1
            continue

        units = round(units, 2)
        out.at[i, "portfolio_candidate_units"] = units
        out.at[i, "portfolio_stake_units"] = units if production_allowed else 0.0
        out.at[i, "portfolio_action"] = "BET" if production_allowed else "SHADOW" if effective_mode == "shadow" else "PAPER"

        if units + 1e-9 < proposed:
            binding = [name for name, value in residuals.items() if abs(value - remaining) < 1e-9]
            out.at[i, "portfolio_limit_reason"] = "; ".join(binding) or "concentration cap reduced stake"
            for name in binding or ["concentration cap reduced stake"]:
                limit_hits[name] = limit_hits.get(name, 0) + 1

        slate += units
        allocated_bets += 1
        by_game[game] = by_game.get(game, 0.0) + units
        by_market[market] = by_market.get(market, 0.0) + units
        by_book[book] = by_book.get(book, 0.0) + units
        by_window[bucket] = by_window.get(bucket, 0.0) + units
        for team in teams:
            by_team[team] = by_team.get(team, 0.0) + units

    allocated = round(float(slate), 2)
    if production_gate_open and not production_history_ok:
        production_block = "production release gate is open but the independent live betting ledger is unavailable"
    elif production_gate_open and bankroll["hard_stop"]:
        production_block = bankroll["reason"]
    else:
        production_block = ""

    summary = {
        **base_summary,
        "production_block_reason": production_block,
        "proposed_units": round(float(out["paper_stake_units"].sum()), 2),
        "risk_adjusted_proposed_units": round(float(out["bankroll_adjusted_units"].sum()), 2),
        "approved_units": round(float(out["portfolio_stake_units"].sum()), 2),
        "paper_allocated_units": allocated,
        "paper_or_shadow_allocated_units": allocated,
        "bets": int((out["portfolio_action"] != "PASS").sum()),
        "approved_bets": int((out["portfolio_action"] == "BET").sum()),
        "execution_blocked_bets": int(execution_blocked),
        "limits": {
            **limits,
            "max_kickoff_window_units": _num(limits.get("max_kickoff_window_units", 2.0), 2.0),
            "kickoff_window_hours": window_hours,
            "max_book_units": _num(limits.get("max_book_units", 2.0), 2.0),
            "max_bets": max_bets,
            "min_allocation_units": min_alloc,
        },
        "effective_unit_caps": {k: round(v, 4) for k, v in caps.items()},
        "allocated_by_market": {k: round(v, 2) for k, v in by_market.items()},
        "allocated_by_book": {k: round(v, 2) for k, v in by_book.items()},
        "allocated_by_kickoff_window": {k: round(v, 2) for k, v in by_window.items()},
        "allocated_by_team_exposure": {k: round(v, 2) for k, v in by_team.items()},
        "limit_hit_counts": dict(sorted(limit_hits.items())),
        "release_blockers": gate.get("blockers", [])[:20],
    }
    return out, summary


def write_portfolio_outputs(
    pred: pd.DataFrame,
    output_dir="outputs",
    policy_path="reports/production_policy.json",
    release_gate_path="outputs/release_gate.json",
    live_bets_path="reports/live_graded_bets.csv",
):
    out, summary = apply_portfolio_controls(pred, policy_path, release_gate_path, live_bets_path)
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    cols = [
        c
        for c in [
            "game_id",
            "date",
            "away_team",
            "home_team",
            "quant_signal",
            "quant_market",
            "quant_side",
            "quant_book",
            "quant_price",
            "quant_odds",
            "quant_probability",
            "quant_ev",
            "quant_edge",
            "risk_multiplier",
            "data_quality_score",
            "paper_stake_units",
            "bankroll_adjusted_units",
            "portfolio_candidate_units",
            "portfolio_stake_units",
            "execution_ready",
            "portfolio_action",
            "portfolio_limit_reason",
        ]
        if c in out.columns
    ]
    out[cols].to_csv(p / "portfolio_card.csv", index=False)
    (p / "portfolio_summary.json").write_text(json.dumps(summary, indent=2))
    return out, summary
