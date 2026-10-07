from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .market import roi
from .edge_price import price_evidence


EDGE_BINS = (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 999.0)
PERSISTENT_MIN_BETS = 180
PERSISTENT_MIN_SEASON_BETS = 50
WATCH_MIN_BETS = 100
SUBGROUP_MIN_BETS = 45
SUBGROUP_MIN_SEASON_BETS = 12
SUBGROUP_CONFIRMATION_MIN_BETS = 12
EDGE_REGIME_SCHEMA_VERSION = 3


def _finite(v):
    try:
        return math.isfinite(float(v))
    except Exception:
        return False


def edge_band(edge):
    try:
        x = abs(float(edge))
    except Exception:
        return None
    for lo, hi in zip(EDGE_BINS[:-1], EDGE_BINS[1:]):
        if lo <= x < hi:
            return f"{lo:g}-{hi:g}"
    return None


def _segment_stats(frame):
    if frame.empty:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": None,
            "units": 0.0,
            "roi": None,
            "avg_clv": None,
            "positive_clv_rate": None,
        }
    result = pd.to_numeric(frame.get("result"), errors="coerce")
    profit = pd.to_numeric(frame.get("profit"), errors="coerce")
    clv = pd.to_numeric(frame.get("clv"), errors="coerce").dropna()
    wins = int((result > 0).sum())
    losses = int((result < 0).sum())
    pushes = int((result == 0).sum())
    units = float(profit.fillna(0).sum())
    return {
        "bets": int(len(frame)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": float(wins / max(1, wins + losses)),
        "units": units,
        "roi": float(units / max(1, len(frame))),
        "avg_clv": float(clv.mean()) if len(clv) else None,
        "positive_clv_rate": float((clv > 0).mean()) if len(clv) else None,
    }


def _status(overall, seasons):
    season_rows = list(seasons.values())
    season_count = len(season_rows)
    profitable = sum(
        row.get("roi") is not None and float(row["roi"]) > 0
        for row in season_rows
    )
    min_season_bets = min(
        (int(row.get("bets", 0)) for row in season_rows),
        default=0,
    )
    roi_value = overall.get("roi")
    avg_clv = overall.get("avg_clv")

    persistent = (
        int(overall.get("bets", 0)) >= PERSISTENT_MIN_BETS
        and season_count >= 3
        and min_season_bets >= PERSISTENT_MIN_SEASON_BETS
        and profitable == season_count
        and roi_value is not None
        and float(roi_value) >= 0.03
        and avg_clv is not None
        and float(avg_clv) > 0
    )
    if persistent:
        return "PERSISTENT_CANDIDATE"

    watch = (
        int(overall.get("bets", 0)) >= WATCH_MIN_BETS
        and season_count >= 2
        and profitable >= max(2, math.ceil(season_count * 2 / 3))
        and roi_value is not None
        and float(roi_value) > 0
        and (avg_clv is None or float(avg_clv) >= 0)
    )
    return "WATCH" if watch else "UNSUPPORTED"


def _subgroup_status(overall, discovery, discovery_seasons, holdout):
    """Classify a subgroup using earlier seasons for discovery and latest for holdout."""

    discovery_rows = list(discovery_seasons.values())
    discovery_count = len(discovery_rows)
    discovery_profitable = sum(
        row.get("roi") is not None and float(row["roi"]) > 0
        for row in discovery_rows
    )
    discovery_negative = sum(
        row.get("roi") is not None and float(row["roi"]) < 0
        for row in discovery_rows
    )
    discovery_min_bets = min(
        (int(row.get("bets", 0)) for row in discovery_rows),
        default=0,
    )
    discovery_roi = discovery.get("roi")
    holdout_roi = holdout.get("roi")
    enough = (
        int(overall.get("bets", 0)) >= SUBGROUP_MIN_BETS
        and discovery_count >= 2
        and discovery_min_bets >= SUBGROUP_MIN_SEASON_BETS
        and int(holdout.get("bets", 0)) >= SUBGROUP_CONFIRMATION_MIN_BETS
    )

    if (
        enough
        and discovery_profitable == discovery_count
        and discovery_roi is not None
        and float(discovery_roi) >= 0.05
        and holdout_roi is not None
        and float(holdout_roi) > 0
        and overall.get("avg_clv") is not None
        and float(overall["avg_clv"]) > 0
    ):
        return "SUPPORTED_SUBGROUP"

    if (
        enough
        and discovery_negative == discovery_count
        and discovery_roi is not None
        and float(discovery_roi) <= -0.03
        and holdout_roi is not None
        and float(holdout_roi) < 0
        and overall.get("roi") is not None
        and float(overall["roi"]) <= -0.03
    ):
        return "CONTRAINDICATED_SUBGROUP"

    return "INCONCLUSIVE_SUBGROUP"


def _spread_subgroups(segment):
    if segment.empty or not {"market_role", "side_location"}.issubset(segment.columns):
        return {}
    data = segment.copy()
    data["market_role"] = data["market_role"].fillna("").astype(str).str.lower()
    data["side_location"] = data["side_location"].fillna("").astype(str).str.lower()
    data = data[
        data["market_role"].isin({"favorite", "underdog", "pickem"})
        & data["side_location"].isin({"home", "away"})
    ]
    out = {}
    for (role, location), rows in data.groupby(["market_role", "side_location"]):
        overall = _segment_stats(rows)
        by_season = {}
        season_frames = {}
        if "season" in rows.columns:
            for season, season_rows in rows.groupby("season"):
                season_key = str(int(float(season)))
                by_season[season_key] = _segment_stats(season_rows)
                season_frames[season_key] = season_rows

        ordered_seasons = sorted(
            by_season,
            key=lambda value: int(float(value)),
        )
        holdout_season = ordered_seasons[-1] if ordered_seasons else None
        discovery_seasons = ordered_seasons[:-1]
        discovery_frame = (
            pd.concat(
                [season_frames[season] for season in discovery_seasons],
                ignore_index=False,
            )
            if discovery_seasons
            else rows.iloc[0:0]
        )
        holdout_frame = (
            season_frames[holdout_season]
            if holdout_season is not None
            else rows.iloc[0:0]
        )
        discovery = _segment_stats(discovery_frame)
        holdout = _segment_stats(holdout_frame)
        discovery_by_season = {
            season: by_season[season]
            for season in discovery_seasons
        }
        status = _subgroup_status(
            overall,
            discovery,
            discovery_by_season,
            holdout,
        )
        profitable = sum(
            row.get("roi") is not None and float(row["roi"]) > 0
            for row in by_season.values()
        )
        holdout_roi = holdout.get("roi")
        discovery_roi = discovery.get("roi")
        holdout_confirmed = (
            status in {"SUPPORTED_SUBGROUP", "CONTRAINDICATED_SUBGROUP"}
        )

        key = f"{role}|{location}"
        out[key] = {
            "key": key,
            "market_role": role,
            "side_location": location,
            "status": status,
            **overall,
            "season_count": int(len(by_season)),
            "profitable_seasons": int(profitable),
            "min_season_bets": min(
                (int(row.get("bets", 0)) for row in by_season.values()),
                default=0,
            ),
            "validation_design": "latest_season_holdout",
            "discovery_seasons": discovery_seasons,
            "discovery_bets": int(discovery.get("bets", 0)),
            "discovery_roi": discovery_roi,
            "discovery_win_rate": discovery.get("win_rate"),
            "discovery_profitable_seasons": int(
                sum(
                    row.get("roi") is not None and float(row["roi"]) > 0
                    for row in discovery_by_season.values()
                )
            ),
            "holdout_season": holdout_season,
            "holdout_bets": int(holdout.get("bets", 0)),
            "holdout_roi": holdout_roi,
            "holdout_win_rate": holdout.get("win_rate"),
            "holdout_confirmed": bool(holdout_confirmed),
            "by_season": by_season,
        }
    return out


def _candidate_subgroup_key(market, side, line, home_team, away_team):
    if str(market or "").lower() != "spread" or not _finite(line):
        return None
    side = str(side or "")
    home = str(home_team or "")
    away = str(away_team or "")
    location = "home" if side == home else "away" if side == away else None
    if location is None:
        return None
    line_value = float(line)
    role = "favorite" if line_value < 0 else "underdog" if line_value > 0 else "pickem"
    return f"{role}|{location}"


def build_edge_regime_report(bets):
    """Find historically persistent model-market disagreement regimes.

    The edge bands are fixed before reading outcomes. A regime can only be promoted to
    PERSISTENT_CANDIDATE when it is profitable in every represented evaluation season,
    has meaningful per-season sample size, positive aggregate ROI, and positive CLV.
    This report never creates a model edge and never changes the fair projection.
    """

    data = bets.copy() if isinstance(bets, pd.DataFrame) else pd.DataFrame(bets)
    if data.empty:
        return {
            "schema_version": EDGE_REGIME_SCHEMA_VERSION,
            "status": "NO_SAMPLE",
            "regimes": [],
            "persistent_regimes": 0,
            "watch_regimes": 0,
            "methodology": _methodology(),
        }

    data["market"] = data.get("market", "").astype(str).str.lower()
    data["edge"] = pd.to_numeric(data.get("edge"), errors="coerce")
    data = data[data["edge"].notna()].copy()
    data["edge_band"] = data["edge"].map(edge_band)

    regimes = []
    for market in ("spread", "total", "moneyline"):
        market_rows = data[data["market"] == market]
        for lo, hi in zip(EDGE_BINS[:-1], EDGE_BINS[1:]):
            band = f"{lo:g}-{hi:g}"
            segment = market_rows[
                (market_rows["edge"].abs() >= lo)
                & (market_rows["edge"].abs() < hi)
            ]
            if segment.empty:
                continue
            overall = _segment_stats(segment)
            by_season = {}
            if "season" in segment.columns:
                for season, season_rows in segment.groupby("season"):
                    by_season[str(int(float(season)))] = _segment_stats(season_rows)
            status = _status(overall, by_season)
            profitable_seasons = sum(
                row.get("roi") is not None and float(row["roi"]) > 0
                for row in by_season.values()
            )
            regimes.append(
                {
                    "market": market,
                    "edge_band": band,
                    "min_edge": lo,
                    "max_edge": hi,
                    "status": status,
                    **overall,
                    "season_count": int(len(by_season)),
                    "profitable_seasons": int(profitable_seasons),
                    "min_season_bets": min(
                        (int(row.get("bets", 0)) for row in by_season.values()),
                        default=0,
                    ),
                    "by_season": by_season,
                    "subgroups": _spread_subgroups(segment) if market == "spread" else {},
                }
            )

    order = {"PERSISTENT_CANDIDATE": 0, "WATCH": 1, "UNSUPPORTED": 2}
    regimes.sort(
        key=lambda row: (
            order.get(row["status"], 9),
            row["market"],
            float(row["min_edge"]),
        )
    )
    return {
        "schema_version": EDGE_REGIME_SCHEMA_VERSION,
        "status": "TRACKING",
        "regimes": regimes,
        "persistent_regimes": int(
            sum(row["status"] == "PERSISTENT_CANDIDATE" for row in regimes)
        ),
        "watch_regimes": int(sum(row["status"] == "WATCH" for row in regimes)),
        "methodology": _methodology(),
    }


def _methodology():
    return (
        "Fixed edge bands; market-specific walk-forward bet outcomes; "
        "PERSISTENT_CANDIDATE requires >=180 bets, >=3 seasons, >=50 bets in every "
        "season, positive ROI in every season, >=3% aggregate ROI, and positive "
        "average CLV. WATCH requires >=100 bets, positive aggregate ROI, and profit "
        "in at least two-thirds of seasons. Spread parent regimes are decomposed into "
        "favorite/underdog × home/away children using chronological confirmation: "
        "all but the latest represented season are discovery and the latest season is "
        "a holdout. SUPPORTED_SUBGROUP requires >=45 total bets, >=2 discovery seasons "
        "with >=12 bets each, >=5% discovery ROI with every discovery season profitable, "
        ">=12 holdout bets with positive holdout ROI, and positive full-sample CLV. "
        "CONTRAINDICATED_SUBGROUP requires the same sample floor, every discovery season "
        "negative with <=-3% discovery ROI, a negative holdout ROI, and <=-3% full-sample "
        "ROI. A holdout-confirmed child may stand independently beneath a WATCH parent, "
        "but never beneath an UNSUPPORTED parent. Regime evidence can prioritize research "
        "and surface candidates but cannot "
        "create an edge, modify the fair line, probability, EV, or inflate stake."
    )


def write_edge_regime_report(bets, json_path, csv_path=None):
    report = build_edge_regime_report(bets)
    Path(json_path).write_text(json.dumps(report, indent=2))
    if csv_path is not None:
        flat = []
        for row in report["regimes"]:
            flat.append({k: v for k, v in row.items() if k != "by_season"})
        pd.DataFrame(flat).to_csv(csv_path, index=False)
    return report


def _edge_report_compatible(report):
    if not isinstance(report, dict):
        return False
    if int(report.get("schema_version", 0) or 0) < EDGE_REGIME_SCHEMA_VERSION:
        return False
    for row in report.get("regimes") or []:
        if (
            str(row.get("market") or "").lower() == "spread"
            and row.get("status") == "PERSISTENT_CANDIDATE"
            and not isinstance(row.get("subgroups"), dict)
        ):
            return False
    return True


def load_or_build_edge_regime_report(reports_dir="reports"):
    reports = Path(reports_dir)
    path = reports / "edge_regimes.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            if _edge_report_compatible(cached):
                return cached
        except Exception:
            pass

    bets_path = reports / "backtest_bets.csv"
    if bets_path.exists():
        try:
            rebuilt = build_edge_regime_report(
                pd.read_csv(bets_path, low_memory=False)
            )
            try:
                path.write_text(json.dumps(rebuilt, indent=2))
            except Exception:
                pass
            return rebuilt
        except Exception:
            pass
    return build_edge_regime_report(pd.DataFrame())


def match_edge_regime(report, market, edge):
    if not _finite(edge):
        return None
    market = str(market or "").lower()
    x = abs(float(edge))
    for row in (report or {}).get("regimes") or []:
        if str(row.get("market") or "").lower() != market:
            continue
        lo = float(row.get("min_edge", -1))
        hi = float(row.get("max_edge", -1))
        if lo <= x < hi:
            return row
    return None


def match_edge_subgroup(report, market, edge, side, line, home_team, away_team):
    regime = match_edge_regime(report, market, edge)
    if not regime:
        return None
    key = _candidate_subgroup_key(market, side, line, home_team, away_team)
    if key is None:
        return None
    subgroup = (regime.get("subgroups") or {}).get(key)
    if subgroup:
        return subgroup
    return {
        "key": key,
        "market_role": key.split("|", 1)[0],
        "side_location": key.split("|", 1)[1],
        "status": "INCONCLUSIVE_SUBGROUP",
        "bets": 0,
        "roi": None,
        "season_count": 0,
        "profitable_seasons": 0,
        "min_season_bets": 0,
    }


def effective_edge_status(regime, subgroup=None):
    """Return the evidence status that is allowed to influence selection.

    A chronologically confirmed child may stand on its own beneath a WATCH parent,
    because the child has already passed independent discovery + latest-season
    holdout requirements. Children under an UNSUPPORTED parent are not promoted;
    this prevents subgroup mining from rescuing broadly failed edge bands.
    """

    parent = (regime or {}).get("status", "UNSUPPORTED")
    child = (subgroup or {}).get("status")
    if parent not in {"PERSISTENT_CANDIDATE", "WATCH"}:
        return parent
    if child == "CONTRAINDICATED_SUBGROUP":
        return "CONTRAINDICATED_SUBGROUP"
    if child == "SUPPORTED_SUBGROUP":
        return "SUPPORTED_SUBGROUP"
    if parent == "PERSISTENT_CANDIDATE":
        return "PERSISTENT_PARENT_ONLY"
    return "WATCH"


def annotate_selected_regimes(frame, report):
    out = frame.copy()
    fields = {
        "edge_regime_parent_status": "UNSUPPORTED",
        "edge_regime_status": "UNSUPPORTED",
        "edge_regime_band": "",
        "edge_regime_bets": 0,
        "edge_regime_roi": np.nan,
        "edge_regime_profitable_seasons": 0,
        "edge_regime_season_count": 0,
        "edge_regime_candidate": False,
        "edge_subgroup_key": "",
        "edge_subgroup_status": "INCONCLUSIVE_SUBGROUP",
        "edge_subgroup_bets": 0,
        "edge_subgroup_roi": np.nan,
        "edge_subgroup_profitable_seasons": 0,
        "edge_subgroup_season_count": 0,
        "edge_subgroup_discovery_roi": np.nan,
        "edge_subgroup_holdout_season": "",
        "edge_subgroup_holdout_bets": 0,
        "edge_subgroup_holdout_roi": np.nan,
        "edge_subgroup_holdout_win_rate": np.nan,
        "edge_subgroup_holdout_confirmed": False,
    }
    for key, default in fields.items():
        if key not in out.columns:
            out[key] = default

    for idx, row in out.iterrows():
        regime = match_edge_regime(
            report,
            row.get("quant_market"),
            row.get("quant_edge"),
        )
        if not regime:
            continue
        subgroup = match_edge_subgroup(
            report,
            row.get("quant_market"),
            row.get("quant_edge"),
            row.get("quant_side"),
            row.get("quant_price"),
            row.get("home_team"),
            row.get("away_team"),
        )
        effective = effective_edge_status(regime, subgroup)
        out.at[idx, "edge_regime_parent_status"] = regime.get("status", "UNSUPPORTED")
        out.at[idx, "edge_regime_status"] = effective
        out.at[idx, "edge_regime_band"] = regime.get("edge_band", "")
        out.at[idx, "edge_regime_bets"] = int(regime.get("bets", 0))
        out.at[idx, "edge_regime_roi"] = regime.get("roi")
        out.at[idx, "edge_regime_profitable_seasons"] = int(
            regime.get("profitable_seasons", 0)
        )
        out.at[idx, "edge_regime_season_count"] = int(regime.get("season_count", 0))
        out.at[idx, "edge_regime_candidate"] = effective in {
            "SUPPORTED_SUBGROUP",
            "PERSISTENT_PARENT_ONLY",
        }
        if subgroup:
            out.at[idx, "edge_subgroup_key"] = subgroup.get("key", "")
            out.at[idx, "edge_subgroup_status"] = subgroup.get(
                "status", "INCONCLUSIVE_SUBGROUP"
            )
            out.at[idx, "edge_subgroup_bets"] = int(subgroup.get("bets", 0) or 0)
            out.at[idx, "edge_subgroup_roi"] = subgroup.get("roi")
            out.at[idx, "edge_subgroup_profitable_seasons"] = int(
                subgroup.get("profitable_seasons", 0) or 0
            )
            out.at[idx, "edge_subgroup_season_count"] = int(
                subgroup.get("season_count", 0) or 0
            )
            out.at[idx, "edge_subgroup_discovery_roi"] = subgroup.get(
                "discovery_roi"
            )
            out.at[idx, "edge_subgroup_holdout_season"] = str(
                subgroup.get("holdout_season") or ""
            )
            out.at[idx, "edge_subgroup_holdout_bets"] = int(
                subgroup.get("holdout_bets", 0) or 0
            )
            out.at[idx, "edge_subgroup_holdout_roi"] = subgroup.get(
                "holdout_roi"
            )
            out.at[idx, "edge_subgroup_holdout_win_rate"] = subgroup.get(
                "holdout_win_rate"
            )
            out.at[idx, "edge_subgroup_holdout_confirmed"] = bool(
                subgroup.get("holdout_confirmed", False)
            )
    return out

def _market_candidate(row, market):
    home = row.get("home_team")
    away = row.get("away_team")
    if market == "spread":
        side = row.get("spread_team")
        edge = row.get("spread_edge_pts")
        probability = row.get("cover_probability")
        line = row.get("spread_line")
        odds = row.get("spread_odds")
        book = row.get("spread_book")
        quote_at = row.get("spread_quote_at")
        badge = row.get("spread_badge")
    elif market == "total":
        side = row.get("total_dir")
        edge = row.get("total_edge_pts")
        probability = row.get("total_probability")
        line = row.get("market_total")
        odds = row.get("total_odds")
        book = row.get("total_book")
        quote_at = row.get("total_quote_at")
        badge = row.get("total_badge")
    else:
        side = row.get("quant_best_ml_side")
        edge = row.get("quant_best_ml_edge_pp")
        if side == home:
            probability = row.get("calibrated_home_probability")
            odds = row.get("best_home_ml")
            book = row.get("best_home_ml_book")
            quote_at = row.get("best_home_ml_quote_at")
        elif side == away and _finite(row.get("calibrated_home_probability")):
            probability = 1.0 - float(row.get("calibrated_home_probability"))
            odds = row.get("best_away_ml")
            book = row.get("best_away_ml_book")
            quote_at = row.get("best_away_ml_quote_at")
        else:
            probability = odds = book = quote_at = None
        line = odds
        badge = row.get("ml_badge")

    if not (
        _finite(edge)
        and _finite(probability)
        and _finite(line)
        and _finite(odds)
        and str(book or "").strip()
        and str(quote_at or "").strip()
    ):
        return None
    p = float(probability)
    price = float(odds)
    return {
        "market": market,
        "side": side,
        "line": float(line),
        "odds": price,
        "probability": p,
        "edge": abs(float(edge)),
        "ev": float(roi(p, price)),
        "badge": str(badge or ""),
        "book": book,
        "quote_at": quote_at,
    }


def current_edge_board(frame, report, statuses=None, include_contraindicated=False):
    """Return executable current candidates in selected parent regimes.

    Persistent parent bands are filtered through their broad role/location child
    evidence. Contraindicated children are excluded by default so the supported-edge
    board cannot present a repeatedly losing subtype as historically supported.
    """

    statuses = set(statuses or {"PERSISTENT_CANDIDATE"})
    rows = []
    for _, game in frame.iterrows():
        for market in ("spread", "total", "moneyline"):
            candidate = _market_candidate(game, market)
            if not candidate:
                continue
            regime = match_edge_regime(report, market, candidate["edge"])
            if not regime or regime.get("status") not in statuses:
                continue
            subgroup = match_edge_subgroup(
                report,
                market,
                candidate["edge"],
                candidate["side"],
                candidate["line"],
                game.get("home_team"),
                game.get("away_team"),
            )
            effective = effective_edge_status(regime, subgroup)
            if effective == "CONTRAINDICATED_SUBGROUP" and not include_contraindicated:
                continue
            evidence_stats = subgroup if effective == "SUPPORTED_SUBGROUP" else regime
            price_info = price_evidence(candidate["odds"], evidence_stats)
            rows.append(
                {
                    "game_id": game.get("game_id"),
                    "date": game.get("date"),
                    "away_team": game.get("away_team"),
                    "home_team": game.get("home_team"),
                    **candidate,
                    "regime_status": regime.get("status"),
                    "edge_reliability_status": effective,
                    "regime_band": regime.get("edge_band"),
                    "historical_bets": regime.get("bets"),
                    "historical_win_rate": regime.get("win_rate"),
                    "historical_roi": regime.get("roi"),
                    "historical_avg_clv": regime.get("avg_clv"),
                    "profitable_seasons": regime.get("profitable_seasons"),
                    "season_count": regime.get("season_count"),
                    "subgroup_key": (subgroup or {}).get("key", ""),
                    "subgroup_status": (subgroup or {}).get(
                        "status", "INCONCLUSIVE_SUBGROUP"
                    ),
                    "subgroup_bets": (subgroup or {}).get("bets", 0),
                    "subgroup_win_rate": (subgroup or {}).get("win_rate"),
                    "subgroup_roi": (subgroup or {}).get("roi"),
                    "subgroup_avg_clv": (subgroup or {}).get("avg_clv"),
                    "subgroup_profitable_seasons": (subgroup or {}).get(
                        "profitable_seasons", 0
                    ),
                    "subgroup_season_count": (subgroup or {}).get("season_count", 0),
                    "subgroup_validation_design": (subgroup or {}).get(
                        "validation_design", ""
                    ),
                    "subgroup_discovery_seasons": "|".join(
                        str(value)
                        for value in (subgroup or {}).get("discovery_seasons", [])
                    ),
                    "subgroup_discovery_bets": (subgroup or {}).get(
                        "discovery_bets", 0
                    ),
                    "subgroup_discovery_roi": (subgroup or {}).get("discovery_roi"),
                    "subgroup_holdout_season": (subgroup or {}).get(
                        "holdout_season", ""
                    ),
                    "subgroup_holdout_bets": (subgroup or {}).get("holdout_bets", 0),
                    "subgroup_holdout_roi": (subgroup or {}).get("holdout_roi"),
                    "subgroup_holdout_win_rate": (subgroup or {}).get(
                        "holdout_win_rate"
                    ),
                    "subgroup_holdout_confirmed": bool(
                        (subgroup or {}).get("holdout_confirmed", False)
                    ),
                    "currently_selected": (
                        str(game.get("quant_market") or "").lower() == market
                        and str(game.get("quant_side") or "") == str(candidate["side"])
                    ),
                    "selected_quant_signal": game.get("quant_signal"),
                    "selected_quant_ev": game.get("quant_ev"),
                    **price_info,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        rank = {
            "SUPPORTED_SUBGROUP": 0,
            "PERSISTENT_PARENT_ONLY": 1,
            "WATCH": 2,
            "CONTRAINDICATED_SUBGROUP": 9,
        }
        price_rank = {"CONFIRMED": 0, "PLAUSIBLE": 1, "OVERPRICED": 2, "UNKNOWN": 3}
        out["_reliability_rank"] = out["edge_reliability_status"].map(rank).fillna(5)
        out["_price_rank"] = out["price_evidence_status"].map(price_rank).fillna(4)
        out = out.sort_values(
            ["_reliability_rank", "_price_rank", "historical_roi", "ev"],
            ascending=[True, True, False, False],
        ).drop(columns=["_reliability_rank", "_price_rank"]).reset_index(drop=True)
    return out

