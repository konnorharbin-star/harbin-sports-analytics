"""Free, strict same-book pregame line-movement research.

This is NOT a sportsbook closing-line feed or betting execution. It reads the
existing public model market snapshots and archived first-seen research signals.
Only same-book, same-game, same-side, recently updated pre-kickoff observations
are accepted. Incomplete/misaligned market evidence is excluded, not imputed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import io
import json
import math
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from platform_ops.free_observer import (
    SOURCE_REPOS, american_profit, finite_number, utc_datetime
)
from platform_ops.grade_observations import collect_archived_candidates

HISTORY_PATH = "history/market_snapshots.csv"
MAX_SOURCE_BYTES = 30_000_000
# Not the official closing bell. Classify only a recent PREKICKOFF observation.
MAX_CLOSE_AGE = timedelta(minutes=90)
MAX_CAPTURE_QUOTE_AGE = timedelta(minutes=90)
MAX_FUTURE_SKEW = timedelta(minutes=2)
MIN_CAPTURE_AFTER_ENTRY = timedelta(seconds=1)


def _book(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _side(market: str, side: object, home: str, away: str) -> str | None:
    value = str(side or "").strip().casefold()
    if market == "total":
        return {"o": "over", "u": "under", "over": "over", "under": "under"}.get(value)
    if value in {"home", str(home).strip().casefold()}:
        return "home"
    if value in {"away", str(away).strip().casefold()}:
        return "away"
    return None


def _acceptable_american(value: object) -> float | None:
    number = finite_number(value)
    return number if american_profit(number) is not None else None


def _normalized_close(
    league: str, game: str, market: str, role: str,
    book: object, provider: object, line: object, odds: object,
    captured: object, quote_at: object, kickoff: object,
    *,
    timestamp_kind: str,
) -> dict[str, Any] | None:
    captured_dt = utc_datetime(captured)
    quote_dt = utc_datetime(quote_at)
    kickoff_dt = utc_datetime(kickoff)
    price = _acceptable_american(odds)
    value = finite_number(line)
    if (
        captured_dt is None or quote_dt is None or kickoff_dt is None
        or not _book(book) or price is None
        or market not in {"moneyline", "spread", "total"}
        or role not in {"home", "away", "over", "under"}
        or (market != "moneyline" and value is None)
        or captured_dt >= kickoff_dt or quote_dt >= kickoff_dt
        or quote_dt > captured_dt + MAX_FUTURE_SKEW
        or captured_dt - quote_dt > MAX_CAPTURE_QUOTE_AGE
        or kickoff_dt - captured_dt > MAX_CLOSE_AGE
    ):
        return None
    if market == "moneyline":
        value = None
    return {
        "league": league, "game_id": game, "market": market, "role": role,
        "book": str(book), "provider": str(provider or ""),
        "line": value, "odds": price,
        "captured_at_utc": captured_dt.isoformat(),
        "source_quote_at_utc": quote_dt.isoformat(),
        "kickoff_utc": kickoff_dt.isoformat(),
        "time_provenance": timestamp_kind,
    }


def parse_cfb_market_rows(source: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reconstruct selected prices for individual named providers, never consensus."""
    quotes = []
    rejected = 0
    bad_json = 0
    for row in csv.DictReader(io.StringIO(source)):
        game = str(row.get("game_id") or "").strip()
        captured = row.get("captured_at")
        kickoff = row.get("kickoff")
        if not game or utc_datetime(kickoff) is None:
            rejected += 1
            continue
        raw = row.get("market_quotes_json")
        if not raw:
            rejected += 1
            continue
        try:
            providers = json.loads(raw)
        except (ValueError, TypeError):
            bad_json += 1
            continue
        if not isinstance(providers, list):
            bad_json += 1
            continue
        for provider in providers:
            if not isinstance(provider, dict):
                rejected += 1
                continue
            book = provider.get("provider")
            source_type = str(provider.get("source") or "").casefold()
            # The collector is free-only: historical optional paid API rows
            # must NOT become trusted evidence in a free recommendation system.
            if source_type not in ("action_network", "espn_core", "primary"):
                rejected += 1
                continue
            updated = provider.get("last_update")
            # Strict: a synthetic captured_at fallback is NOT source evidence.
            if utc_datetime(updated) is None:
                rejected += 1
                continue
            for market, role, line, odds in (
                ("moneyline", "home", None, provider.get("home_ml")),
                ("moneyline", "away", None, provider.get("away_ml")),
                ("spread", "home", provider.get("home_spread"),
                 provider.get("home_spread_price")),
                ("spread", "away",
                 -float(provider["home_spread"])
                 if finite_number(provider.get("home_spread")) is not None else None,
                 provider.get("away_spread_price")),
                ("total", "over", provider.get("market_total"), provider.get("over_price")),
                ("total", "under", provider.get("market_total"), provider.get("under_price")),
            ):
                converted = _normalized_close(
                    "CFB", game, market, role, book, source_type, line, odds,
                    captured, updated, kickoff, timestamp_kind="source_last_update",
                )
                if converted:
                    quotes.append(converted)
    return quotes, {
        "accepted_side_quotes": len(quotes),
        "malformed_quote_json": bad_json,
        "rejected_or_incomplete_source_rows": rejected,
    }


def parse_nfl_market_rows(source: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Use NFL verified two-way snapshots; no mixing of provider-side records."""
    quotes = []
    rejected = 0
    snapshot_sides: dict[tuple[str, str, str, str, str], set[str]] = {}
    preliminary: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    for row in csv.DictReader(io.StringIO(source)):
        market = str(row.get("market_type") or "").strip().casefold()
        game = str(row.get("game_id") or "").strip()
        provider = str(row.get("provider") or "").strip()
        book = str(row.get("book") or "").strip()
        event_id = str(row.get("source_event_id") or "").strip()
        captured = row.get("captured_at")
        kickoff = row.get("kickoff")
        if not all((game, provider, book, event_id, str(captured or ""), str(kickoff or ""))):
            rejected += 1
            continue
        snap_id = str(row.get("snapshot_id") or "")
        if not snap_id or market not in {"moneyline", "spread", "total"}:
            rejected += 1
            continue
        first = _side(market, row.get("first_side"), "home", "away")
        second = _side(market, row.get("second_side"), "home", "away")
        if (first is None or second is None
            or {first, second} != ({"over", "under"} if market == "total" else {"home", "away"})):
            rejected += 1
            continue
        key = (game, _book(book), provider.casefold(), market, snap_id)
        outputs = []
        for side, line, odds in (
            (first, row.get("first_line"), row.get("first_american_odds")),
            (second, row.get("second_line"), row.get("second_american_odds")),
        ):
            converted = _normalized_close(
                "NFL", game, market, side, book, provider,
                line, odds, captured, captured, kickoff,
                timestamp_kind="public_capture_only",
            )
            if converted is None:
                break
            outputs.append(converted)
        if len(outputs) != 2:
            rejected += 1
            continue
        snapshot_sides[key] = {first, second}
        preliminary.extend((key, item) for item in outputs)
    for key, item in preliminary:
        if len(snapshot_sides.get(key, ())) == 2:
            quotes.append(item)
    return quotes, {
        "accepted_side_quotes": len(quotes),
        "rejected_or_incomplete_source_rows": rejected,
    }


def select_later_same_book(
    candidate: dict[str, Any], closes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare original side, book, and price to an actual later pregame snapshot."""
    market = str(candidate.get("market") or "").lower()
    role = _side(
        market, candidate.get("side"), str(candidate.get("home_team") or ""),
        str(candidate.get("away_team") or ""),
    )
    entry_price = _acceptable_american(candidate.get("odds"))
    kickoff = utc_datetime(candidate.get("kickoff_utc"))
    observed = utc_datetime(candidate.get("observed_at_utc"))
    quoted = utc_datetime(candidate.get("quoted_at_utc"))
    original_line = finite_number(candidate.get("line"))
    book = candidate.get("book")
    base = {
        "league": candidate.get("league"), "game_id": candidate.get("game_id"),
        "market": market, "side": candidate.get("side"), "book": book,
        "entry_line": original_line, "entry_american_odds": entry_price,
        "entry_observed_at_utc": candidate.get("observed_at_utc"),
        "entry_quote_at_utc": candidate.get("quoted_at_utc"),
        "original_research_blockers": candidate.get("blockers", []),
        "source_contract_status": candidate.get("source_contract_status"),
        "source_reconciliation_status": candidate.get("source_reconciliation_status"),
        "status": "NO_VERIFIABLE_LATER_SAME_BOOK_SNAPSHOT",
        "closing_line": None, "closing_american_odds": None,
        "closing_captured_at_utc": None, "source_quote_at_utc": None,
        "line_advantage_points": None, "same_line_implied_probability_move": None,
        "official_book_closing_verified": False,
        "wager_executed": False, "price_executability_verified": False,
    }
    if (
        role is None or kickoff is None or observed is None or quoted is None
        or quoted >= kickoff or observed >= kickoff
        or quoted > observed + MAX_FUTURE_SKEW
        or entry_price is None or not _book(book)
        or (market != "moneyline" and original_line is None)
    ):
        base["status"] = "INVALID_OR_UNTIMED_ORIGINAL_QUOTE"
        return base
    if (
        candidate.get("source_contract_status") != "PASS"
        or candidate.get("source_reconciliation_status") != "PASS"
        or candidate.get("source_audit_status") in ("FAIL", "ERROR", "UNKNOWN")
    ):
        base["status"] = "ORIGINAL_PUBLICATION_CONTRACT_UNVERIFIED"
        return base
    valid = []
    for close in closes:
        if (
            close["league"] != candidate.get("league")
            or str(close["game_id"]) != str(candidate.get("game_id"))
            or close["market"] != market or close["role"] != role
            or _book(close["book"]) != _book(book)
            or (close_kickoff := utc_datetime(close.get("kickoff_utc"))) is None
            or abs(close_kickoff - kickoff) > timedelta(minutes=5)
        ):
            continue
        captured = utc_datetime(close["captured_at_utc"])
        source_dt = utc_datetime(close["source_quote_at_utc"])
        if (captured is None or source_dt is None
            or captured <= observed + MIN_CAPTURE_AFTER_ENTRY
            or source_dt <= quoted or source_dt <= observed
            or captured >= kickoff or source_dt >= kickoff
            or kickoff - captured > MAX_CLOSE_AGE
            or kickoff - source_dt > MAX_CLOSE_AGE
        ):
            continue
        valid.append(close)
    if not valid:
        return base
    valid.sort(key=lambda row:(row["captured_at_utc"],row["source_quote_at_utc"]))
    last = valid[-1]
    # Conflicting same-book quotes at the same effective latest observation
    # must not silently be resolved in favor of a favorable result.
    latest_at = (last["captured_at_utc"], last["source_quote_at_utc"])
    simultaneous = [
        x for x in valid
        if (x["captured_at_utc"], x["source_quote_at_utc"]) == latest_at
    ]
    prices = {(x["line"], x["odds"], x["provider"]) for x in simultaneous}
    if len(prices) != 1:
        base["status"] = "AMBIGUOUS_LATEST_SAME_BOOK_QUOTES"
        return base
    base.update({
        "status": "OBSERVED_SAME_BOOK_NEAR_KICKOFF",
        "closing_line": last["line"],
        "closing_american_odds": last["odds"],
        "closing_captured_at_utc": last["captured_at_utc"],
        "source_quote_at_utc": last["source_quote_at_utc"],
        "closing_provider": last["provider"],
        "time_provenance": last["time_provenance"],
    })
    if market == "moneyline":
        base["same_line_implied_probability_move"] = (
            (1/(1+american_profit(last["odds"]))) -
            (1/(1+american_profit(entry_price)))
        )
    else:
        base["line_advantage_points"] = (
            original_line-last["line"] if market == "spread"
            else (last["line"]-original_line if role == "over" else
                  original_line-last["line"])
        )
        if math.isclose(original_line, last["line"], abs_tol=1e-9):
            base["same_line_implied_probability_move"] = (
                (1/(1+american_profit(last["odds"]))) -
                (1/(1+american_profit(entry_price)))
            )
    # Public captured odds are observations, not actual wager fills or a
    # sportsbook-certified final closing price. Claim neither.
    return base


def latest_movement_report(
    candidates: list[dict[str, Any]],
    closing_quotes: dict[str, list[dict[str, Any]]],
    *,
    parse_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    for cand in candidates:
        league = cand.get("league")
        rows.append(select_later_same_book(cand, closing_quotes.get(league, [])))
    rows.sort(key=lambda x:(str(x["league"]),str(x["game_id"]),str(x["market"])))
    matched = [x for x in rows if x["status"]=="OBSERVED_SAME_BOOK_NEAR_KICKOFF"]
    return {
        "schema_version": 1, "mode": "READ_ONLY_SAME_BOOK_LINE_MOVEMENT",
        "automatic_betting_enabled": False, "paid_sources_used": False,
        "official_closing_prices_verified": False,
        "entry_execution_prices_verified": False,
        "summary": {
            "frozen_candidates": len(rows),
            "same_book_near_kickoff_observations": len(matched),
            "unverified_or_unmatched": len(rows)-len(matched),
            "profitable_edge_proven": False,
        },
        "limitations": (
            "Historical model-reported entry quotes may not have been executable; "
            "same-book public snapshots within 90 minutes of kickoff are NOT "
            "bookmaker-certified closing prices. Movements across different point "
            "lines are reported in points; implied probabilities are compared "
            "only at the SAME point line. No sharp-consensus or vig-adjusted CLV "
            "can be inferred from these single-side observations."
        ),
        "source_diagnostics": parse_diagnostics or {},
        "comparisons": rows,
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Harbin free same-book line movement audit",
        "",
        "**No automatic bets, no paid data, no official closing-line certification.**",
        "",
        f"Frozen research candidates: {summary['frozen_candidates']}.",
        f"Near-kickoff same-book observations: {summary['same_book_near_kickoff_observations']}.",
        f"Unavailable or invalid: {summary['unverified_or_unmatched']}.",
        "",
        "| League | Game | Market / side | Book | Entry | Later prekickoff | Line advantage | Status |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    for row in report["comparisons"][:100]:
        def safe(val):
            return str(val if val is not None else "—").replace("|","\\|").replace("\n"," ")
        advantage = row["line_advantage_points"]
        change = "—" if advantage is None else f"{advantage:+g} points"
        entry = f"{safe(row['entry_line'])} / {safe(row['entry_american_odds'])}"
        later = f"{safe(row['closing_line'])} / {safe(row['closing_american_odds'])}"
        lines.append(
            f"| {safe(row['league'])} | {safe(row['game_id'])} | "
            f"{safe(row['market'])} {safe(row['side'])} | {safe(row['book'])} | "
            f"{entry} | {later} | {change} | {safe(row['status'])} |"
        )
    lines.extend([
        "",
        "No unmatched row is given a fabricated closing value. The report measures "
        "observed same-book prekickoff movement, **not official closing-line value**, "
        "and cannot establish profitability or betting authorization.",
    ])
    return "\n".join(lines)+"\n"


def free_market_file(league: str) -> str:
    """Retrieve free, already-public model history, bounded and read-only."""
    if league not in SOURCE_REPOS:
        raise ValueError("Unsupported league")
    url = (
        f"https://raw.githubusercontent.com/{SOURCE_REPOS[league]}/main/"
        f"{HISTORY_PATH}"
    )
    request = Request(url, headers={
        "User-Agent": "HarbinFreeQuoteResearch/1.0", "Accept": "text/csv",
    }, method="GET")
    with urlopen(request, timeout=30) as response:  # noqa: S310 - constant GitHub URL
        body = response.read(MAX_SOURCE_BYTES+1)
    if len(body)>MAX_SOURCE_BYTES:
        raise ValueError(f"{league}: market history exceeds bounded free collector size")
    return body.decode("utf-8-sig")


def main(argv: list[str] | None = None) -> int:
    p=argparse.ArgumentParser(description="Free read-only same-book pregame odds research")
    p.add_argument("--archive-root",type=Path,required=True)
    p.add_argument("--json-out",type=Path,required=True)
    p.add_argument("--markdown-out",type=Path,required=True)
    args=p.parse_args(argv)
    candidates,counters=collect_archived_candidates(args.archive_root)
    quotes={}
    diagnostics={"first_seen_archive":counters}
    for league in ("NFL","CFB"):
        raw=free_market_file(league)
        quotes[league], stats = (
            parse_nfl_market_rows(raw) if league=="NFL"
            else parse_cfb_market_rows(raw)
        )
        diagnostics[league]=stats
    report=latest_movement_report(candidates,quotes,parse_diagnostics=diagnostics)
    args.json_out.parent.mkdir(parents=True,exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True,exist_ok=True)
    args.json_out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    args.markdown_out.write_text(markdown_report(report))
    print(json.dumps(report["summary"],sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
