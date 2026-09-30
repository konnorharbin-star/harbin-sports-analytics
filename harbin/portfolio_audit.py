from __future__ import annotations

import math

import pandas as pd

from . import portfolio as _base

_ORIGINAL_EXECUTION_ISSUE = _base._execution_issue


def _audited_execution_issue(row, limits):
    issue = _ORIGINAL_EXECUTION_ISSUE(row, limits)
    if issue:
        return issue

    min_books = max(1, int(_base._num(limits.get("min_market_book_count_for_execution", 1), 1)))
    if not _base._finite(row.get("market_book_count")):
        return "missing market book count"
    if int(float(row.get("market_book_count"))) < min_books:
        return f"market book count below {min_books}"

    if not bool(limits.get("require_quote_timestamp_for_execution", True)):
        return ""
    quote = pd.to_datetime(row.get("quant_quote_at"), utc=True, errors="coerce")
    kickoff = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
    if pd.isna(quote):
        return "missing executable quote timestamp"
    if pd.isna(kickoff):
        return "missing kickoff timestamp for quote validation"
    if quote >= kickoff:
        return "executable quote is not strictly pre-kickoff"

    now = pd.Timestamp.now(tz="UTC")
    if quote > now + pd.Timedelta(minutes=5):
        return "executable quote timestamp is in the future"
    max_age = max(1.0, _base._num(limits.get("max_quote_age_minutes", 60), 60))
    age_minutes = (now - quote).total_seconds() / 60.0
    if age_minutes > max_age:
        return f"executable quote is stale ({age_minutes:.1f}m > {max_age:.0f}m)"
    return ""


def install_execution_audit():
    _base._execution_issue = _audited_execution_issue


def apply_portfolio_controls(*args, **kwargs):
    install_execution_audit()
    return _base.apply_portfolio_controls(*args, **kwargs)


def build_bankroll_risk_state(*args, **kwargs):
    return _base.build_bankroll_risk_state(*args, **kwargs)


def write_portfolio_outputs(*args, **kwargs):
    install_execution_audit()
    return _base.write_portfolio_outputs(*args, **kwargs)
