"""Regression tests for fail-closed archived entry verification."""
from __future__ import annotations

import pandas as pd

from harbin.entry_provenance import is_explicitly_verified, verified_entry_mask
from harbin.policy import _promotion_sample as policy_promotion_sample
from harbin.proof import _promotion_sample as evidence_promotion_sample
from harbin.proof import build_evidence_report


def test_false_strings_never_become_verified_bets():
    data = pd.DataFrame({
        "entry_quote_verified": [
            "False", "FALSE", "false", "0", "unknown", "", None,
            False, 0, 0.0, float("nan"), "True", " TRUE ", "1", True, 1, 1.0,
        ]
    }, index=list(range(100, 117)))
    mask = verified_entry_mask(data)
    assert mask.index.equals(data.index)
    assert mask.tolist() == [False] * 11 + [True] * 6
    assert is_explicitly_verified([]) is False
    assert is_explicitly_verified("yes") is False
    assert is_explicitly_verified("verified") is False


def test_legacy_opening_fields_cannot_promote_without_explicit_verification():
    legacy = pd.DataFrame({
        "used_distinct_open": [True, "True", 1],
        "profit": [0.5, 0.5, 0.5],
    })
    assert policy_promotion_sample(legacy).empty
    evidence, provenance = evidence_promotion_sample(legacy)
    assert evidence.empty
    assert provenance == "unverified"


def test_policy_and_evidence_share_identical_verification_mask():
    data = pd.DataFrame({
        "entry_quote_verified": ["False", "True", None, False, True, "0"],
        "used_distinct_open": [True] * 6,
        "profit": [1.0] * 6,
        "market": ["spread"] * 6,
    })
    selected = policy_promotion_sample(data)
    evidence, provenance = evidence_promotion_sample(data)
    assert selected.index.tolist() == [1, 4]
    assert evidence.index.tolist() == [1, 4]
    assert provenance == "entry_quote_verified"


def test_evidence_report_never_counts_string_false_as_verified(tmp_path):
    bets = pd.DataFrame([
        {"game_id": "1", "season": 2024, "week": 1, "market": "spread",
         "signal": "BET", "profit": 0.9, "result": 1, "clv": 0.1,
         "entry_quote_verified": "False"},
        {"game_id": "2", "season": 2024, "week": 1, "market": "spread",
         "signal": "BET", "profit": 0.9, "result": 1, "clv": 0.1,
         "entry_quote_verified": "unverified"},
        {"game_id": "3", "season": 2024, "week": 1, "market": "spread",
         "signal": "BET", "profit": -1, "result": -1, "clv": -0.1,
         "entry_quote_verified": "True"},
    ])
    bp = tmp_path / "bets.csv"
    bets.to_csv(bp, index=False)
    report = build_evidence_report(tmp_path / "missing.json", bp, tmp_path / "evidence.json")
    assert report["promotion_sample"]["verified_bets"] == 1
    assert report["promotion_sample"]["excluded_unverified_bets"] == 2
    assert report["overall"]["bets"] == 1
    assert report["overall"]["roi"] == -1
    assert report["status"] == "UNPROVEN"


def test_archive_opening_pair_requires_real_side_and_same_book(monkeypatch):
    from types import SimpleNamespace
    from harbin.backtest_audit import AuditedArchiveMarketStore
    from harbin.backtest_runtime import CanonicalArchiveMarketStore

    game = SimpleNamespace(game_id="10", home_team="Home", away_team="Away")

    def fake_base_quote(_store, _game):
        return {"book": "BookA"}

    monkeypatch.setattr(CanonicalArchiveMarketStore, "quote", fake_base_quote)
    store = object.__new__(AuditedArchiveMarketStore)
    store.by_id = {
        "10": pd.DataFrame([
            # Two opening rows for Home do NOT mean both sides are available.
            {"book": "BookA", "market_type": "moneyline", "abbr": "Home",
             "opening_odds": -160, "opening_lines": None},
            {"book": "BookA", "market_type": "moneyline", "abbr": "Home",
             "opening_odds": -150, "opening_lines": None},
            {"book": "BookA", "market_type": "spread", "abbr": "Home",
             "opening_odds": -110, "opening_lines": -3.5},
            {"book": "BookA", "market_type": "spread", "abbr": "Away",
             "opening_odds": None, "opening_lines": 3.5},
            {"book": "BookA", "market_type": "total", "abbr": "Over",
             "opening_odds": -110, "opening_lines": 51.5},
            {"book": "BookA", "market_type": "total", "abbr": "Over",
             "opening_odds": -115, "opening_lines": 51.5},
            # Another sportsbook cannot fill a missing selected-book quote.
            {"book": "BookB", "market_type": "moneyline", "abbr": "Away",
             "opening_odds": 135, "opening_lines": None},
            {"book": "BookB", "market_type": "total", "abbr": "Under",
             "opening_odds": -105, "opening_lines": 51.5},
        ])
    }
    quote = store.quote(game)
    assert quote["open_moneyline_verified"] is False
    assert quote["open_spread_verified"] is False
    assert quote["open_total_verified"] is False

    extra = pd.DataFrame([
        {"book": "BookA", "market_type": "moneyline", "abbr": "Away",
         "opening_odds": 135, "opening_lines": None},
        {"book": "BookA", "market_type": "spread", "abbr": "Away",
         "opening_odds": -110, "opening_lines": 3.5},
        {"book": "BookA", "market_type": "total", "abbr": "Under",
         "opening_odds": -105, "opening_lines": 51.5},
    ])
    store.by_id["10"] = pd.concat([store.by_id["10"], extra], ignore_index=True)
    quote = store.quote(game)
    assert quote["open_moneyline_verified"] is True
    assert quote["open_spread_verified"] is True
    assert quote["open_total_verified"] is True


def test_verification_breakdown_uses_safe_flags_by_market_and_season():
    from harbin.backtest_audit import _verification_breakdown

    bets = pd.DataFrame({
        "market": ["spread", "spread", "moneyline", "total"],
        "season": [2024, 2024, 2025, 2025],
        "entry_quote_verified": ["False", "True", "False", None],
    })
    by_market = _verification_breakdown(bets, "market")
    assert by_market["spread"] == {
        "archive_bets": 2, "verified_opening_entry_bets": 1,
        "excluded_unverified_bets": 1,
    }
    assert by_market["moneyline"]["verified_opening_entry_bets"] == 0
    assert by_market["total"]["excluded_unverified_bets"] == 1
    by_season = _verification_breakdown(bets, "season")
    assert by_season["2024"]["verified_opening_entry_bets"] == 1
    assert by_season["2025"]["excluded_unverified_bets"] == 2
