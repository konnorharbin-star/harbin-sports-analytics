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
