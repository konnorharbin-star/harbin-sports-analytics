import pandas as pd

from harbin.edge_priority import (
    actionable_priority_edges,
    classify_edge_priority,
    enrich_edge_priority,
)


def _row(**overrides):
    row = {
        "market": "spread",
        "edge_reliability_status": "SUPPORTED_SUBGROUP",
        "price_evidence_status": "CONFIRMED",
        "regime_band": "6-8",
        "line": -13.0,
        "edge": 7.9,
        "conservative_price_margin": 0.08,
        "ev": 0.25,
    }
    row.update(overrides)
    return row


def test_robust_core_requires_price_and_line_room():
    robust = _row()
    price_thin = _row(conservative_price_margin=0.01)
    line_thin = _row(edge=6.25)
    both_thin = _row(edge=6.25, conservative_price_margin=0.01)

    assert classify_edge_priority(robust) == "ROBUST_CORE"
    assert classify_edge_priority(price_thin) == "CORE_PRICE_THIN"
    assert classify_edge_priority(line_thin) == "CORE_LINE_THIN"
    assert classify_edge_priority(both_thin) == "CORE_BOTH_THIN"


def test_priority_rejects_overpriced_and_does_not_promote_parent_only():
    assert (
        classify_edge_priority(
            _row(price_evidence_status="OVERPRICED")
        )
        == "REJECT_OVERPRICED"
    )
    assert (
        classify_edge_priority(
            _row(edge_reliability_status="PERSISTENT_PARENT_ONLY")
        )
        == "PARENT_ONLY"
    )
    assert (
        classify_edge_priority(
            _row(edge_reliability_status="CONTRAINDICATED_SUBGROUP")
        )
        == "REJECT_CONTRAINDICATED"
    )


def test_spread_bet_to_line_never_drops_below_supported_minimum():
    frame = pd.DataFrame(
        [
            _row(line=10.5, edge=7.840656546559058),
            _row(line=-13.0, edge=7.899162035221796),
            _row(line=-2.5, edge=6.266374760164988),
        ]
    )

    ranked = enrich_edge_priority(frame)

    by_line = {float(row.line): row for _, row in ranked.iterrows()}
    assert by_line[10.5].bet_to_line == 9.0
    assert by_line[-13.0].bet_to_line == -14.5
    assert by_line[-2.5].bet_to_line == -2.5

    for _, row in ranked.iterrows():
        # A selected-side spread becomes worse as its numeric line moves lower.
        remaining_edge = float(row.edge) - (float(row.line) - float(row.bet_to_line))
        assert remaining_edge >= 6.0 - 1e-12


def test_actionable_edges_exclude_thin_and_plausible_prices():
    frame = pd.DataFrame(
        [
            _row(edge=7.9, conservative_price_margin=0.08, ev=0.25),
            _row(line=-9.5, edge=6.8, conservative_price_margin=0.03, ev=0.20),
            _row(line=10.5, edge=7.8, conservative_price_margin=0.01, ev=0.30),
            _row(
                line=7.5,
                edge=7.8,
                conservative_price_margin=-0.02,
                price_evidence_status="PLAUSIBLE",
                ev=0.28,
            ),
        ]
    )

    actionable = actionable_priority_edges(frame)

    assert list(actionable["edge_priority"]) == ["ROBUST_CORE", "CORE"]
    assert len(actionable) == 2


def test_priority_sorting_prefers_price_and_line_cushion_not_raw_ev():
    frame = pd.DataFrame(
        [
            _row(
                line=10.5,
                edge=7.8,
                conservative_price_margin=0.01,
                ev=0.40,
            ),
            _row(
                line=-13.0,
                edge=7.9,
                conservative_price_margin=0.08,
                ev=0.20,
            ),
        ]
    )

    ranked = enrich_edge_priority(frame)

    assert ranked.iloc[0]["edge_priority"] == "ROBUST_CORE"
    assert ranked.iloc[0]["ev"] == 0.20
    assert ranked.iloc[1]["edge_priority"] == "CORE_PRICE_THIN"
