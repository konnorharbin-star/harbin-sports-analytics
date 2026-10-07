from harbin.edge_price import price_evidence, wilson_lower_bound


def test_wilson_lower_bound_is_conservative():
    lower = wilson_lower_bound(56, 33)

    assert 0.52 < lower < 0.63


def test_price_evidence_confirms_supported_price_when_lower_bound_clears_break_even():
    result = price_evidence(-108, {"wins": 56, "losses": 33})

    assert result["price_evidence_status"] == "CONFIRMED"
    assert result["historical_price_win_rate"] > result["current_break_even_probability"]
    assert result["historical_price_wilson_lower"] > result["current_break_even_probability"]
    assert result["conservative_price_margin"] > 0


def test_price_evidence_marks_expensive_price_plausible_not_confirmed():
    result = price_evidence(-122, {"wins": 56, "losses": 33})

    assert result["price_evidence_status"] == "PLAUSIBLE"
    assert result["historical_price_margin"] > 0
    assert result["conservative_price_margin"] <= 0


def test_price_evidence_rejects_price_above_historical_point_estimate():
    result = price_evidence(-200, {"wins": 56, "losses": 33})

    assert result["price_evidence_status"] == "OVERPRICED"
    assert result["historical_price_margin"] < 0


def test_price_evidence_returns_unknown_without_sample():
    result = price_evidence(-110, {"wins": 0, "losses": 0})

    assert result["price_evidence_status"] == "UNKNOWN"
    assert result["historical_price_wilson_lower"] is None
