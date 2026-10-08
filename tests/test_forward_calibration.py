"""Descriptive, independent-score-only reliability tests."""
import math

from platform_ops.forward_calibration import (
    MIN_ECE_SAMPLE, forward_calibration
)


def item(p, result, *, verified=True, market="spread", game="g1", league="NFL"):
    return {
        "model_probability": p, "result": result,
        "independent_score_verified": verified,
        "game_id": game, "league": league, "market": market,
    }


def test_unverified_results_never_enter_calibration():
    metrics = forward_calibration([item(.95, "WIN", verified=False)])
    overall = metrics["results"]["NFL"]["all_markets"]
    assert overall["graded_probabilities"] == 0
    assert overall["status"] == "NO_VERIFIED_SAMPLE"
    assert overall["brier"] is None


def test_binary_brier_and_log_loss_are_correct():
    metrics = forward_calibration([
        item(.8, "WIN"), item(.8, "LOSS", game="g2")
    ])
    stats = metrics["results"]["NFL"]["all_markets"]
    assert stats["graded_probabilities"] == 2
    assert math.isclose(stats["brier"], (.04+.64)/2)
    assert math.isclose(stats["log_loss"],
                        (-math.log(.8)-math.log(.2))/2)
    assert stats["status"] == "INSUFFICIENT_SAMPLE"
    assert stats["calibration_ece"] is None
    assert not stats["profitability_proven"]


def test_pushes_invalid_probabilities_and_bad_results_ignored():
    scores = forward_calibration([
        item(.6, "PUSH"), item(None, "WIN"), item(1.4, "WIN"),
        item(float("nan"), "WIN"), item(.6, "UNSETTLED"),
    ])
    stats = scores["results"]["NFL"]["all_markets"]
    assert stats["graded_probabilities"] == 0
    assert stats["omitted_pushes"] == 1
    assert stats["brier"] is None


def test_grouping_keeps_nfl_cfb_and_market_separate():
    metrics = forward_calibration([
        item(.7, "WIN"), item(.5, "LOSS", market="total", game="g2"),
        item(.55, "WIN", league="CFB", market="moneyline", game="g3"),
    ])
    nfl = metrics["results"]["NFL"]
    assert nfl["all_markets"]["graded_probabilities"] == 2
    assert nfl["by_market"]["spread"]["graded_probabilities"] == 1
    assert nfl["by_market"]["total"]["graded_probabilities"] == 1
    assert nfl["by_market"]["moneyline"]["graded_probabilities"] == 0
    assert metrics["results"]["CFB"]["all_markets"]["graded_probabilities"] == 1


def test_ece_does_not_appear_until_sufficient_independent_sample():
    sample = [item(.55, "WIN" if i % 2 else "LOSS", game=f"g{i}")
              for i in range(MIN_ECE_SAMPLE)]
    report = forward_calibration(sample)
    stats = report["results"]["NFL"]["all_markets"]
    assert stats["graded_probabilities"] == MIN_ECE_SAMPLE
    assert stats["distinct_games"] == MIN_ECE_SAMPLE
    assert stats["status"] == "DESCRIPTIVE_ONLY"
    assert math.isclose(stats["calibration_ece"], .05, abs_tol=1e-10)
    assert len(stats["reliability_bins"]) == 1
    assert stats["reliability_bins"][0]["count"] == MIN_ECE_SAMPLE
    assert stats["profitability_proven"] is False


def test_report_outputs_never_claim_brier_proves_market_edge():
    report = forward_calibration([])
    assert "NOT a verified profitable edge" in report["limitations"]
    assert "correlated" in report["limitations"]
