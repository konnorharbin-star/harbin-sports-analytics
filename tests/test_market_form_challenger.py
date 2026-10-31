"""Point-in-time information and holdout purity tests for CFB form research."""
import csv

import numpy as np
import pytest

from scripts import market_form_challenger as m


def rows():
    data = []
    teams = [f"Team-{i}" for i in range(32)]
    for season in (2023, 2024, 2025):
        for week in range(1, 16):
            # Sixteen nonoverlapping games in each week; repeated teams next week.
            for i in range(16):
                home, away = teams[i], teams[(i + 16) % 32]
                market = .3 + .4 * ((i + week) % 8) / 7
                model = min(.99, max(.01, market + .03 * ((week + i) % 3 - 1)))
                result = int(((i * 23 + week * 11 + season) % 100) < 100 * market)
                actual = (3 + (i + week) % 4) * (1 if result else -1)
                data.append({
                    "game_id": f"{season}-{week}-{i}",
                    "season": season, "week": week,
                    "home_team": home, "away_team": away,
                    "market": market, "independent": model,
                    "margin": 8.5 * (2*market-1), "total": 50.0,
                    "actual_margin": actual, "y": result
                })
    return data


def test_same_week_outcomes_cannot_affect_current_features():
    sample = rows()
    original = m.asof_features(sample)
    varied = [{**r, "y": 1-r["y"], "actual_margin": -r["actual_margin"]}
              if r["season"] == 2024 and r["week"] == 5 else r for r in sample]
    other = m.asof_features(list(reversed(varied)))
    baseline = {r["game_id"]: r for r in original}
    reversed_by_id = {r["game_id"]: r for r in other}
    for r in sample:
        if r["season"] == 2024 and r["week"] == 5:
            for key in ("market_form3", "market_form6", "score_form3", "score_form6"):
                assert baseline[r["game_id"]][key] == reversed_by_id[r["game_id"]][key]
    assert any(baseline[r["game_id"]]["market_form3"] !=
               reversed_by_id[r["game_id"]]["market_form3"]
               for r in sample if r["season"] == 2024 and r["week"] == 6)


def test_no_cross_season_carryover():
    z = m.asof_features(rows())
    for r in z:
        if r["week"] == 1:
            assert r["market_form3"] == r["score_form3"] == 0
            assert r["market_form6"] == r["score_form6"] == 0


def test_swap_home_away_flips_feature_direction():
    row = m.asof_features(rows())[200]
    reverse = {**row, "home_team": row["away_team"],
               "away_team": row["home_team"],
               "market": 1-row["market"],
               "independent": 1-row["independent"]}
    for f in ("market_form3", "market_form6", "score_form3", "score_form6"):
        reverse[f] = -row[f]
    for family in m.FAMILIES:
        assert np.allclose(m.basis(row, family), -np.asarray(m.basis(reverse, family)))


def test_evaluation_does_not_select_params():
    sample = rows()
    orig = m.experiment(sample)
    altered = [{**r, "y": 1-r["y"], "actual_margin": -r["actual_margin"]}
               if r["season"] == 2025 else r for r in sample]
    modified = m.experiment(altered)
    assert orig["selected"] == modified["selected"]
    assert orig["estimated_coefficients"] == modified["estimated_coefficients"]
    assert orig["tune_best_raw"] == modified["tune_best_raw"]
    assert orig["evaluation_market"] != modified["evaluation_market"]
    assert orig["retrospective_screen"] in (
        "PASSED_RESEARCH_ONLY_NOT_PROSPECTIVE",
        "REJECTED_NO_INCREMENTAL_FORECAST_ADVANTAGE",
    )
    if orig["evaluation_challenger"]["correct"] < orig["evaluation_market"]["correct"]:
        assert orig["recommendation"] == "USE_MARKET_BASELINE_REJECT_FORM_ADJUSTMENT"
    assert not orig["promotion_eligible"]
    assert not orig["betting_authorized"]


def test_missing_year_fails_closed():
    with pytest.raises(ValueError, match="Insufficient"):
        m.experiment([r for r in rows() if r["season"] != 2024])


def test_csv_rejects_duplicate_and_mismatched_winner(tmp_path):
    path = tmp_path / "paired.csv"
    sample = {"market_type": "moneyline", "game_id": "a", "season": 2023, "week": 1,
              "home_team": "A", "away_team": "B", "pred_margin_home": 3,
              "pred_total": 48, "actual_margin_home": 7,
              "model_probability": .58, "no_vig_probability": .53,
              "result": "win", "side": "home"}
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample))
        writer.writeheader()
        writer.writerow(sample)
        writer.writerow(sample)
    with pytest.raises(ValueError, match="duplicated"):
        m.load_rows(path)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample))
        writer.writeheader()
        writer.writerow({**sample, "result": "loss"})
    with pytest.raises(ValueError, match="inconsistent"):
        m.load_rows(path)


def test_market_fallback_is_exact():
    enriched = m.asof_features(rows())
    pred = m.forecast(enriched)
    assert np.allclose(pred, [r["market"] for r in enriched])
