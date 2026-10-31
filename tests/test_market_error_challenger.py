"""Contract and chronology tests for the no-bet CFB market-error challenger."""
import csv
from pathlib import Path

import numpy as np
import pytest

from scripts import market_error_challenger as m


def sample():
    rows = []
    for season in (2023, 2024, 2025):
        for i in range(240):
            p = .20 + .60 * ((i % 31) / 30)
            independent = min(.99, max(.01, p + .10 * ((i % 5) - 2) / 2))
            y = int(((i * 17 + season) % 100) < round(100 * p))
            rows.append({"game_id": f"{season}-{i}", "season": season, "week": i % 15 + 1,
                         "market": p, "independent": independent, "y": y})
    return rows


def test_orientation_symmetry():
    row = {"market": .68, "independent": .59}
    reverse = {"market": .32, "independent": .41}
    for family in m.FAMILIES:
        assert np.allclose(m.basis(row, family), -np.asarray(m.basis(reverse, family)))


def test_2025_outcomes_never_change_selected_coefficients():
    rows = sample()
    a = m.experiment(rows)
    altered = [{**r, "y": 1-r["y"]} if r["season"] == 2025 else r for r in rows]
    b = m.experiment(altered)
    assert a["selected_on_tuning_only"] == b["selected_on_tuning_only"]
    assert a["fitted_coefficients"] == b["fitted_coefficients"]
    assert a["evaluation_market"] != b["evaluation_market"]
    assert a["promotion_eligible"] is False
    assert a["bets_authorized"] is False


def test_missing_season_fails_closed():
    with pytest.raises(ValueError, match="Not enough"):
        m.experiment([r for r in sample() if r["season"] != 2024])


def test_duplicate_and_invalid_quote_rejected(tmp_path: Path):
    path = tmp_path / "paired.csv"
    fields = ["game_id", "season", "week", "side", "market_type",
              "model_probability", "no_vig_probability", "result"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        row = dict(game_id="1", season=2023, week=1, side="home", market_type="moneyline",
                   model_probability=.53, no_vig_probability=.55, result="win")
        w.writerow(row)
        w.writerow(row)
    with pytest.raises(ValueError, match="duplicate"):
        m.load_paired(path)
    assert m._american_to_p(-110) == pytest.approx(110/210)
    assert m._american_to_p(100) == .5
    assert m._american_to_p("not an odd") is None


def test_no_adjustment_is_exact_market_and_clips():
    rows = [{"market": .52, "independent": .99, "y": 1}]
    assert m.predict(rows, None, None).tolist() == [.52]
    assert m.predict(rows, "disagreement", [10])[0] == .999
