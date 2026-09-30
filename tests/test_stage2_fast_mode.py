import numpy as np
import pandas as pd

import harbin.models as models


def _frame():
    rng = np.random.default_rng(27)
    rows = []
    gid = 1
    for season in range(2018, 2026):
        for week in range(1, 13):
            for _ in range(8):
                x = rng.normal(size=10)
                base_m = 1.5 + 1.2 * x[0] - 0.8 * x[1]
                base_t = 50.0 + 1.1 * x[2]
                rows.append({
                    "game_id": gid,
                    "season": season,
                    "week": week,
                    "baseline_margin": base_m,
                    "baseline_total": base_t,
                    "target_margin_home": base_m + 1.3 * x[3] + rng.normal(scale=4.5),
                    "target_total": base_t + 1.1 * x[4] + rng.normal(scale=5.0),
                    **{f"f{i}": x[i] for i in range(10)},
                })
                gid += 1
    return pd.DataFrame(rows)


def test_fast_backtest_env_skips_redundant_secondary_walkfolds(monkeypatch):
    monkeypatch.setenv("HARBIN_FAST_BACKTEST", "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("secondary season diagnostics should be skipped in fast outer backtest mode")

    monkeypatch.setattr(models, "_walkfolds", forbidden)
    bundle = models.train_models(_frame())
    assert bundle["metrics"]["margin_walkforward_fold_count"] == 0
    assert bundle["metrics"]["total_walkforward_fold_count"] == 0
    assert bundle["metrics"]["selection_uses_evaluation"] is False
