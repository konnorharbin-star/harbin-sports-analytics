import numpy as np
import pandas as pd

from harbin.models import train_models
from harbin.walkforward import chronological_partition, walk_forward_week_splits


def _frame(rows_per_week=12, seasons=range(2018, 2026), weeks=range(1, 13)):
    rng = np.random.default_rng(26)
    rows = []
    gid = 1
    for season in seasons:
        for week in weeks:
            for _ in range(rows_per_week):
                x = rng.normal(size=10)
                base_m = 2.0 + 2.0 * x[0] - 1.2 * x[1]
                base_t = 51.0 + 1.5 * x[2]
                rows.append({
                    "game_id": gid,
                    "season": season,
                    "week": week,
                    "date": f"{season}-09-{min(28, week + 1):02d}",
                    "baseline_margin": base_m,
                    "baseline_total": base_t,
                    "target_margin_home": base_m + 1.8 * x[3] + rng.normal(scale=4.0),
                    "target_total": base_t + 1.5 * x[4] + rng.normal(scale=5.0),
                    **{f"f{i}": x[i] for i in range(10)},
                })
                gid += 1
    return pd.DataFrame(rows)


def test_partition_uses_whole_ordered_week_blocks():
    d = _frame(rows_per_week=8)
    core, tune, calib, evaluation, meta = chronological_partition(d)
    parts = [core, tune, calib, evaluation]
    assert meta["whole_week_boundaries"] is True
    assert meta["selection_uses_evaluation"] is False
    for left, right in zip(parts, parts[1:]):
        assert (int(left.iloc[-1].season), int(left.iloc[-1].week)) < (
            int(right.iloc[0].season), int(right.iloc[0].week)
        )
        assert set(map(tuple, left[["season", "week"]].drop_duplicates().to_numpy())).isdisjoint(
            set(map(tuple, right[["season", "week"]].drop_duplicates().to_numpy()))
        )


def test_outer_walkforward_never_trains_on_target_week_or_future():
    d = _frame(rows_per_week=8)
    splits = list(walk_forward_week_splits(d, start_season=2024, end_season=2025, min_train_rows=500))
    assert splits
    for split in splits:
        assert split.train_end < (split.season, split.week)
        assert not ((split.train.season == split.season) & (split.train.week >= split.week)).any()
        assert ((split.target.season == split.season) & (split.target.week == split.week)).all()


def test_final_evaluation_cannot_retune_residual_weight():
    d = _frame(rows_per_week=8)
    first = train_models(d, validation_diagnostics=False)
    _, _, _, evaluation, _ = chronological_partition(d)
    mutated = d.copy()
    idx = mutated.game_id.isin(evaluation.game_id)
    mutated.loc[idx, "target_margin_home"] += 80.0
    mutated.loc[idx, "target_total"] -= 60.0
    second = train_models(mutated, validation_diagnostics=False)
    assert first["margin_weight"] == second["margin_weight"]
    assert first["total_weight"] == second["total_weight"]
    assert first["metrics"]["selection_uses_evaluation"] is False
    assert second["metrics"]["selection_uses_evaluation"] is False
    assert first["metrics"]["margin_mae"] != second["metrics"]["margin_mae"]


def test_probability_calibration_is_scored_on_untouched_evaluation():
    bundle = train_models(_frame(rows_per_week=8), validation_diagnostics=False)
    m = bundle["metrics"]
    assert m["win_calibration_fit_rows"] > 0
    assert m["win_calibration_eval_rows"] > 0
    assert m["win_production_calibration_oof_rows"] == m["win_calibration_fit_rows"] + m["win_calibration_eval_rows"]
    assert 0 <= m["win_brier"] <= 1
    assert np.isfinite(m["win_log_loss"])
    assert 0 <= m["win_ece"] <= 1
