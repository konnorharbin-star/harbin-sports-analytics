# Stage 2 — Walk-Forward Backtesting and Calibration

Stage 2 makes chronological validation explicit and prevents model selection, probability calibration, or residual sizing from learning from the final evaluation block.

## Outer walk-forward rule

Historical backtests are expanding-window by season/week. For a target Week `W`, the training set contains only games from earlier seasons or weeks `< W` in the same season. The target week is never present in its own training set.

`harbin.walkforward.walk_forward_week_splits` centralizes this invariant and raises on any temporal overlap.

## Inner chronological partition

Every model fit uses four ordered, whole-week partitions:

1. **Core** — fit candidate residual models.
2. **Tune** — choose the residual blend weight. Weight `0` is always eligible and therefore falls back to the baseline when the learned residual does not improve chronological tuning MAE.
3. **Calibration** — generate out-of-sample score-margin predictions, estimate residual scale, and fit the win-probability calibrator.
4. **Evaluation** — untouched until all weight and evaluation-calibrator choices are fixed. It is used only for final MAE/RMSE and Brier/log-loss/ECE reporting.

A season/week block is never divided across two partitions.

## Untouched evaluation guarantee

Evaluation results do not retune the residual blend weight. A release diagnostic records whether the already-selected residual beat the baseline on the evaluation block, but that diagnostic cannot rewrite the selected weight.

This removes the previous feedback path where final-holdout performance could change the deployed residual weight.

## Probability calibration

The evaluation probability calibrator is fitted on genuinely out-of-sample calibration-block margins. Logistic/Platt calibration is the low-variance default; isotonic calibration is used only when the calibration sample and class support are sufficient.

After untouched evaluation metrics are frozen, the live production calibrator may be refit using the combined calibration and evaluation out-of-sample predictions. That uses more historical calibration evidence without contaminating the reported evaluation scores.

## Final production refit

After validation:

- residual models are refit on all available historical training rows;
- the blend weight remains the weight selected on the Tune block;
- residual sigma comes from the Calibration block, not the Evaluation block;
- live probabilities use the out-of-sample production calibrator described above.

## Diagnostics

Stage 2 records:

- margin and total MAE/RMSE on untouched chronological evaluation;
- baseline-vs-selected model diagnostics;
- Brier score, log loss, and expected calibration error;
- temporal partition row counts and season/week spans;
- optional expanding-season walk-forward diagnostics, which are reporting-only and never participate in final weight selection.

The backtest workflow sets `HARBIN_FAST_BACKTEST=1`; expensive secondary fold diagnostics may be skipped there because the outer weekly expanding-window run is itself the primary validation.

## CI protection

`tests/test_stage2_walkforward.py` verifies that:

- partitions occur only at whole season/week boundaries;
- an outer target week never appears in its own training set;
- changing only final-evaluation outcomes cannot change tuned residual weights;
- probability calibration is fitted before and scored on a separate untouched evaluation block.

The legacy `release_weight_guard` helper remains available for compatibility tests, but Stage 2 model selection no longer uses evaluation performance to choose a weight.

## Stage boundary

Stage 2 does not redesign multi-book selection, CLV sourcing, or market execution. Those belong to Stage 3. Existing archived market results remain a downstream scoring layer and do not enter score-model features.

Stage 2 is complete only after the full repository test suite passes and the CFB walk-forward backtest workflow is green on `main`.
