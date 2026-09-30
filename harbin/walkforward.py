from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


TIME_COLUMNS = ("season", "week", "date", "game_id")


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Return a stable chronological copy without ever sorting by a future target."""
    missing = [c for c in ("season", "week") if c not in df.columns]
    if missing:
        raise ValueError(f"Temporal validation requires columns: {', '.join(missing)}")
    cols = [c for c in TIME_COLUMNS if c in df.columns]
    return df.sort_values(cols, kind="stable").reset_index(drop=True)


def _week_key(frame: pd.DataFrame, which: str) -> tuple[int, int] | None:
    if frame.empty:
        return None
    row = frame.iloc[0] if which == "first" else frame.iloc[-1]
    return int(row["season"]), int(row["week"])


def temporal_span(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "start": _week_key(frame, "first"),
        "end": _week_key(frame, "last"),
    }


def _week_blocks(d: pd.DataFrame) -> pd.DataFrame:
    return (
        d.groupby(["season", "week"], sort=True, dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["season", "week"], kind="stable")
        .reset_index(drop=True)
    )


def _choose_boundary(cumulative, target: float, low: int, high: int) -> int:
    candidates = [(i, int(v)) for i, v in enumerate(cumulative) if int(v) >= low and int(v) <= high]
    if not candidates:
        raise RuntimeError(
            f"Unable to create whole-week chronological partition within row bounds [{low}, {high}]"
        )
    _, value = min(candidates, key=lambda item: (abs(item[1] - target), item[1]))
    return value


def chronological_partition(
    df: pd.DataFrame,
    *,
    min_core_rows: int = 300,
    min_section_rows: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split history into core/tune/calibration/evaluation on whole week boundaries.

    The final evaluation block is untouched by model-weight and calibrator selection.
    The function adapts section minima for smaller valid training sets, but never cuts a
    season/week block in half.
    """
    d = sort_chronologically(df)
    n = len(d)
    if n < 500:
        raise RuntimeError(f"Need at least 500 historical games for nested chronology; got {n}")

    blocks = _week_blocks(d)
    if len(blocks) < 8:
        raise RuntimeError(f"Need at least 8 historical season/week blocks; got {len(blocks)}")
    cumulative = blocks["rows"].cumsum().astype(int).tolist()

    # Preserve a substantial fitting core while guaranteeing three chronological
    # post-core sections. For a 500-row floor this yields roughly 65 rows each.
    available_after_core = max(0, n - int(min_core_rows))
    section_min = min(int(min_section_rows), max(30, available_after_core // 3))
    if int(min_core_rows) + 3 * section_min > n:
        raise RuntimeError("Insufficient rows for core/tune/calibration/evaluation partition")

    b1 = _choose_boundary(
        cumulative,
        target=0.58 * n,
        low=int(min_core_rows),
        high=n - 3 * section_min,
    )
    b2 = _choose_boundary(
        cumulative,
        target=0.72 * n,
        low=b1 + section_min,
        high=n - 2 * section_min,
    )
    b3 = _choose_boundary(
        cumulative,
        target=0.86 * n,
        low=b2 + section_min,
        high=n - section_min,
    )

    core = d.iloc[:b1].copy()
    tune = d.iloc[b1:b2].copy()
    calibration = d.iloc[b2:b3].copy()
    evaluation = d.iloc[b3:].copy()

    parts = {
        "core": temporal_span(core),
        "tune": temporal_span(tune),
        "calibration": temporal_span(calibration),
        "evaluation": temporal_span(evaluation),
        "whole_week_boundaries": True,
        "selection_uses_evaluation": False,
    }
    assert_strict_prior(core, tune)
    assert_strict_prior(tune, calibration)
    assert_strict_prior(calibration, evaluation)
    return core, tune, calibration, evaluation, parts


def assert_strict_prior(train: pd.DataFrame, target: pd.DataFrame) -> None:
    if train.empty or target.empty:
        return
    train_end = _week_key(sort_chronologically(train), "last")
    target_start = _week_key(sort_chronologically(target), "first")
    if train_end is None or target_start is None or not (train_end < target_start):
        raise RuntimeError(
            f"Walk-forward leakage: training ends at {train_end}, target begins at {target_start}"
        )


@dataclass(frozen=True)
class WalkForwardSplit:
    season: int
    week: int
    train: pd.DataFrame
    target: pd.DataFrame
    train_rows: int
    train_end: tuple[int, int] | None


def walk_forward_week_splits(
    df: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
    min_train_rows: int = 800,
) -> Iterator[WalkForwardSplit]:
    """Yield strict expanding-window weekly train/target splits."""
    d = sort_chronologically(df)
    eval_blocks = (
        d[(d.season >= int(start_season)) & (d.season <= int(end_season))][["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"], kind="stable")
    )
    for row in eval_blocks.itertuples(index=False):
        season, week = int(row.season), int(row.week)
        train = d[(d.season < season) | ((d.season == season) & (d.week < week))].copy()
        target = d[(d.season == season) & (d.week == week)].copy()
        if len(train) < int(min_train_rows) or target.empty:
            continue
        assert_strict_prior(train, target)
        yield WalkForwardSplit(
            season=season,
            week=week,
            train=train,
            target=target,
            train_rows=int(len(train)),
            train_end=_week_key(train, "last"),
        )
