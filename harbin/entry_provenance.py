"""Strict verification flag parsing for historical market-entry research.

Generic pandas boolean casting treats the text False as true. This module
accepts only explicit affirmative values and NEVER upgrades legacy distinct
opening price indicators into verified entry provenance.
Upstream archive auditing remains responsible for verifying that assertion.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd


def is_explicitly_verified(value: object) -> bool:
    """Recognize only literal affirmative verification flags."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    if isinstance(value, Real):
        return bool(np.isfinite(value) and value == 1)
    return False


def verified_entry_mask(frame: pd.DataFrame) -> pd.Series:
    """Return a fail-closed mask, retaining the original row index."""
    if "entry_quote_verified" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame["entry_quote_verified"].map(is_explicitly_verified).astype(bool)
