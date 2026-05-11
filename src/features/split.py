"""
split.py

Strict temporal train/test split for time-series anomaly detection.

Random splits are forbidden here: a random draw can place minute T+1 in
training while T is in test, making the model see future context during
training and inflate evaluation metrics without any real predictive power.
Positional 80/20 preserves causality regardless of the data's date range.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
METADATA_PATH = PROCESSED_DIR / "split_metadata.json"


def time_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Split a time-indexed DataFrame into train/test by absolute row position.

    The cutoff is computed as df.index[floor(n * train_frac)], which
    guarantees every training timestamp is strictly earlier than every test
    timestamp.  Using a fixed fraction (rather than a calendar date) makes
    the split portable across datasets with different date ranges.

    Parameters
    ----------
    df:
        DataFrame with a strictly monotonic DatetimeIndex.
    train_frac:
        Fraction of rows assigned to train. Remaining rows go to test.

    Returns
    -------
    train_df:
        Rows before the cutoff timestamp.
    test_df:
        Rows from the cutoff timestamp onwards.
    metadata:
        Dict with split_timestamp, row counts, and boundary timestamps.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing.")
    if not (0 < train_frac < 1):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}.")

    n = len(df)
    cutoff_pos = int(n * train_frac)
    cutoff_ts = df.index[cutoff_pos]

    train_df = df.loc[df.index < cutoff_ts]
    test_df = df.loc[df.index >= cutoff_ts]

    metadata: dict[str, Any] = {
        "split_timestamp": cutoff_ts.isoformat(),
        "train_frac": train_frac,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_start": train_df.index.min().isoformat(),
        "train_end": train_df.index.max().isoformat(),
        "test_start": test_df.index.min().isoformat(),
        "test_end": test_df.index.max().isoformat(),
    }
    return train_df, test_df, metadata


def save_metadata(
    metadata: dict[str, Any],
    path: Path = METADATA_PATH,
) -> None:
    """Persist split metadata so every downstream step can reproduce the exact split."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"Split metadata -> {path}")


def load_metadata(path: Path = METADATA_PATH) -> dict[str, Any]:
    """Load a previously saved split metadata file."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
