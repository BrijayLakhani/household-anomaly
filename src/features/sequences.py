"""
sequences.py

Builds overlapping sliding-window tensors of shape (N, window, 7) for the
LSTM Autoencoder.  Only the 7 raw sensor columns are used — the LSTM learns
temporal structure from the raw signal itself rather than from hand-crafted
lag features, which would be redundant and would obscure the reconstruction
task.

Window size choice (default = 60 minutes)
------------------------------------------
A 60-minute window captures one complete appliance cycle (dishwasher ~55 min,
washing machine ~60-90 min, kettle ~2 min) while keeping GPU memory cost
tractable.  The alternative of window=1440 (full day) would provide richer
circadian context but:
  - Reduces the training set from ~1.66 M to only ~1.16 M sequences (30% fewer).
  - Creates arrays ~24x larger: train array alone would be ~67 GB at float32.
  - Slows LSTM training substantially (BPTT over 1440 steps vs 60).
Trade-off accepted: the LSTM captures sub-hourly patterns; daily periodicity
is handled by the cyclical time features in the tabular path.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

RAW_COLS: list[str] = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def build_sequences(
    df: pd.DataFrame,
    window: int = 60,
    stride: int = 1,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex, StandardScaler]:
    """
    Scale the 7 raw columns and build a sliding-window tensor.

    If ``scaler`` is None a new StandardScaler is fitted on the rows of
    ``df`` that have no NaN (i.e. call with train data first, then pass
    the returned scaler for the test call to avoid data leakage).

    Any window that contains at least one NaN value (from the original
    sensor gaps) is silently dropped so the returned array is always
    fully numeric.

    Parameters
    ----------
    df:
        DataFrame with DatetimeIndex and at least the 7 RAW_COLS columns.
    window:
        Number of consecutive 1-minute rows per sequence.
    stride:
        Step in minutes between sequence start positions.
        stride=1 maximises training data but produces very large arrays;
        see the module docstring for memory estimates.
    scaler:
        Pre-fitted StandardScaler.  Pass None to fit a fresh one on ``df``.

    Returns
    -------
    sequences:
        float32 array of shape (N, window, 7).  N < total possible windows
        because windows overlapping NaN rows are excluded.
    timestamps:
        DatetimeIndex of the END timestamp of each returned window
        (length N, aligned with sequences).
    scaler:
        The StandardScaler that was used (fitted or the one passed in).
    """
    raw = df[RAW_COLS].copy()

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(raw.dropna())

    # Scale row-by-row; NaN rows become NaN after transform (propagated below)
    valid_mask = raw.notna().all(axis=1)
    scaled_arr = np.full((len(raw), len(RAW_COLS)), np.nan, dtype=np.float64)
    scaled_arr[valid_mask.values] = scaler.transform(raw.loc[valid_mask])

    # Build sliding windows using advanced indexing (avoids Python loop)
    n = len(scaled_arr)
    starts = np.arange(0, n - window + 1, stride)
    idx_matrix = starts[:, None] + np.arange(window)[None, :]  # (n_windows, window)
    windows = scaled_arr[idx_matrix]  # (n_windows, window, 7)

    # Drop any window containing NaN (comes from original sensor gaps)
    has_nan = np.isnan(windows).any(axis=(1, 2))
    sequences = windows[~has_nan].astype(np.float32)

    end_positions = starts[~has_nan] + window - 1
    timestamps = df.index[end_positions]

    return sequences, timestamps, scaler


def save_sequences(
    sequences: np.ndarray,
    timestamps: pd.DatetimeIndex,
    array_path: Path,
    index_path: Path,
) -> None:
    """Save the sequence array and its timestamp index to disk."""
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(array_path, sequences)
    pd.DataFrame(index=timestamps).to_parquet(index_path)
    size_mb = sequences.nbytes / 1_048_576
    print(f"Sequences -> {array_path}  shape={sequences.shape}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(_PROJECT_ROOT))
    from src.features.split import load_metadata

    clean_path = PROCESSED_DIR / "clean.parquet"
    meta = load_metadata()
    split_ts = pd.Timestamp(meta["split_timestamp"])

    full_df = pd.read_parquet(clean_path)
    train_df = full_df.loc[full_df.index < split_ts]
    test_df = full_df.loc[full_df.index >= split_ts]

    print("Building train sequences ...")
    seqs_tr, idx_tr, scaler_lstm = build_sequences(train_df, window=60, stride=1)
    save_sequences(
        seqs_tr, idx_tr,
        PROCESSED_DIR / "sequences_train.npy",
        PROCESSED_DIR / "sequences_train_index.parquet",
    )

    joblib.dump(scaler_lstm, PROCESSED_DIR / "scaler_lstm.pkl")
    print(f"LSTM scaler -> {PROCESSED_DIR / 'scaler_lstm.pkl'}")

    print("Building test sequences ...")
    seqs_te, idx_te, _ = build_sequences(test_df, window=60, stride=1, scaler=scaler_lstm)
    save_sequences(
        seqs_te, idx_te,
        PROCESSED_DIR / "sequences_test.npy",
        PROCESSED_DIR / "sequences_test_index.parquet",
    )
