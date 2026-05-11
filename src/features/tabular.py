"""
tabular.py

Constructs the 24-column tabular feature matrix fed to Isolation Forest.

Feature engineering rationale
-------------------------------
Raw values       : anchor the model to actual sensor readings; IF can then
                   detect outlier *combinations* of raw readings.
Lag features     : anomaly detection benefits from knowing recent context.
                   1-min lag catches instantaneous spikes; 60-min tracks
                   deviation from the recent hour; 1440-min compares with
                   the same time yesterday (strong daily periodicity).
Rolling stats    : anomalies often appear as breaks in smooth local
                   distributions.  15-min window captures appliance-cycle
                   noise; 60-min captures household routine fluctuation.
Rate of change   : sudden power-draw jumps (gap_diff_1) are a primary
                   anomaly signal for faults and appliance surges.
Cyclical time    : sin/cos encoding avoids the artificial discontinuity
                   at midnight (23h->0h) and at end-of-week that integer
                   hour/dow encodings would introduce.
Sub-meter ratios : detect when the three metered circuits collectively
                   account for an anomalous fraction of total consumption
                   (e.g., ratio > 1 signals meter miscalibration).

Final column list (24 columns)
-------------------------------
Raw (7)   : Global_active_power, Global_reactive_power, Voltage,
            Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3
Lags (3)  : gap_lag_1, gap_lag_60, gap_lag_1440
Rolling(4): gap_roll_mean_15, gap_roll_std_15, gap_roll_mean_60, gap_roll_std_60
RoC (1)   : gap_diff_1
Time (6)  : hour_sin, hour_cos, dow_sin, dow_cos, is_weekend, is_night
Ratios(3) : ratio_sub1, ratio_sub2, ratio_sub3
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "features_tabular.parquet"

RAW_COLS: list[str] = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

EXPECTED_N_COLS: int = 24


def build_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the 24-column tabular feature matrix from the cleaned DataFrame.

    All lag and rolling operations are purely backward-looking, so computing
    them on the full DataFrame before the train/test split introduces no
    temporal leakage.  The scaler, however, must still be fitted exclusively
    on training rows (enforced in build_all.py).

    Parameters
    ----------
    df:
        Cleaned 7-column DataFrame from clean.parquet with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        24-column feature matrix sharing the same DatetimeIndex as the input.
        NaN rows from lags / rolling windows at the series boundary are
        preserved here; callers decide whether to drop or keep them.
    """
    gap = df["Global_active_power"]
    out = df[RAW_COLS].copy()

    # --- Lag features -------------------------------------------------------
    out["gap_lag_1"] = gap.shift(1)
    out["gap_lag_60"] = gap.shift(60)
    out["gap_lag_1440"] = gap.shift(1440)  # same time yesterday

    # --- Rolling statistics -------------------------------------------------
    # min_periods=1 for mean: partial windows still yield a useful estimate.
    # min_periods=2 for std: std of a single value is undefined (ddof=1).
    out["gap_roll_mean_15"] = gap.rolling(15, min_periods=1).mean()
    out["gap_roll_std_15"] = gap.rolling(15, min_periods=2).std()
    out["gap_roll_mean_60"] = gap.rolling(60, min_periods=1).mean()
    out["gap_roll_std_60"] = gap.rolling(60, min_periods=2).std()

    # --- Rate of change -----------------------------------------------------
    out["gap_diff_1"] = gap.diff(1)

    # --- Cyclical time features ---------------------------------------------
    hour = df.index.hour.astype(float)
    dow = df.index.dayofweek.astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (df.index.dayofweek >= 5).astype(np.int8)
    out["is_night"] = ((hour < 6) | (hour > 22)).astype(np.int8)

    # --- Sub-metering ratios ------------------------------------------------
    # Sub-metering columns are in Wh; active power converted to Wh/min.
    # Clipped to [0, 1]: ratios slightly above 1 occur due to rounding in
    # the meter; hard-clamping avoids encoding meter artefacts as features.
    total_wh_per_min = (gap * 1000.0 / 60.0).replace(0, np.nan)
    for i, col in enumerate(
        ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"], start=1
    ):
        ratio = df[col] / total_wh_per_min
        out[f"ratio_sub{i}"] = ratio.clip(0.0, 1.0)

    assert out.shape[1] == EXPECTED_N_COLS, (
        f"Expected {EXPECTED_N_COLS} columns, got {out.shape[1]}"
    )
    return out


def save_tabular(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """Save the full (unsplit) tabular feature matrix before scaling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"Tabular features (full) -> {path}  shape={df.shape}")
