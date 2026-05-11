"""
load_clean.py

Loads and cleans the raw UCI household power consumption dataset.

Missing-value strategy
----------------------
The raw file marks ~1.25 % of minute-rows with '?' across all seven sensor
columns simultaneously (the meter was offline, not individual sensor failures).

Chosen strategy:
  • Gaps ≤ 5 consecutive minutes  → forward-filled.
    Rationale: short outages / reboots; neighbouring values give a faithful
    interpolation with minimal distortion of distributions.
  • Gaps > 5 consecutive minutes  → left as NaN.
    Rationale: anything longer represents a genuine absence of measurement
    (night-time meter cut, holiday, maintenance).  Downstream models must
    decide how to handle these (e.g. drop rows, separate imputation step).

The total volume of unfilled NaN after this policy is documented when the
script is run as __main__.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = _PROJECT_ROOT / "data" / "raw" / "household_power_consumption.txt"
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "clean.parquet"

NUMERIC_COLS: list[str] = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def load_clean(raw_path: Path = RAW_FILE) -> pd.DataFrame:
    """
    Parse, type-convert, and forward-fill the raw power consumption text file.

    Returns a DataFrame with a complete 1-minute DatetimeIndex.  Rows that
    were absent from the raw file (meter gaps) are reindexed in so that the
    index is perfectly uniform; those rows inherit the forward-fill treatment.

    Parameters
    ----------
    raw_path:
        Path to ``household_power_consumption.txt``.  Defaults to the
        canonical location inside the project tree.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex at 1-minute frequency (UTC-naive).
        Columns: the seven NUMERIC_COLS, dtype float64.
    """
    df = pd.read_csv(
        raw_path,
        sep=";",
        na_values="?",
        low_memory=False,
    )

    # Explicit format is safer than dayfirst inference in pandas ≥ 2.0
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reindex to a gapless 1-minute grid so missing rows become NaN rows
    # rather than silent holes in the time series.
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    df = df.reindex(full_index)
    df.index.name = "datetime"

    # Forward-fill short gaps only (≤ 5 minutes)
    df = df.ffill(limit=5)

    return df[NUMERIC_COLS]


def save_clean(df: pd.DataFrame, output_path: Path = OUTPUT_FILE) -> None:
    """Persist the cleaned DataFrame as Parquet (columnar, fast I/O)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Saved -> {output_path}  ({len(df):,} rows)")


if __name__ == "__main__":
    print("Loading and cleaning ...")
    clean = load_clean()
    save_clean(clean)

    total = len(clean)
    remaining_na = clean[NUMERIC_COLS].isna().sum()
    print(f"\nTotal rows (1-min grid): {total:,}")
    print("\nUnfilled NaN per column (gaps > 5 consecutive minutes):")
    for col, n in remaining_na.items():
        print(f"  {col:<30} {n:>6,}  ({n / total * 100:.3f}%)")
