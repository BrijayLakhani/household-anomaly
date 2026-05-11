"""
test_data.py

Smoke-tests for the data loading and cleaning pipeline.

The fixture prefers the pre-built parquet (fast); if that's absent it calls
load_clean() directly (requires the raw dataset).  Either way the tests skip
gracefully in a fresh checkout before any data has been downloaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PARQUET = _ROOT / "data" / "processed" / "clean.parquet"
_RAW = _ROOT / "data" / "raw" / "household_power_consumption.txt"

EXPECTED_COLUMNS: list[str] = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    """
    Return the cleaned DataFrame from whichever source is available.

    Order of preference:
    1. clean.parquet  (written by load_clean.py __main__)
    2. load_clean()   (called directly; requires raw dataset)
    3. pytest.skip    (neither source is ready)
    """
    if _PARQUET.exists():
        return pd.read_parquet(_PARQUET)

    if not _RAW.exists():
        pytest.skip(
            "No data available — run 'python src/data/download.py' then "
            "'python src/data/load_clean.py' before executing tests."
        )

    sys.path.insert(0, str(_ROOT))
    from src.data.load_clean import load_clean  # noqa: PLC0415

    return load_clean(_RAW)


# ── tests ────────────────────────────────────────────────────────────────────


def test_expected_columns(clean_df: pd.DataFrame) -> None:
    """All seven sensor columns must be present in the cleaned output."""
    missing = [c for c in EXPECTED_COLUMNS if c not in clean_df.columns]
    assert not missing, f"Columns absent from cleaned DataFrame: {missing}"


def test_datetime_index_monotonic_and_1min(clean_df: pd.DataFrame) -> None:
    """
    The DatetimeIndex must be strictly monotonic and uniformly spaced at
    exactly one-minute intervals.

    A non-uniform diff signals either duplicate timestamps (parse error) or
    dropped rows (missed by the reindex step in load_clean).
    """
    assert isinstance(clean_df.index, pd.DatetimeIndex), (
        "Index is not a DatetimeIndex"
    )
    assert clean_df.index.is_monotonic_increasing, (
        "DatetimeIndex is not monotonically increasing — duplicate or out-of-order timestamps"
    )

    diffs = pd.Series(clean_df.index).diff().dropna()
    one_min = pd.Timedelta("1min")
    bad = diffs[diffs != one_min]
    assert bad.empty, (
        f"Found {len(bad)} gaps that are not exactly 1 minute:\n{bad.value_counts()}"
    )


def test_no_question_mark_strings(clean_df: pd.DataFrame) -> None:
    """
    Sensor columns must be numeric (float64), not object.

    An object dtype indicates that '?' placeholders were not converted, which
    would silently poison every downstream numeric operation.
    """
    non_numeric = {
        col: str(clean_df[col].dtype)
        for col in EXPECTED_COLUMNS
        if clean_df[col].dtype == object
    }
    assert not non_numeric, (
        "These columns still have object dtype — '?' conversion failed: "
        + str(non_numeric)
    )
