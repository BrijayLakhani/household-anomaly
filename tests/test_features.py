"""
test_features.py

Correctness tests for the Phase-2 feature engineering pipeline.
All tests skip gracefully if the required artefacts have not been built yet;
run 'python src/features/build_all.py' first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PROC = _ROOT / "data" / "processed"
sys.path.insert(0, str(_ROOT))

# Artefact paths
FEATURES_TRAIN = _PROC / "features_tabular_train.parquet"
FEATURES_TEST = _PROC / "features_tabular_test.parquet"
SCALER_TABULAR = _PROC / "scaler_tabular.pkl"
SEQ_TRAIN = _PROC / "sequences_train.npy"
METADATA = _PROC / "split_metadata.json"

EXPECTED_N_COLS = 24
WINDOW = 60
N_RAW_FEATURES = 7


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "Required artefacts not found — run 'python src/features/build_all.py':\n"
            + "\n".join(f"  {p}" for p in missing)
        )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_feat() -> pd.DataFrame:
    _require(FEATURES_TRAIN)
    return pd.read_parquet(FEATURES_TRAIN)


@pytest.fixture(scope="module")
def test_feat() -> pd.DataFrame:
    _require(FEATURES_TEST)
    return pd.read_parquet(FEATURES_TEST)


@pytest.fixture(scope="module")
def metadata() -> dict:
    _require(METADATA)
    return json.loads(METADATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def scaler_tab():
    _require(SCALER_TABULAR)
    return joblib.load(SCALER_TABULAR)


@pytest.fixture(scope="module")
def seq_train() -> np.ndarray:
    _require(SEQ_TRAIN)
    return np.load(SEQ_TRAIN)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_time_split_no_overlap(train_feat: pd.DataFrame, test_feat: pd.DataFrame) -> None:
    """
    Every training timestamp must be strictly earlier than every test timestamp.
    Any overlap means future data leaked into training, invalidating the split.
    """
    assert train_feat.index.max() < test_feat.index.min(), (
        f"Overlap detected: train max={train_feat.index.max()}, "
        f"test min={test_feat.index.min()}"
    )


def test_tabular_no_train_nan(train_feat: pd.DataFrame) -> None:
    """
    Training tabular features must contain zero NaN values after dropna.
    A single NaN propagated to the scaler would silently corrupt all scaled
    values and produce NaN anomaly scores at inference time.
    """
    nan_counts = train_feat.isna().sum()
    columns_with_nan = nan_counts[nan_counts > 0]
    assert columns_with_nan.empty, (
        f"Training features still contain NaN:\n{columns_with_nan}"
    )


def test_tabular_column_count(train_feat: pd.DataFrame) -> None:
    """
    Exactly 24 columns must be present, matching the documented feature list
    in tabular.py.  A mismatch means a feature was silently added or removed,
    which would break any saved model that expects a fixed input dimension.
    """
    assert train_feat.shape[1] == EXPECTED_N_COLS, (
        f"Expected {EXPECTED_N_COLS} columns, got {train_feat.shape[1]}.\n"
        f"Columns present: {list(train_feat.columns)}"
    )


def test_sequence_shape(seq_train: np.ndarray) -> None:
    """
    Training sequences must be a 3-D float32 array with the correct window
    and feature dimensions.  Wrong shape means the LSTM's input layer
    definition would be mismatched at training time.
    """
    assert seq_train.ndim == 3, f"Expected 3-D array, got shape {seq_train.shape}"
    assert seq_train.shape[1] == WINDOW, (
        f"Expected window dim={WINDOW}, got {seq_train.shape[1]}"
    )
    assert seq_train.shape[2] == N_RAW_FEATURES, (
        f"Expected feature dim={N_RAW_FEATURES}, got {seq_train.shape[2]}"
    )
    assert seq_train.dtype == np.float32, (
        f"Expected float32, got {seq_train.dtype}"
    )


def test_scaler_fit_on_train_only(
    train_feat: pd.DataFrame, scaler_tab
) -> None:
    """
    The tabular scaler's stored mean_ must match the column means of the
    training set, not the full-dataset mean.  This verifies that the scaler
    was fitted before test data was included — the core leakage guard.
    """
    train_means = train_feat[scaler_tab.feature_names_in_].mean().values
    np.testing.assert_allclose(
        scaler_tab.mean_, train_means, rtol=1e-4,
        err_msg=(
            "scaler.mean_ does not match train column means — "
            "scaler may have been fitted on the full dataset (data leakage)."
        ),
    )


def test_no_temporal_leakage(metadata: dict) -> None:
    """
    The recorded training end timestamp must be strictly before the test
    start timestamp.  This is redundant with test_time_split_no_overlap but
    validates the persisted metadata file itself — the source of truth used
    by every downstream pipeline step.
    """
    train_end = pd.Timestamp(metadata["train_end"])
    test_start = pd.Timestamp(metadata["test_start"])
    assert train_end < test_start, (
        f"Temporal leakage in metadata: train_end={train_end}, "
        f"test_start={test_start}"
    )
