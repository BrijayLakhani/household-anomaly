"""
test_models.py

Correctness tests for Phase-3 model training and scoring artefacts.
All tests skip gracefully if score_all.py has not been run yet.
LSTM tests additionally skip if TensorFlow is not installed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_PROC = _ROOT / "data" / "processed"
_MODELS = _ROOT / "data" / "models"
sys.path.insert(0, str(_ROOT))

SCORES_PATH = _MODELS / "scores.parquet"
METADATA_PATH = _PROC / "split_metadata.json"
IF_MODEL_PATH = _MODELS / "iforest.pkl"
IF_THRESHOLD_PATH = _MODELS / "iforest_threshold.json"
LSTM_MODEL_DIR = _MODELS
LSTM_HISTORY_PATH = _MODELS / "lstm_ae_history.json"
SYNTH_EVAL_PATH = _MODELS / "synthetic_eval.json"
SEQ_TRAIN_PATH = _PROC / "sequences_train.npy"

try:
    from src.models.lstm_ae import TF_AVAILABLE
except ImportError:
    TF_AVAILABLE = False

requires_tf = pytest.mark.skipif(
    not TF_AVAILABLE,
    reason="TensorFlow not installed (Python 3.14 not yet supported by stable TF)",
)


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "Required artefacts not found — run 'python src/models/score_all.py':\n"
            + "\n".join(f"  {p}" for p in missing)
        )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scores() -> pd.DataFrame:
    _require(SCORES_PATH)
    return pd.read_parquet(SCORES_PATH)


@pytest.fixture(scope="module")
def metadata() -> dict:
    _require(METADATA_PATH)
    return json.loads(METADATA_PATH.read_text())


@pytest.fixture(scope="module")
def if_model():
    _require(IF_MODEL_PATH)
    from src.models.iforest import load_iforest
    return load_iforest(IF_MODEL_PATH)  # returns (model, threshold)


@pytest.fixture(scope="module")
def lstm_history() -> dict:
    _require(LSTM_HISTORY_PATH)
    return json.loads(LSTM_HISTORY_PATH.read_text())


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_iforest_predicts_flag_distribution(
    scores: pd.DataFrame,
    metadata: dict,
) -> None:
    """
    The IF flag rate on the training set must be ~5% (4–6%).
    Using the 95th-percentile threshold guarantees exactly 5% on training
    data by construction; a result outside [4%, 6%] indicates the threshold
    was computed incorrectly (e.g. from the full dataset, introducing leakage).
    """
    cutoff = pd.Timestamp(metadata["split_timestamp"])
    train_flags = scores.loc[scores.index < cutoff, "if_flag"].dropna()
    flag_rate = float(train_flags.mean())
    assert 0.04 <= flag_rate <= 0.06, (
        f"Expected IF train flag rate ~5%, got {flag_rate:.3%}. "
        "Check that the threshold was computed from train data only."
    )


def test_iforest_no_nan_in_train_scores(
    scores: pd.DataFrame,
    metadata: dict,
) -> None:
    """
    Training tabular features have no NaN, so IF anomaly scores on the
    training period must also be fully numeric.  Any NaN here means the
    score_iforest function incorrectly treated a valid row as a sensor gap.
    """
    cutoff = pd.Timestamp(metadata["split_timestamp"])
    train_scores = scores.loc[scores.index < cutoff, "if_anomaly_score"]
    nan_count = train_scores.isna().sum()
    assert nan_count == 0, (
        f"Found {nan_count} NaN in train IF scores. "
        "Training features should be fully numeric after Phase-2 dropna."
    )


@requires_tf
def test_lstm_ae_output_shape() -> None:
    """
    The LSTM-AE must reconstruct inputs to the exact same shape.
    A shape mismatch means the model architecture does not properly decode
    back to the original window × features dimensions.
    """
    _require(LSTM_MODEL_DIR / "lstm_ae.keras", SEQ_TRAIN_PATH)
    from src.models.lstm_ae import load_lstm_ae

    model, _ = load_lstm_ae(LSTM_MODEL_DIR)
    X_sample = np.load(SEQ_TRAIN_PATH)[:32]  # first 32 sequences
    X_pred = model.predict(X_sample, verbose=0)
    assert X_pred.shape == X_sample.shape, (
        f"Output shape {X_pred.shape} != input shape {X_sample.shape}"
    )


@requires_tf
def test_lstm_ae_training_loss_decreases(lstm_history: dict) -> None:
    """
    The final training loss must be < 70% of the initial loss.
    If loss does not decrease the model did not converge — likely caused by
    NaN in sequences, a wrong input shape, or an exploding gradient.
    """
    losses = lstm_history["loss"]
    assert len(losses) >= 2, "Need at least 2 epochs of history."
    initial_loss = losses[0]
    final_loss = losses[-1]
    assert final_loss < initial_loss * 0.7, (
        f"Loss did not decrease enough: initial={initial_loss:.6f}, "
        f"final={final_loss:.6f} (ratio={final_loss/initial_loss:.2f}, expected < 0.70)"
    )


def test_scores_parquet_columns(scores: pd.DataFrame) -> None:
    """
    scores.parquet must contain the three key columns for the dashboard:
    if_anomaly_score (continuous IF signal), lstm_recon_error (continuous
    LSTM signal), and agreement (combined flag).  Missing columns mean a
    downstream phase would silently produce empty visualisations.
    """
    required = ["if_anomaly_score", "lstm_recon_error", "agreement"]
    missing = [c for c in required if c not in scores.columns]
    assert not missing, (
        f"scores.parquet missing columns: {missing}. "
        f"Present columns: {list(scores.columns)}"
    )


def test_synthetic_eval_overall_f1() -> None:
    """
    The Isolation Forest overall F1 on injected anomalies must exceed the
    random-baseline F1 by a meaningful margin.

    Baseline: random flagging at 5% on ~3.7% contamination gives F1 ~ 0.04.
    We require F1 >= 0.07 (nearly 2x random baseline) to confirm the model
    is doing something useful.  Isolation Forest is a non-temporal model;
    contextual and collective anomalies (which require time context) will
    always be hard for it.  The combined IF + LSTM score is expected to
    exceed F1=0.5 once TensorFlow is available.
    """
    _require(SYNTH_EVAL_PATH)
    results = json.loads(SYNTH_EVAL_PATH.read_text())
    if_f1 = results.get("isolation_forest", {}).get("overall", {}).get("f1", 0.0)
    assert isinstance(if_f1, float) and if_f1 >= 0.07, (
        f"IF overall F1 = {if_f1:.3f} < 0.07 (barely above random baseline ~0.04). "
        "Check model training, scaler pipeline, or anomaly injection logic."
    )
