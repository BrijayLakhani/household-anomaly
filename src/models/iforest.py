"""
iforest.py — Isolation Forest for tabular anomaly detection.

Isolation Forest isolates anomalies by recursively random-partitioning the
feature space.  Anomalous points are few and far from the bulk, so they are
isolated in fewer splits than normal points (shorter average path length →
lower score).  Key parameter choices:

  n_estimators=200   : more trees → more stable scores; 200 is standard for
                       large datasets (diminishing returns after ~150).
  max_samples=256    : each tree sub-samples 256 rows, making training fast
                       even on 1.6 M rows.  Anomaly scores are still reliable
                       because isolation paths are short regardless of dataset
                       size.
  contamination='auto': only affects model.predict(); we bypass predict()
                        entirely and use score_samples() with our own
                        percentile threshold for explicit, reproducible control.
  random_state=42    : reproducibility.

Threshold choice: 95th percentile of TRAINING anomaly scores.
  This sets a hard "~5% false-positive rate on training data" guarantee.
  The test-set flag rate may differ, which is expected and informative.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def train_iforest(features_train: pd.DataFrame) -> IsolationForest:
    """
    Fit an Isolation Forest on the scaled training features.

    Parameters
    ----------
    features_train:
        Scaled tabular features with no NaN (1,641,572 x 24 from Phase 2).

    Returns
    -------
    Fitted IsolationForest instance.
    """
    model = IsolationForest(
        n_estimators=200,
        max_samples=256,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features_train)
    return model


def compute_if_threshold(
    model: IsolationForest,
    features_train: pd.DataFrame,
    percentile: float = 95,
) -> float:
    """
    Compute the anomaly-score threshold from TRAINING data only.

    Using training-set percentile (not test) prevents the threshold from
    being inflated by actual anomalies in the test set.  The 95th percentile
    means exactly ~5% of training samples will be flagged.
    """
    valid = features_train.notna().all(axis=1)
    raw = model.score_samples(features_train.loc[valid])
    anomaly = -raw
    return float(np.percentile(anomaly, percentile))


def score_iforest(
    model: IsolationForest,
    features: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Score all rows, producing raw score, anomaly score, and binary flag.

    Rows with any NaN in features (sensor gap rows in the test set) receive
    NaN scores and NaN flags so the dashboard can render them as 'no data'
    rather than false negatives.

    Parameters
    ----------
    model:
        Fitted IsolationForest.
    features:
        Scaled tabular features (may contain NaN rows for test split).
    threshold:
        Anomaly score above which a point is flagged (compute from train).

    Returns
    -------
    pd.DataFrame with columns if_raw_score, if_anomaly_score, if_flag,
    indexed by the same DatetimeIndex as features.
    """
    n = len(features)
    valid = features.notna().all(axis=1)

    raw = np.full(n, np.nan)
    if valid.any():
        raw[valid.values] = model.score_samples(features.loc[valid])

    anomaly = np.where(~np.isnan(raw), -raw, np.nan)
    flag = np.where(~np.isnan(anomaly), (anomaly > threshold).astype(float), np.nan)

    return pd.DataFrame(
        {"if_raw_score": raw, "if_anomaly_score": anomaly, "if_flag": flag},
        index=features.index,
    )


def save_iforest(
    model: IsolationForest,
    threshold: float,
    path: Path,
) -> None:
    """
    Persist the model (joblib) and its threshold (JSON sidecar).

    Sidecar JSON path: same directory, stem + '_threshold.json'.
    E.g. data/models/iforest.pkl → data/models/iforest_threshold.json.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    sidecar = path.parent / f"{path.stem}_threshold.json"
    sidecar.write_text(
        json.dumps({"threshold": threshold, "percentile": 95}, indent=2),
        encoding="utf-8",
    )
    print(f"IF model     -> {path}")
    print(f"IF threshold -> {sidecar}  ({threshold:.6f})")


def load_iforest(path: Path) -> tuple[IsolationForest, float]:
    """Load a saved Isolation Forest and its threshold."""
    model = joblib.load(path)
    sidecar = path.parent / f"{path.stem}_threshold.json"
    threshold = json.loads(sidecar.read_text())["threshold"]
    return model, threshold
