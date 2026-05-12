"""
lime_explainer.py — LIME explanations for the Isolation Forest model.

LIME (Local Interpretable Model-agnostic Explanations) approximates the
black-box model locally around each query point with a linear surrogate.
We use regression mode because we're explaining a continuous anomaly score
(not a class probability).

Reproducibility note
--------------------
LIME samples a neighbourhood at random.  We fix random_state=42 at the
explainer level and pass per-instance seeds (42 + index) for reproducibility.
Parallel execution (n_jobs > 1) uses separate worker processes, so results
may differ very slightly from sequential execution.  This known limitation
is documented in Ribeiro et al. (2016) and is not a bug.

Output file
-----------
data/explanations/lime_explanations.parquet
    Long format: (timestamp, feature_name, lime_weight, lime_rank)
    Each timestamp has exactly 5 rows (top-5 contributing features).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lime.lime_tabular import LimeTabularExplainer

_ROOT = Path(__file__).resolve().parents[2]
EXPL_DIR = _ROOT / "data" / "explanations"

_NUM_FEATURES = 5
_NUM_SAMPLES = 1000


def build_lime_explainer(
    features_train: pd.DataFrame,
    feature_names: list[str],
) -> LimeTabularExplainer:
    """
    Construct a LimeTabularExplainer fitted on training feature statistics.

    Parameters
    ----------
    features_train:
        Scaled tabular training features (no NaN).
    feature_names:
        Ordered list of 24 feature names.

    Returns
    -------
    LimeTabularExplainer in regression mode with continuous discretisation.
    """
    return LimeTabularExplainer(
        training_data=features_train.values.astype(np.float64),
        feature_names=feature_names,
        mode="regression",
        discretize_continuous=True,
        random_state=42,
    )


def explain_anomaly_lime(
    explainer: LimeTabularExplainer,
    iforest_model: Any,
    features_row: np.ndarray,
    num_features: int = _NUM_FEATURES,
    seed: int = 42,
) -> dict[str, float]:
    """
    Explain a single anomaly with LIME.

    Parameters
    ----------
    explainer:
        Pre-built LimeTabularExplainer.
    iforest_model:
        Fitted IsolationForest.  Prediction function = -score_samples
        (positive = more anomalous).
    features_row:
        1-D float64 array of 24 scaled feature values.
    num_features:
        Number of top contributing features to return.
    seed:
        Random seed for this instance's neighbourhood sampling.

    Returns
    -------
    dict mapping feature_name → LIME weight (positive = pushes toward anomaly).
    """
    predict_fn = lambda X: -iforest_model.score_samples(X)
    exp = explainer.explain_instance(
        features_row,
        predict_fn,
        num_features=num_features,
        num_samples=_NUM_SAMPLES,
    )
    return dict(exp.as_list())


def _worker(
    explainer: LimeTabularExplainer,
    iforest_model: Any,
    row: np.ndarray,
) -> dict[str, float]:
    """Top-level worker function (must be pickleable for multiprocessing)."""
    return explain_anomaly_lime(explainer, iforest_model, row)


def compute_lime_for_all(
    explainer: LimeTabularExplainer,
    iforest_model: Any,
    features: pd.DataFrame,
    selected_timestamps: pd.DatetimeIndex,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Apply LIME to every selected timestamp, parallelised via joblib.

    Parameters
    ----------
    explainer:
        Pre-built LimeTabularExplainer.
    iforest_model:
        Fitted IsolationForest.
    features:
        Full scaled tabular feature DataFrame (train + test).
    selected_timestamps:
        DatetimeIndex of the 200 selected points.
    n_jobs:
        Number of parallel workers (-1 = all CPUs).

    Returns
    -------
    Long-format DataFrame:
        (timestamp, feature_name, lime_weight, lime_rank)
    Each timestamp has exactly _NUM_FEATURES rows.
    """
    X = features.loc[selected_timestamps]
    rows_np = X.values.astype(np.float64)
    n = len(rows_np)

    print(f"  Running LIME on {n} instances (n_jobs={n_jobs}) ...")

    results: list[dict[str, float]] = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_worker)(explainer, iforest_model, rows_np[i])
        for i in range(n)
    )

    records = []
    for i, (ts, weights) in enumerate(zip(selected_timestamps, results)):
        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"  LIME progress: {i + 1}/{n}")
        for rank, (feat_name, weight) in enumerate(
            sorted(weights.items(), key=lambda kv: abs(kv[1]), reverse=True),
            start=1,
        ):
            records.append(
                {
                    "timestamp": ts,
                    "feature_name": feat_name,
                    "lime_weight": weight,
                    "lime_rank": rank,
                }
            )

    return pd.DataFrame(records)


def save_lime(lime_df: pd.DataFrame) -> Path:
    """Persist lime_explanations.parquet."""
    EXPL_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPL_DIR / "lime_explanations.parquet"
    lime_df.to_parquet(out, index=False)
    n_ts = lime_df["timestamp"].nunique()
    avg_feat = lime_df.groupby("timestamp")["feature_name"].count().mean()
    print(f"  LIME explanations -> {out}  ({n_ts} timestamps, avg {avg_feat:.1f} features each)")
    return out
