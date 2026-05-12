"""
shap_iforest.py — TreeSHAP explanations for the Isolation Forest model.

Uses shap.TreeExplainer with feature_perturbation='tree_path_dependent':
  - Exact (not approximate) SHAP values via the tree structure
  - No background dataset needed — uses in-tree path statistics
  - For IsolationForest, SHAP explains -score_samples(x) = if_anomaly_score
    so higher SHAP value for a feature = that feature pushed the point
    toward being more anomalous

Output files
------------
data/explanations/shap_values.parquet
    Long format: (timestamp, feature_name, shap_value, feature_value)
    200 timestamps × 24 features = 4800 rows
data/explanations/shap_global_importance.parquet
    (feature_name, mean_abs_shap, rank)  — aggregated over all 200 points
data/explanations/shap_metadata.json
    expected_value (float) needed by tests to verify SHAP additivity property
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

_ROOT = Path(__file__).resolve().parents[2]
EXPL_DIR = _ROOT / "data" / "explanations"


def compute_treeshap(
    iforest_model,
    features: pd.DataFrame,
    selected_timestamps: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Compute per-feature SHAP values for the selected timestamps.

    Parameters
    ----------
    iforest_model:
        Fitted sklearn IsolationForest.
    features:
        Full scaled tabular feature DataFrame (train + test concatenated),
        indexed by DatetimeIndex.  Must contain all selected_timestamps.
    selected_timestamps:
        DatetimeIndex of the 200 selected points.

    Returns
    -------
    shap_long:
        Long-format DataFrame with columns
        (timestamp, feature_name, shap_value, feature_value).
    global_importance:
        DataFrame (feature_name, mean_abs_shap, rank) sorted by rank asc.
    expected_value:
        Model base value (scalar float).
    """
    X = features.loc[selected_timestamps]

    print(f"  Running TreeSHAP on {len(X)} instances × {X.shape[1]} features ...")
    explainer = shap.TreeExplainer(
        iforest_model,
        feature_perturbation="tree_path_dependent",
    )
    shap_values = explainer.shap_values(X)   # (n_timestamps, n_features)
    expected_value = float(explainer.expected_value)
    print(f"  TreeSHAP done.  expected_value = {expected_value:.6f}")

    feature_names = list(X.columns)
    n_ts, n_feat = shap_values.shape

    # Per-timestamp model output: expected_value + sum_i(shap_i)
    # This is what SHAP's TreeExplainer models (path lengths for IF),
    # NOT the normalised if_anomaly_score.  Stored so the test can verify
    # the additivity property without assuming a particular output scale.
    shap_model_output = expected_value + shap_values.sum(axis=1)  # (n_ts,)

    # Long format
    ts_repeated = np.repeat(selected_timestamps, n_feat)
    feat_repeated = feature_names * n_ts
    shap_long = pd.DataFrame(
        {
            "timestamp": ts_repeated,
            "feature_name": feat_repeated,
            "shap_value": shap_values.ravel(),
            "feature_value": X.values.ravel(),
            # Same model output repeated 24× per timestamp for easy groupby queries
            "shap_model_output": np.repeat(shap_model_output, n_feat),
        }
    )

    # Global importance: mean |SHAP| per feature across selected points
    mean_abs = np.abs(shap_values).mean(axis=0)
    global_importance = pd.DataFrame(
        {"feature_name": feature_names, "mean_abs_shap": mean_abs}
    )
    global_importance = global_importance.sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)
    global_importance["rank"] = global_importance.index + 1

    return shap_long, global_importance, expected_value


def save_treeshap(
    shap_long: pd.DataFrame,
    global_importance: pd.DataFrame,
    expected_value: float,
) -> None:
    """Persist all three SHAP artefacts to data/explanations/."""
    EXPL_DIR.mkdir(parents=True, exist_ok=True)

    shap_path = EXPL_DIR / "shap_values.parquet"
    shap_long.to_parquet(shap_path, index=False)
    n_ts = shap_long["timestamp"].nunique()
    n_feat = shap_long["feature_name"].nunique()
    print(f"  SHAP values       -> {shap_path}  ({n_ts} timestamps x {n_feat} features)")

    gi_path = EXPL_DIR / "shap_global_importance.parquet"
    global_importance.to_parquet(gi_path, index=False)
    print(f"  SHAP global imp.  -> {gi_path}")
    print(f"  Top-5 features:")
    for _, row in global_importance.head(5).iterrows():
        print(f"    {int(row['rank'])}.  {row['feature_name']:<30}  {row['mean_abs_shap']:.6f}")

    meta_path = EXPL_DIR / "shap_metadata.json"
    meta_path.write_text(
        json.dumps({"expected_value": expected_value}, indent=2),
        encoding="utf-8",
    )
    print(f"  SHAP metadata     -> {meta_path}")
