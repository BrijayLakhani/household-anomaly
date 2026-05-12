"""
counterfactuals.py — DiCE counterfactual explanations for anomalies.

Counterfactuals answer: "What is the minimum change to this anomalous reading
that would make the model classify it as normal?"

Only the 7 raw sensor features are allowed to vary.  Engineered features
(lags, rolling stats, time, ratios) are derived quantities — changing them
independently of the raw values would be physically meaningless.

Implementation notes
--------------------
- method='random': fast, model-agnostic; generates candidate CFs by random
  sampling and filters those that satisfy desired_class=0 (normal).
- IsolationForest is wrapped in a sklearn-compatible classifier that exposes
  predict() and predict_proba() so DiCE can interact with it.
- CF generation can fail for extreme anomalies where no random sample in
  the neighbourhood satisfies the normal class.  Failures are logged to
  data/explanations/cf_failures.txt rather than crashing the pipeline.
- The training DataFrame passed to DiCE is subsampled to 50k rows to keep
  memory and setup time reasonable while preserving feature statistics.

Output file
-----------
data/explanations/counterfactuals.parquet
    Long format: (timestamp, cf_index, feature_name, original_value,
                  cf_value, abs_change)
    Only the 7 RAW_COLS appear as feature_name rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
EXPL_DIR = _ROOT / "data" / "explanations"
MODELS_DIR = _ROOT / "data" / "models"

RAW_COLS: list[str] = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

_N_CFS = 3
_DICE_TRAIN_ROWS = 50_000


# ---------------------------------------------------------------------------
# IsolationForest sklearn-compatible wrapper
# ---------------------------------------------------------------------------


class _IForestClassifier:
    """
    Thin wrapper giving IsolationForest a binary-classifier interface.

    predict(X) returns 1 (anomaly) / 0 (normal) using the stored threshold.
    predict_proba(X) returns a soft probability via a logistic-sigmoid
    scaled around the threshold — a heuristic, not a calibrated probability.
    """

    def __init__(self, iforest: Any, threshold: float) -> None:
        self.iforest = iforest
        self.threshold = threshold

    def fit(self, X: Any, y: Any = None) -> "_IForestClassifier":
        return self  # already fitted

    def predict(self, X: Any) -> np.ndarray:
        if hasattr(X, "values"):
            X = X.values
        scores = -self.iforest.score_samples(X.astype(np.float64))
        return (scores > self.threshold).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(X, "values"):
            X = X.values
        scores = -self.iforest.score_samples(X.astype(np.float64))
        # Sigmoid centred on threshold; scale=10 gives reasonable separation
        p_anomaly = 1.0 / (1.0 + np.exp(-(scores - self.threshold) * 10.0))
        p_anomaly = np.clip(p_anomaly, 1e-6, 1 - 1e-6)
        return np.column_stack([1.0 - p_anomaly, p_anomaly])


# ---------------------------------------------------------------------------
# DiCE explainer construction
# ---------------------------------------------------------------------------


def build_dice_explainer(
    iforest_model: Any,
    threshold: float,
    features_train: pd.DataFrame,
    feature_names: list[str],
) -> Any:
    """
    Build a dice_ml.Dice explainer using the wrapped IsolationForest.

    Parameters
    ----------
    iforest_model:
        Fitted sklearn IsolationForest.
    threshold:
        IF anomaly-score threshold (95th percentile on train).
    features_train:
        Full scaled training features (no NaN).  Subsampled to _DICE_TRAIN_ROWS.
    feature_names:
        Ordered list of 24 tabular feature names.

    Returns
    -------
    dice_ml.Dice instance ready for generate_counterfactuals().
    """
    import dice_ml

    # Subsample training data — DiCE only needs feature ranges and class balance
    sample = features_train.sample(
        n=min(_DICE_TRAIN_ROWS, len(features_train)), random_state=42
    )
    train_scores = -iforest_model.score_samples(sample.values)
    sample = sample.copy()
    sample["is_anomaly"] = (train_scores > threshold).astype(int)

    # Ensure both classes are represented (at least 10 of the minority class)
    n_anomaly = sample["is_anomaly"].sum()
    if n_anomaly < 10:
        extra_idx = features_train.index[
            (-iforest_model.score_samples(features_train.values) > threshold)
        ][:50]
        extra = features_train.loc[extra_idx].copy()
        extra["is_anomaly"] = 1
        sample = pd.concat([sample, extra])

    d = dice_ml.Data(
        dataframe=sample,
        continuous_features=feature_names,
        outcome_name="is_anomaly",
    )
    wrapped_model = _IForestClassifier(iforest_model, threshold)
    m = dice_ml.Model(model=wrapped_model, backend="sklearn")
    return dice_ml.Dice(d, m, method="random")


# ---------------------------------------------------------------------------
# Single-instance counterfactual generation
# ---------------------------------------------------------------------------


def generate_counterfactuals(
    dice_exp: Any,
    features_row: pd.DataFrame,
    n_cfs: int = _N_CFS,
) -> pd.DataFrame | None:
    """
    Generate counterfactual explanations for one anomalous instance.

    Parameters
    ----------
    dice_exp:
        dice_ml.Dice explainer.
    features_row:
        Single-row DataFrame with 24 feature columns (the anomaly to explain).
    n_cfs:
        Number of counterfactuals to generate.

    Returns
    -------
    DataFrame with columns (cf_index, feature_name, original_value,
    cf_value, abs_change, change_summary) or None on failure.
    """
    cf_result = dice_exp.generate_counterfactuals(
        features_row,
        total_CFs=n_cfs,
        desired_class=0,
        features_to_vary=RAW_COLS,
    )
    cf_df = cf_result.cf_examples_list[0].final_cfs_df
    if cf_df is None or len(cf_df) == 0:
        return None

    # Extract only the feature columns (drop outcome column)
    feat_cols = [c for c in cf_df.columns if c in features_row.columns]
    orig_vals = features_row[feat_cols].values[0]

    records = []
    for cf_idx in range(len(cf_df)):
        cf_vals = cf_df.iloc[cf_idx][feat_cols].values.astype(float)
        for feat, orig, cf_v in zip(feat_cols, orig_vals, cf_vals):
            if feat in RAW_COLS:
                records.append(
                    {
                        "cf_index": cf_idx,
                        "feature_name": feat,
                        "original_value": float(orig),
                        "cf_value": float(cf_v),
                        "abs_change": float(abs(cf_v - orig)),
                    }
                )
    return pd.DataFrame(records) if records else None


# ---------------------------------------------------------------------------
# Batch counterfactual generation
# ---------------------------------------------------------------------------


def compute_counterfactuals_for_all(
    dice_exp: Any,
    features: pd.DataFrame,
    selected_anomalies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate counterfactuals for all non-normal selected anomalies.

    Skips the 50 normal control points.  Logs failures to
    data/explanations/cf_failures.txt without raising exceptions.

    Parameters
    ----------
    dice_exp:
        dice_ml.Dice explainer.
    features:
        Full scaled tabular feature DataFrame (train + test).
    selected_anomalies:
        DataFrame from select_top_anomalies (has 'timestamp', 'selection_group').

    Returns
    -------
    Long-format DataFrame:
        (timestamp, cf_index, feature_name, original_value, cf_value, abs_change)
    """
    EXPL_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    all_records: list[pd.DataFrame] = []

    anomalies = selected_anomalies[selected_anomalies["selection_group"] != "normal"]
    n = len(anomalies)
    print(f"  Generating CFs for {n} anomalies (skipping 50 normal points) ...")

    for i, (_, row) in enumerate(anomalies.iterrows()):
        ts = row["timestamp"]
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  CF progress: {i + 1}/{n}")
        try:
            feat_row = features.loc[[ts]]
            result = generate_counterfactuals(dice_exp, feat_row, n_cfs=_N_CFS)
            if result is None:
                failures.append(f"{ts}: no valid CF found")
                continue
            result.insert(0, "timestamp", ts)
            all_records.append(result)
        except Exception as exc:
            failures.append(f"{ts}: {exc}")

    if failures:
        fail_path = EXPL_DIR / "cf_failures.txt"
        fail_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"  CF failures ({len(failures)}): see {fail_path}")

    if not all_records:
        return pd.DataFrame(
            columns=["timestamp", "cf_index", "feature_name",
                     "original_value", "cf_value", "abs_change"]
        )

    result_df = pd.concat(all_records, ignore_index=True)
    success_rate = (n - len(failures)) / n
    avg_changes = (
        result_df[result_df["abs_change"] > 1e-6]
        .groupby(["timestamp", "cf_index"])["feature_name"]
        .count()
        .mean()
    )
    print(
        f"  CF success rate: {success_rate:.0%}  "
        f"avg features changed per CF: {avg_changes:.1f}"
    )
    return result_df


def save_counterfactuals(cf_df: pd.DataFrame) -> Path:
    """Persist counterfactuals.parquet."""
    EXPL_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPL_DIR / "counterfactuals.parquet"
    cf_df.to_parquet(out, index=False)
    print(f"  Counterfactuals   -> {out}  shape={cf_df.shape}")
    return out
