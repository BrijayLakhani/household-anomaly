"""
synthetic_eval.py — Controlled evaluation using injected anomalies.

Because the UCI dataset has no ground-truth anomaly labels, we evaluate
the models by injecting known anomalies into a clean window from the test
period and measuring detection performance.

Anomaly taxonomy follows Chandola, Banerjee & Kumar (2009)
"Anomaly Detection: A Survey", ACM Computing Surveys, 41(3):
  - POINT anomaly      : single observation far from all others.
  - CONTEXTUAL anomaly : normal value globally, anomalous given context
                         (e.g. high kitchen usage at 3 AM).
  - COLLECTIVE anomaly : sequence of values that is anomalous as a group
                         even though individual points may be normal
                         (e.g. sustained voltage drop).

Injection choices:
  POINT (50 events)       : spike Global_active_power × 8 for one minute.
                            Factor 8 is well above the 99th percentile of the
                            raw distribution, making these clearly anomalous.
  CONTEXTUAL (20 events)  : Sub_metering_1 = 40 Wh during night hours
                            (00:00–05:59).  Kitchen meter reading 40 Wh/min
                            at 3 AM is locally anomalous despite being a
                            feasible daytime value.
  COLLECTIVE (10 events)  : Voltage − 15 V sustained for 30 consecutive
                            minutes.  A 15 V drop (from ~241 V to ~226 V)
                            sustained for half an hour indicates a grid
                            disturbance, which is normal for 1 second but
                            anomalous as a sustained pattern.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

PROC = _ROOT / "data" / "processed"
MODELS_DIR = _ROOT / "data" / "models"

from src.features.tabular import build_tabular_features
from src.models.iforest import load_iforest, score_iforest

try:
    import joblib
    from src.models.lstm_ae import TF_AVAILABLE, load_lstm_ae, score_lstm_ae

    if TF_AVAILABLE:
        import tensorflow as tf
except ImportError:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# anomaly injection
# ---------------------------------------------------------------------------

def inject_anomalies(
    df_clean: pd.DataFrame,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inject synthetic anomalies into a 7-day clean window from the test period.

    A 7-day window (10,080 minutes) is large enough to include all three
    anomaly types at realistic densities while remaining computationally cheap
    for feature engineering and sequence building.

    Parameters
    ----------
    df_clean:
        Full cleaned DataFrame (clean.parquet).  The function extracts a 7-day
        window that starts at the beginning of the test period.
    seed:
        NumPy random seed for reproducible injection positions.

    Returns
    -------
    injected_df:
        7-day DataFrame with anomalies injected.  Original data is unchanged.
    ground_truth:
        DataFrame indexed by the same DatetimeIndex with columns:
        anomaly_type ('point'/'contextual'/'collective'/'normal'), is_anomaly (0/1).
    """
    meta = json.loads((PROC / "split_metadata.json").read_text())
    test_start = pd.Timestamp(meta["test_start"])
    window_end = test_start + pd.Timedelta("7D")

    window = df_clean.loc[test_start:window_end].copy()
    injected = window.copy()
    rng = np.random.default_rng(seed)

    gt = pd.DataFrame(
        {"anomaly_type": "normal", "is_anomaly": 0},
        index=window.index,
    )

    # ── POINT anomalies (50 single-minute spikes) ───────────────────────────
    point_idx = rng.choice(len(window), size=50, replace=False)
    point_ts = window.index[point_idx]
    injected.loc[point_ts, "Global_active_power"] = (
        window.loc[point_ts, "Global_active_power"] * 8.0
    ).clip(lower=0.01)   # guard against zero × 8 = 0
    gt.loc[point_ts, "anomaly_type"] = "point"
    gt.loc[point_ts, "is_anomaly"] = 1

    # ── CONTEXTUAL anomalies (20 night-time kitchen spikes) ─────────────────
    night_mask = window.index.hour < 6
    night_positions = np.where(night_mask)[0]
    if len(night_positions) >= 20:
        ctx_idx = rng.choice(night_positions, size=20, replace=False)
        ctx_ts = window.index[ctx_idx]
        injected.loc[ctx_ts, "Sub_metering_1"] = 40.0
        gt.loc[ctx_ts, "anomaly_type"] = "contextual"
        gt.loc[ctx_ts, "is_anomaly"] = 1

    # ── COLLECTIVE anomalies (10 × 30-min sustained voltage drops) ──────────
    max_start = len(window) - 30
    coll_starts = rng.choice(max_start, size=10, replace=False)
    for s in coll_starts:
        ts_range = window.index[s: s + 30]
        injected.loc[ts_range, "Voltage"] = (
            window.loc[ts_range, "Voltage"] - 15.0
        ).clip(lower=0.0)
        # Only mark as collective where not already labelled point/contextual
        new_anom = gt.loc[ts_range, "is_anomaly"] == 0
        gt.loc[ts_range[new_anom], "anomaly_type"] = "collective"
        gt.loc[ts_range[new_anom], "is_anomaly"] = 1

    n_point = int((gt["anomaly_type"] == "point").sum())
    n_ctx = int((gt["anomaly_type"] == "contextual").sum())
    n_coll = int((gt["anomaly_type"] == "collective").sum())
    n_total = int(gt["is_anomaly"].sum())
    print(
        f"Injected: {n_point} point, {n_ctx} contextual, "
        f"{n_coll} collective ({n_total} total anomalous minutes "
        f"out of {len(window):,})"
    )
    return injected, gt


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute precision, recall, F1 for binary arrays."""
    if y_true.sum() == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_models(
    injected_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:
    """
    Score both models on the injected window and compute detection metrics.

    Features and sequences are built from injected_df using the ALREADY
    FITTED scalers (no refitting — that would invalidate the test).

    Parameters
    ----------
    injected_df:
        7-day DataFrame with synthetic anomalies from inject_anomalies().
    ground_truth:
        DataFrame with is_anomaly and anomaly_type columns.

    Returns
    -------
    metrics dict with precision, recall, F1 per anomaly type and overall,
    for each model.
    """
    # Load trained models and scalers
    model_if, threshold_if = load_iforest(MODELS_DIR / "iforest.pkl")
    scaler_tab = joblib.load(PROC / "scaler_tabular.pkl")

    # ── IF evaluation (minute-level) ─────────────────────────────────────
    print("Building tabular features on injected window ...")
    raw_feat = build_tabular_features(injected_df)
    raw_feat_clean = raw_feat.dropna()

    from pandas import DataFrame as _DF
    scaled_vals = scaler_tab.transform(raw_feat_clean)
    feat_scaled = _DF(
        scaled_vals, index=raw_feat_clean.index, columns=raw_feat_clean.columns
    )

    scores_if = score_iforest(model_if, feat_scaled, threshold_if)
    aligned_if = ground_truth.join(scores_if, how="inner")
    y_true_if = aligned_if["is_anomaly"].values
    y_pred_if = aligned_if["if_flag"].fillna(0).astype(int).values

    results: dict = {"isolation_forest": {}, "lstm_ae": {}}

    # Overall IF metrics
    results["isolation_forest"]["overall"] = _metrics(y_true_if, y_pred_if)

    # Per-type IF metrics
    for atype in ["point", "contextual", "collective"]:
        mask = aligned_if["anomaly_type"].isin([atype, "normal"])
        yt = aligned_if.loc[mask, "is_anomaly"].values
        yp = aligned_if.loc[mask, "if_flag"].fillna(0).astype(int).values
        results["isolation_forest"][atype] = _metrics(yt, yp)

    # ── LSTM evaluation (window-level upsampled to minute-level) ─────────
    if TF_AVAILABLE:
        print("Building LSTM sequences on injected window ...")
        scaler_lstm = joblib.load(PROC / "scaler_lstm.pkl")
        model_lstm, threshold_lstm = load_lstm_ae(MODELS_DIR)

        from src.features.sequences import RAW_COLS, build_sequences

        seqs, seq_ts, _ = build_sequences(
            injected_df, window=60, stride=1, scaler=scaler_lstm
        )
        scores_lstm = score_lstm_ae(model_lstm, seqs, seq_ts, threshold_lstm)

        # Expand window-level flags to minute-level
        # A minute is flagged if it falls within any flagged 60-min window
        lstm_minute = pd.Series(0.0, index=injected_df.index)
        for end_ts, row in scores_lstm.iterrows():
            if row["lstm_flag"] == 1:
                start_ts = end_ts - pd.Timedelta("59min")
                in_window = (injected_df.index >= start_ts) & (
                    injected_df.index <= end_ts
                )
                lstm_minute[in_window] = 1.0

        aligned_lstm = ground_truth.join(
            lstm_minute.rename("lstm_flag"), how="inner"
        )
        y_true_lstm = aligned_lstm["is_anomaly"].values
        y_pred_lstm = aligned_lstm["lstm_flag"].fillna(0).astype(int).values

        results["lstm_ae"]["overall"] = _metrics(y_true_lstm, y_pred_lstm)
        for atype in ["point", "contextual", "collective"]:
            mask = aligned_lstm["anomaly_type"].isin([atype, "normal"])
            yt = aligned_lstm.loc[mask, "is_anomaly"].values
            yp = aligned_lstm.loc[mask, "lstm_flag"].fillna(0).astype(int).values
            results["lstm_ae"][atype] = _metrics(yt, yp)
    else:
        results["lstm_ae"] = {"note": "TensorFlow not available — skipped"}

    return results


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    clean = pd.read_parquet(PROC / "clean.parquet")

    print("Injecting synthetic anomalies ...")
    injected, ground_truth = inject_anomalies(clean, seed=42)

    print("\nEvaluating models ...")
    metrics = evaluate_models(injected, ground_truth)

    # Pretty-print results table
    print("\n" + "=" * 60)
    print("Synthetic Anomaly Detection Results")
    print("=" * 60)
    for model_name, model_results in metrics.items():
        print(f"\n[{model_name}]")
        if isinstance(model_results, dict) and "note" not in model_results:
            header = f"  {'Type':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for atype, m in model_results.items():
                if isinstance(m, dict):
                    p = m.get("precision", float("nan"))
                    r = m.get("recall", float("nan"))
                    f = m.get("f1", float("nan"))
                    print(f"  {atype:<15} {p:>10.3f} {r:>10.3f} {f:>10.3f}")
        else:
            print(f"  {model_results}")

    # Warn on low F1
    for model_name, model_results in metrics.items():
        if isinstance(model_results, dict) and "overall" in model_results:
            f1_overall = model_results["overall"].get("f1", 1.0)
            if isinstance(f1_overall, float) and f1_overall < 0.5:
                print(
                    f"\n[WARNING] {model_name} overall F1={f1_overall:.3f} < 0.5. "
                    "Check model training or threshold."
                )

    # Save
    out_path = MODELS_DIR / "synthetic_eval.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nResults -> {out_path}")
