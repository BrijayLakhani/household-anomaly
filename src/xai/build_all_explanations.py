"""
build_all_explanations.py — Phase-4 XAI orchestrator.

Run order
---------
1. Load models (IF + scaler) and feature data
2. Load scores.parquet
3. select_top_anomalies          → selected_anomalies.parquet
4. TreeSHAP                      → shap_values.parquet
                                   shap_global_importance.parquet
                                   shap_metadata.json
5. LIME                          → lime_explanations.parquet
6. DiCE counterfactuals          → counterfactuals.parquet
                                   cf_failures.txt  (if any)
7. Save xai_build_log.txt with per-step timings and summary

All outputs go to data/explanations/.
The dashboard reads directly from these cached parquet files — no model
inference at dashboard serve time.

Runtime budget: 25 minutes on CPU.
  TreeSHAP : <2 min
  LIME     : 3-7 min (parallelised with joblib threads)
  DiCE     : 10-15 min
"""

from __future__ import annotations

import json
import sys
import time
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

PROC = _ROOT / "data" / "processed"
MODELS_DIR = _ROOT / "data" / "models"
EXPL_DIR = _ROOT / "data" / "explanations"

from src.xai.select_anomalies import save_selected_anomalies, select_top_anomalies
from src.xai.shap_iforest import compute_treeshap, save_treeshap
from src.xai.lime_explainer import (
    build_lime_explainer,
    compute_lime_for_all,
    save_lime,
)
from src.xai.counterfactuals import (
    build_dice_explainer,
    compute_counterfactuals_for_all,
    save_counterfactuals,
)
from src.models.iforest import load_iforest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _section(log_print, title: str) -> None:
    log_print()
    log_print("-" * 70)
    log_print(f"  {title}")
    log_print("-" * 70)


def _hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    EXPL_DIR.mkdir(parents=True, exist_ok=True)

    log = StringIO()
    t_global = time.perf_counter()

    def log_print(msg: str = "") -> None:
        print(msg)
        log.write(msg + "\n")

    log_print("=" * 70)
    log_print("Phase 4 - XAI Layer")
    log_print(f"Started: {pd.Timestamp.now().isoformat(timespec='seconds')}")
    log_print("=" * 70)

    # ── 1. Load models and data ─────────────────────────────────────────────
    _section(log_print, "Loading artefacts")

    log_print("Loading Isolation Forest ...")
    model_if, threshold_if = load_iforest(MODELS_DIR / "iforest.pkl")
    log_print(f"  IF threshold: {threshold_if:.6f}")

    log_print("Loading scaler ...")
    scaler_tab = joblib.load(PROC / "scaler_tabular.pkl")

    log_print("Loading tabular features ...")
    X_train = pd.read_parquet(PROC / "features_tabular_train_scaled.parquet")
    X_test = pd.read_parquet(PROC / "features_tabular_test_scaled.parquet")
    all_features = pd.concat([X_train, X_test])
    feature_names = list(X_train.columns)
    log_print(f"  Features: {all_features.shape}")

    log_print("Loading scores.parquet ...")
    scores = pd.read_parquet(MODELS_DIR / "scores.parquet")
    log_print(f"  Scores : {scores.shape}")

    # ── 2. Select anomalies ─────────────────────────────────────────────────
    _section(log_print, "Anomaly Selection")
    t0 = time.perf_counter()

    selected = select_top_anomalies(scores, n=200)
    save_selected_anomalies(selected)
    selected_ts = pd.DatetimeIndex(selected["timestamp"])

    t_select = time.perf_counter() - t0
    for _, row in selected["selection_group"].value_counts().items():
        pass
    counts = selected["selection_group"].value_counts().to_dict()
    log_print(f"  if_only={counts.get('if_only',0)}  lstm_only={counts.get('lstm_only',0)}  both={counts.get('both',0)}  normal={counts.get('normal',0)}")
    log_print(f"  Selection time: {_hms(t_select)}")

    # ── 3. TreeSHAP ─────────────────────────────────────────────────────────
    _section(log_print, "TreeSHAP")
    t0 = time.perf_counter()

    shap_long, global_importance, expected_value = compute_treeshap(
        model_if, all_features, selected_ts
    )
    save_treeshap(shap_long, global_importance, expected_value)

    t_shap = time.perf_counter() - t0
    log_print(f"  SHAP rows saved : {len(shap_long):,}  (expected {len(selected_ts) * len(feature_names):,})")
    log_print(f"  SHAP time       : {_hms(t_shap)}")

    # ── 4. LIME ─────────────────────────────────────────────────────────────
    _section(log_print, "LIME")
    t0 = time.perf_counter()

    lime_explainer = build_lime_explainer(X_train, feature_names)
    lime_df = compute_lime_for_all(
        lime_explainer, model_if, all_features, selected_ts, n_jobs=-1
    )
    save_lime(lime_df)

    t_lime = time.perf_counter() - t0
    avg_feats = lime_df.groupby("timestamp")["feature_name"].count().mean()
    log_print(f"  LIME rows saved : {len(lime_df):,}")
    log_print(f"  Avg features/ts : {avg_feats:.1f}")
    log_print(f"  LIME time       : {_hms(t_lime)}")

    # ── 5. DiCE Counterfactuals ─────────────────────────────────────────────
    _section(log_print, "DiCE Counterfactuals")
    t0 = time.perf_counter()

    dice_exp = build_dice_explainer(model_if, threshold_if, X_train, feature_names)
    cf_df = compute_counterfactuals_for_all(dice_exp, all_features, selected)
    save_counterfactuals(cf_df)

    t_dice = time.perf_counter() - t0
    n_anomalies = len(selected[selected["selection_group"] != "normal"])
    n_success = cf_df["timestamp"].nunique() if len(cf_df) > 0 else 0
    success_rate = n_success / n_anomalies if n_anomalies > 0 else 0.0
    log_print(f"  CF rows saved    : {len(cf_df):,}")
    log_print(f"  Success rate     : {n_success}/{n_anomalies}  ({success_rate:.0%})")
    log_print(f"  DiCE time        : {_hms(t_dice)}")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_global
    _section(log_print, "Summary")

    if t_total > 1500:
        log_print(f"  [WARNING] Total runtime {_hms(t_total)} exceeds 25-min budget.")

    log_print(f"  Total runtime  : {_hms(t_total)}")
    log_print(f"  Selection      : {_hms(t_select)}")
    log_print(f"  TreeSHAP       : {_hms(t_shap)}")
    log_print(f"  LIME           : {_hms(t_lime)}")
    log_print(f"  DiCE           : {_hms(t_dice)}")
    log_print()
    log_print("  Output files:")
    for name in [
        "selected_anomalies.parquet",
        "shap_values.parquet",
        "shap_global_importance.parquet",
        "shap_metadata.json",
        "lime_explanations.parquet",
        "counterfactuals.parquet",
    ]:
        p = EXPL_DIR / name
        size = f"{p.stat().st_size / 1024:.0f} KB" if p.exists() else "MISSING"
        log_print(f"    {name:<40} {size}")

    log_path = EXPL_DIR / "xai_build_log.txt"
    log_path.write_text(log.getvalue(), encoding="utf-8")
    print(f"\nLog -> {log_path}")


if __name__ == "__main__":
    main()
