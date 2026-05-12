"""
score_all.py — Phase-3 orchestrator: train both models and produce scores.

Run order:
  1. Isolation Forest   (fast, ~5 s on CPU for 1.6 M rows)
  2. LSTM Autoencoder   (slower, ~5-30 min on CPU depending on stride/epochs)
  3. Align scores by timestamp (outer join)
  4. Compute agreement column
  5. Save all artefacts + training log

Memory notes:
  sequences_train.npy (stride=5): ~528 MB — fits in RAM.
  If you used stride=1 at Phase 2, peak RAM here can reach ~4 GB.

TensorFlow:
  Python 3.14 is not yet supported by stable TF.  If TensorFlow is missing,
  LSTM training is skipped and LSTM columns in scores.parquet are filled with
  NaN.  Install:  pip install tf-nightly
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

from src.models.iforest import (
    compute_if_threshold,
    load_iforest,
    save_iforest,
    score_iforest,
    train_iforest,
)

try:
    from src.models.lstm_ae import (
        TF_AVAILABLE,
        compute_lstm_threshold,
        load_lstm_ae,
        save_lstm_ae,
        score_lstm_ae,
        train_lstm_ae,
    )
except ImportError:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _section(log_print, title: str) -> None:
    log_print()
    log_print("-" * 70)
    log_print(f"  {title}")
    log_print("-" * 70)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(epochs: int = 10, batch_size: int = 512) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    log = StringIO()
    t_global = time.perf_counter()

    def log_print(msg: str = "") -> None:
        print(msg)
        log.write(msg + "\n")

    log_print("=" * 70)
    log_print("Phase 3 — Model Training & Scoring")
    log_print(f"Started: {pd.Timestamp.now().isoformat(timespec='seconds')}")
    log_print("=" * 70)

    # ── 1. Load scaled tabular features ────────────────────────────────────
    _section(log_print, "Isolation Forest")

    log_print("Loading tabular features ...")
    X_train_tab = pd.read_parquet(PROC / "features_tabular_train_scaled.parquet")
    X_test_tab = pd.read_parquet(PROC / "features_tabular_test_scaled.parquet")
    log_print(f"  Train: {X_train_tab.shape}  |  Test: {X_test_tab.shape}")

    # ── 2. Train Isolation Forest ───────────────────────────────────────────
    t0 = time.perf_counter()
    log_print("Training Isolation Forest ...")
    model_if = train_iforest(X_train_tab)
    t_if = time.perf_counter() - t0
    log_print(f"  Training time: {t_if:.1f}s")

    # ── 3. Compute threshold from TRAIN scores only ─────────────────────────
    threshold_if = compute_if_threshold(model_if, X_train_tab, percentile=95)
    log_print(f"  IF threshold (95th pct on train): {threshold_if:.6f}")

    # ── 4. Score full dataset (train + test) ────────────────────────────────
    all_tab = pd.concat([X_train_tab, X_test_tab])
    log_print(f"Scoring IF on {len(all_tab):,} rows ...")
    scores_if = score_iforest(model_if, all_tab, threshold_if)

    train_flags_if = int(scores_if.loc[scores_if.index < pd.Timestamp(
        json.loads((PROC / "split_metadata.json").read_text())["split_timestamp"]
    ), "if_flag"].fillna(0).sum())
    test_flags_if = int(scores_if["if_flag"].fillna(0).sum()) - train_flags_if
    flag_rate_train = train_flags_if / len(X_train_tab)
    log_print(f"  Train flags : {train_flags_if:,}  ({flag_rate_train:.2%})")
    log_print(f"  Test  flags : {test_flags_if:,}")

    save_iforest(model_if, threshold_if, MODELS_DIR / "iforest.pkl")

    # ── 5. LSTM Autoencoder ─────────────────────────────────────────────────
    _section(log_print, "LSTM Autoencoder")

    history_dict: dict = {}
    threshold_lstm: float = float("nan")
    scores_lstm = pd.DataFrame(
        {"lstm_recon_error": pd.array([], dtype=float),
         "lstm_flag": pd.array([], dtype=float)},
    )

    if not TF_AVAILABLE:
        log_print("[SKIP] TensorFlow not available on Python 3.14.")
        log_print("  Install: pip install tf-nightly   OR use Python 3.12")
        log_print("  LSTM columns in scores.parquet will be NaN.")
    else:
        log_print("Loading sequence artefacts ...")
        X_train_seq = np.load(PROC / "sequences_train.npy")
        X_test_seq = np.load(PROC / "sequences_test.npy")
        train_idx = pd.read_parquet(PROC / "sequences_train_index.parquet").index
        test_idx = pd.read_parquet(PROC / "sequences_test_index.parquet").index
        log_print(f"  Train seq: {X_train_seq.shape}  |  Test seq: {X_test_seq.shape}")

        t0 = time.perf_counter()
        log_print(f"Training LSTM-AE (epochs={epochs}, batch={batch_size}) ...")
        model_lstm, history_dict = train_lstm_ae(
            X_train_seq, epochs=epochs, batch_size=batch_size
        )
        t_lstm = time.perf_counter() - t0
        log_print(f"  Training time   : {t_lstm:.1f}s")
        log_print(f"  Epochs completed: {history_dict['epochs_run']}")
        log_print(f"  Final train loss: {history_dict['loss'][-1]:.6f}")
        log_print(f"  Final val loss  : {history_dict['val_loss'][-1]:.6f}")

        log_print("Computing LSTM threshold from train sequences ...")
        threshold_lstm = compute_lstm_threshold(model_lstm, X_train_seq, percentile=95)
        log_print(f"  LSTM threshold (95th pct): {threshold_lstm:.6f}")

        log_print("Scoring LSTM-AE on train + test sequences ...")
        scores_lstm_train = score_lstm_ae(
            model_lstm, X_train_seq, train_idx, threshold_lstm
        )
        scores_lstm_test = score_lstm_ae(
            model_lstm, X_test_seq, test_idx, threshold_lstm
        )
        scores_lstm = pd.concat([scores_lstm_train, scores_lstm_test])

        lstm_train_flags = int(scores_lstm_train["lstm_flag"].sum())
        lstm_test_flags = int(scores_lstm_test["lstm_flag"].sum())
        log_print(f"  Train LSTM flags: {lstm_train_flags:,}  ({lstm_train_flags/len(scores_lstm_train):.2%})")
        log_print(f"  Test  LSTM flags: {lstm_test_flags:,}")

        save_lstm_ae(model_lstm, threshold_lstm, MODELS_DIR)

        # Save training history
        history_path = MODELS_DIR / "lstm_ae_history.json"
        history_path.write_text(json.dumps(history_dict, indent=2), encoding="utf-8")
        log_print(f"  History -> {history_path}")

<<<<<<< HEAD
    # ── 6. Align scores by timestamp (left join on IF index) ────────────────
    # Left join keeps every IF-scored minute; LSTM scores fill in where
    # timestamps match. Early LSTM windows (before tabular features start)
    # are intentionally dropped — they have no IF score and no context.
    _section(log_print, "Alignment & Agreement")

    combined = scores_if.join(scores_lstm, how="left")
=======
    # ── 6. Align scores by timestamp (outer join) ───────────────────────────
    _section(log_print, "Alignment & Agreement")

    combined = scores_if.join(scores_lstm, how="outer")
>>>>>>> f57149dcdca90c22ba533cda4d8625ed83e75941

    if TF_AVAILABLE and len(scores_lstm) > 0:
        both_valid = combined["if_flag"].notna() & combined["lstm_flag"].notna()
        combined["agreement"] = np.where(
            both_valid,
            ((combined["if_flag"] == 1) & (combined["lstm_flag"] == 1)).astype(float),
            np.nan,
        )
        n_agree = int(combined["agreement"].fillna(0).sum())
        n_both_valid = int(both_valid.sum())
        agree_rate = n_agree / n_both_valid if n_both_valid > 0 else 0.0
        log_print(f"  Jointly scored timestamps: {n_both_valid:,}")
        log_print(f"  Agreement (both flag=1)  : {n_agree:,}  ({agree_rate:.2%})")
    else:
        combined["agreement"] = np.nan
        log_print("  Agreement: N/A (LSTM not available)")

    # ── 7. Save scores.parquet ──────────────────────────────────────────────
    scores_path = MODELS_DIR / "scores.parquet"
    combined.to_parquet(scores_path)
    log_print(f"\nScores -> {scores_path}  shape={combined.shape}")
<<<<<<< HEAD
    log_print(f"  (IF rows: {len(scores_if):,}  |  LSTM rows joined: {combined['lstm_recon_error'].notna().sum():,})")
=======
>>>>>>> f57149dcdca90c22ba533cda4d8625ed83e75941

    # ── 8. Top-10 agreement preview ─────────────────────────────────────────
    if TF_AVAILABLE and "agreement" in combined.columns:
        agreed = combined[combined["agreement"] == 1].nlargest(10, "lstm_recon_error")
        if not agreed.empty:
            log_print("\nTop 10 timestamps where both models agree (by LSTM error):")
            log_print(agreed[["if_anomaly_score", "lstm_recon_error"]].to_string())

    # ── 9. Summary ───────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_global
    _section(log_print, "Summary")
    log_print(f"  Total runtime     : {t_total:.1f}s")
    log_print(f"  IF flags (total)  : {int(scores_if['if_flag'].fillna(0).sum()):,}")
    if TF_AVAILABLE and len(scores_lstm) > 0:
        log_print(f"  LSTM flags (total): {int(scores_lstm['lstm_flag'].fillna(0).sum()):,}")
    log_print(f"  scores.parquet    : {combined.shape}")
    log_print(f"  Columns           : {list(combined.columns)}")

    # Save log
    log_path = MODELS_DIR / "training_log.txt"
    log_path.write_text(log.getvalue(), encoding="utf-8")
    print(f"\nLog -> {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3 model trainer")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)
