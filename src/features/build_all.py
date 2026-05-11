"""
build_all.py

Phase-2 feature engineering orchestrator.

Run order matters:
  1. time_split  -- establishes the leakage barrier
  2. tabular     -- computed on full df (lags/rolling are backward-only)
  3. scaler_tab  -- fitted on train rows ONLY, then applied to both splits
  4. sequences   -- LSTM scaler also fitted on train ONLY

Memory note
-----------
Building sequences with stride=1 allocates large arrays:
  Train sequences ~ 1.66 M x 60 x 7 x 4 bytes  ~= 2.8 GB
  Test  sequences ~ 0.41 M x 60 x 7 x 4 bytes  ~= 0.7 GB
  Peak usage (both in memory simultaneously)     ~= 4-5 GB
If RAM is limited, reduce stride (e.g. stride=5) by passing --stride 5.
"""

from __future__ import annotations

import sys
import textwrap
import time
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.sequences import RAW_COLS, build_sequences, save_sequences
from src.features.split import load_metadata, save_metadata, time_split
from src.features.tabular import EXPECTED_N_COLS, build_tabular_features

PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _scale_preserve_nan(
    scaler: StandardScaler,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply a fitted scaler while preserving NaN rows as NaN in the output.

    StandardScaler.transform raises on NaN by default.  Test data contains
    NaN rows (genuine sensor gaps > 5 min) that we keep for the dashboard's
    continuous timeline.  We scale only the fully-observed rows and leave
    the rest as NaN.
    """
    valid = df.notna().all(axis=1)
    result = pd.DataFrame(
        np.nan,
        index=df.index,
        columns=df.columns,
        dtype=np.float64,
    )
    if valid.any():
        result.loc[valid] = scaler.transform(df.loc[valid])
    return result


def _row(label: str, n_in: int | str, n_out: int | str, note: str = "") -> str:
    in_s = f"{n_in:>10,}" if isinstance(n_in, int) else f"{n_in:>10}"
    out_s = f"{n_out:>10,}" if isinstance(n_out, int) else f"{n_out:>10}"
    return f"{label:<40} {in_s}  {out_s}  {note}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(stride: int = 1) -> None:
    log = StringIO()
    t0 = time.perf_counter()

    def log_print(msg: str = "") -> None:
        print(msg)
        log.write(msg + "\n")

    log_print("=" * 72)
    log_print("Phase 2 — Feature Engineering")
    log_print("=" * 72)
    header = _row("Stage", "Rows In", "Rows Out", "Notes")
    separator = "-" * len(header)
    log_print(header)
    log_print(separator)

    # ── 1. Load clean data ──────────────────────────────────────────────────
    clean_path = PROCESSED_DIR / "clean.parquet"
    full_df = pd.read_parquet(clean_path)
    n_full = len(full_df)
    log_print(_row("Load clean.parquet", n_full, n_full, "7 raw cols"))

    # ── 2. Time split ────────────────────────────────────────────────────────
    train_raw, test_raw, metadata = time_split(full_df, train_frac=0.8)
    save_metadata(metadata)
    split_ts = pd.Timestamp(metadata["split_timestamp"])
    log_print(_row("Train split (raw)", n_full, len(train_raw),
                   f"cutoff {metadata['split_timestamp'][:10]}"))
    log_print(_row("Test  split (raw)", n_full, len(test_raw), "last 20%"))

    # ── 3. Tabular features (full df, no split leakage: lags are backward) ──
    log_print()
    log_print("Building tabular features ...")
    full_feat = build_tabular_features(full_df)
    log_print(_row("Tabular features (full)", n_full, len(full_feat),
                   f"{EXPECTED_N_COLS} cols"))

    # Split by the exact cutoff timestamp from metadata
    train_feat = full_feat.loc[full_feat.index < split_ts].copy()
    test_feat = full_feat.loc[full_feat.index >= split_ts].copy()

    # Drop NaN from train ONLY (lag/rolling boundary + original sensor gaps)
    n_train_before = len(train_feat)
    train_feat_clean = train_feat.dropna()
    n_dropped = n_train_before - len(train_feat_clean)
    log_print(_row("Train tabular (after dropna)", n_train_before,
                   len(train_feat_clean),
                   f"dropped {n_dropped:,} NaN rows"))

    # Test keeps NaN rows for continuous dashboard timeline
    n_test_nan = int(test_feat.isna().any(axis=1).sum())
    log_print(_row("Test  tabular (NaN kept)", len(test_feat), len(test_feat),
                   f"{n_test_nan:,} NaN rows preserved"))

    # Save pre-scale versions
    train_feat_clean.to_parquet(PROCESSED_DIR / "features_tabular_train.parquet")
    test_feat.to_parquet(PROCESSED_DIR / "features_tabular_test.parquet")
    log_print("Saved features_tabular_train.parquet + features_tabular_test.parquet")

    # ── 4. Fit tabular scaler on train ONLY ─────────────────────────────────
    log_print()
    log_print("Fitting tabular scaler on train ...")
    scaler_tab = StandardScaler()
    scaler_tab.fit(train_feat_clean)          # sets feature_names_in_ for tests
    joblib.dump(scaler_tab, PROCESSED_DIR / "scaler_tabular.pkl")
    log_print(f"Scaler -> {PROCESSED_DIR / 'scaler_tabular.pkl'}")

    # Scale both splits
    train_scaled = pd.DataFrame(
        scaler_tab.transform(train_feat_clean),
        index=train_feat_clean.index,
        columns=train_feat_clean.columns,
    )
    test_scaled = _scale_preserve_nan(scaler_tab, test_feat)

    train_scaled.to_parquet(PROCESSED_DIR / "features_tabular_train_scaled.parquet")
    test_scaled.to_parquet(PROCESSED_DIR / "features_tabular_test_scaled.parquet")
    log_print("Saved features_tabular_train_scaled.parquet + features_tabular_test_scaled.parquet")

    log_print(_row("Train scaled (no NaN)", len(train_feat_clean),
                   len(train_scaled), "ready for IF"))
    log_print(_row("Test  scaled (NaN kept)", len(test_feat), len(test_scaled),
                   "NaN preserved"))

    # ── 5. LSTM sequences ────────────────────────────────────────────────────
    log_print()
    log_print(f"Building LSTM sequences (window=60, stride={stride}) ...")
    log_print("  [!] Memory warning: stride=1 requires ~3.5 GB RAM for both splits.")

    seqs_tr, idx_tr, scaler_lstm = build_sequences(
        train_raw, window=60, stride=stride, scaler=None
    )
    joblib.dump(scaler_lstm, PROCESSED_DIR / "scaler_lstm.pkl")
    save_sequences(
        seqs_tr, idx_tr,
        PROCESSED_DIR / "sequences_train.npy",
        PROCESSED_DIR / "sequences_train_index.parquet",
    )
    log_print(_row("Train sequences", len(train_raw), len(seqs_tr),
                   f"shape={seqs_tr.shape}"))

    seqs_te, idx_te, _ = build_sequences(
        test_raw, window=60, stride=stride, scaler=scaler_lstm
    )
    save_sequences(
        seqs_te, idx_te,
        PROCESSED_DIR / "sequences_test.npy",
        PROCESSED_DIR / "sequences_test_index.parquet",
    )
    log_print(_row("Test  sequences", len(test_raw), len(seqs_te),
                   f"shape={seqs_te.shape}"))

    # ── 6. Summary ───────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    log_print()
    log_print("=" * 72)
    log_print(f"Phase 2 complete in {elapsed:.1f}s")
    log_print(f"  Train tabular shape : {train_feat_clean.shape}")
    log_print(f"  Test  tabular shape : {test_feat.shape}")
    log_print(f"  Train seq shape     : {seqs_tr.shape}")
    log_print(f"  Test  seq shape     : {seqs_te.shape}")
    log_print(f"  Scaler tabular      : {PROCESSED_DIR / 'scaler_tabular.pkl'}")
    log_print(f"  Scaler LSTM         : {PROCESSED_DIR / 'scaler_lstm.pkl'}")
    log_print(f"  Split timestamp     : {metadata['split_timestamp']}")
    log_print("=" * 72)

    # Write log to disk
    log_path = PROCESSED_DIR / "feature_build_log.txt"
    log_path.write_text(log.getvalue(), encoding="utf-8")
    print(f"\nLog -> {log_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 feature builder")
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Stride for LSTM sliding windows (default=1). Use 5-10 to reduce memory."
    )
    args = parser.parse_args()
    main(stride=args.stride)
