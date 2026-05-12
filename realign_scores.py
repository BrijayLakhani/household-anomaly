"""
realign_scores.py — Rebuild scores.parquet from saved models using left join.

Loads the already-trained IF and LSTM models, re-scores train+test data,
then aligns with a LEFT join on the IF index so LSTM-only timestamps
(before tabular features start) are dropped while sensor-gap rows
(NaN IF scores from missing readings) are preserved.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"

from src.models.iforest import load_iforest, score_iforest
from src.models.lstm_ae import TF_AVAILABLE, load_lstm_ae, score_lstm_ae

# ── Load tabular features ─────────────────────────────────────────────────────
print("Loading tabular features ...")
X_train_tab = pd.read_parquet(PROC / "features_tabular_train_scaled.parquet")
X_test_tab  = pd.read_parquet(PROC / "features_tabular_test_scaled.parquet")
all_tab = pd.concat([X_train_tab, X_test_tab])

# ── IF scoring ────────────────────────────────────────────────────────────────
print("Scoring with saved Isolation Forest ...")
model_if, threshold_if = load_iforest(MODELS / "iforest.pkl")
scores_if = score_iforest(model_if, all_tab, threshold_if)

meta = json.loads((PROC / "split_metadata.json").read_text())
cutoff = pd.Timestamp(meta["split_timestamp"])
train_flags = int(scores_if.loc[scores_if.index < cutoff, "if_flag"].fillna(0).sum())
print(f"  Train flags: {train_flags:,}  ({train_flags/len(X_train_tab):.2%})")
nan_if_train = int(scores_if.loc[scores_if.index < cutoff, "if_anomaly_score"].isna().sum())
print(f"  NaN IF scores in train: {nan_if_train}")

# ── LSTM scoring ──────────────────────────────────────────────────────────────
if TF_AVAILABLE:
    print("Scoring with saved LSTM-AE ...")
    X_train_seq = np.load(PROC / "sequences_train.npy")
    X_test_seq  = np.load(PROC / "sequences_test.npy")
    train_idx = pd.read_parquet(PROC / "sequences_train_index.parquet").index
    test_idx  = pd.read_parquet(PROC / "sequences_test_index.parquet").index

    model_lstm, threshold_lstm = load_lstm_ae(MODELS)
    scores_lstm_train = score_lstm_ae(model_lstm, X_train_seq, train_idx, threshold_lstm)
    scores_lstm_test  = score_lstm_ae(model_lstm, X_test_seq,  test_idx,  threshold_lstm)
    scores_lstm = pd.concat([scores_lstm_train, scores_lstm_test])
    print(f"  LSTM train flags: {int(scores_lstm_train['lstm_flag'].sum()):,}")
    print(f"  LSTM test  flags: {int(scores_lstm_test['lstm_flag'].sum()):,}")
else:
    print("[SKIP] TensorFlow not available.")
    scores_lstm = pd.DataFrame(
        {"lstm_recon_error": pd.array([], dtype=float),
         "lstm_flag":        pd.array([], dtype=float)},
    )

# ── Left join on IF index ─────────────────────────────────────────────────────
# Keeps every IF-scored minute (including sensor-gap NaNs in test period).
# Drops LSTM-only timestamps that predate the tabular feature window.
print("Joining scores (left join on IF index) ...")
combined = scores_if.join(scores_lstm, how="left")

if TF_AVAILABLE and len(scores_lstm) > 0:
    both_valid = combined["if_flag"].notna() & combined["lstm_flag"].notna()
    combined["agreement"] = np.where(
        both_valid,
        ((combined["if_flag"] == 1) & (combined["lstm_flag"] == 1)).astype(float),
        np.nan,
    )
    n_agree = int(combined["agreement"].fillna(0).sum())
    n_both  = int(both_valid.sum())
    print(f"  Jointly scored: {n_both:,}  |  Agreement (both=1): {n_agree:,}  ({n_agree/n_both:.2%})")
else:
    combined["agreement"] = np.nan

out = MODELS / "scores.parquet"
combined.to_parquet(out)
print(f"\nSaved: {out}  shape={combined.shape}")
print(f"Columns: {list(combined.columns)}")
nan_if_train_check = int(combined.loc[combined.index < cutoff, "if_anomaly_score"].isna().sum())
print(f"NaN IF scores in train period: {nan_if_train_check}  (should be 0)")
