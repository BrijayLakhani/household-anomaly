"""
select_anomalies.py — Diverse anomaly selection for XAI explanations.

Builds a 200-point set covering four distinct model behaviours so the
dashboard can show explanations across the full anomaly taxonomy:
  if_only   — high IF score; LSTM did not flag
  lstm_only — high LSTM reconstruction error; IF did not flag
  both      — both models flagged (strongest signal)
  normal    — confirmed normal points (contrast / control)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
EXPL_DIR = _ROOT / "data" / "explanations"

_OUTPUT_COLS: list[str] = [
    "timestamp",
    "if_anomaly_score",
    "lstm_recon_error",
    "if_flag",
    "lstm_flag",
    "agreement",
    "selection_group",
]


def select_top_anomalies(scores: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """
    Build a diverse selection of n points from scores.parquet for XAI.

    Parameters
    ----------
    scores:
        DataFrame with DatetimeIndex and columns from score_all.py
        (if_anomaly_score, if_flag, lstm_recon_error, lstm_flag, agreement).
    n:
        Total points.  Must be divisible by 4 (default 200 → 50 per group).

    Returns
    -------
    DataFrame with columns listed in _OUTPUT_COLS.  timestamp is a plain
    column (not the index) for easy joining in downstream parquet files.
    """
    per_group = n // 4  # 50

    # IF-only: IF flagged; LSTM did not flag (includes rows with no LSTM score)
    if_only_mask = (scores["if_flag"] == 1) & (scores["lstm_flag"] != 1)
    if_only = (
        scores.loc[if_only_mask]
        .nlargest(per_group, "if_anomaly_score")
        .copy()
    )
    if_only["selection_group"] = "if_only"

    # LSTM-only: LSTM flagged; IF scored the row as normal (flag == 0)
    lstm_only_mask = (scores["lstm_flag"] == 1) & (scores["if_flag"] == 0)
    lstm_only = (
        scores.loc[lstm_only_mask]
        .nlargest(per_group, "lstm_recon_error")
        .copy()
    )
    lstm_only["selection_group"] = "lstm_only"

    # Both: agreement column exactly 1
    both_mask = scores["agreement"] == 1
    both = (
        scores.loc[both_mask]
        .nlargest(per_group, "if_anomaly_score")
        .copy()
    )
    both["selection_group"] = "both"

    # Normal: neither model flagged (concrete 0, not NaN)
    normal_mask = (scores["if_flag"] == 0) & (scores["lstm_flag"] == 0)
    normal = (
        scores.loc[normal_mask]
        .sample(n=per_group, random_state=42)
        .copy()
    )
    normal["selection_group"] = "normal"

    combined = pd.concat([if_only, lstm_only, both, normal])
    combined.index.name = "timestamp"
    result = combined.reset_index()[_OUTPUT_COLS]
    return result


def save_selected_anomalies(selected: pd.DataFrame) -> Path:
    """Persist selected_anomalies.parquet and print a group summary."""
    EXPL_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPL_DIR / "selected_anomalies.parquet"
    selected.to_parquet(out, index=False)
    counts = selected["selection_group"].value_counts()
    print(f"Selected anomalies -> {out}  ({len(selected)} total)")
    for grp in ["if_only", "lstm_only", "both", "normal"]:
        print(f"  {grp:<12}: {counts.get(grp, 0)}")
    return out
