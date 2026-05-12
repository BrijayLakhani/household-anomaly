"""
test_xai.py

Correctness tests for Phase-4 XAI artefacts.
All tests skip gracefully if build_all_explanations.py has not been run yet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_EXPL = _ROOT / "data" / "explanations"
sys.path.insert(0, str(_ROOT))

SELECTED_PATH = _EXPL / "selected_anomalies.parquet"
SHAP_PATH = _EXPL / "shap_values.parquet"
SHAP_GLOBAL_PATH = _EXPL / "shap_global_importance.parquet"
SHAP_META_PATH = _EXPL / "shap_metadata.json"
LIME_PATH = _EXPL / "lime_explanations.parquet"
CF_PATH = _EXPL / "counterfactuals.parquet"

RAW_COLS: list[str] = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

_SELECTION_GROUPS = {"if_only", "lstm_only", "both", "normal"}


def _require(*paths: Path) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(
            "Required XAI artefacts not found — run "
            "'python src/xai/build_all_explanations.py':\n"
            + "\n".join(f"  {p}" for p in missing)
        )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def selected() -> pd.DataFrame:
    _require(SELECTED_PATH)
    return pd.read_parquet(SELECTED_PATH)


@pytest.fixture(scope="module")
def shap_df() -> pd.DataFrame:
    _require(SHAP_PATH)
    return pd.read_parquet(SHAP_PATH)


@pytest.fixture(scope="module")
def shap_meta() -> dict:
    _require(SHAP_META_PATH)
    return json.loads(SHAP_META_PATH.read_text())


@pytest.fixture(scope="module")
def lime_df() -> pd.DataFrame:
    _require(LIME_PATH)
    return pd.read_parquet(LIME_PATH)


@pytest.fixture(scope="module")
def cf_df() -> pd.DataFrame:
    _require(CF_PATH)
    return pd.read_parquet(CF_PATH)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_selected_anomalies_diverse(selected: pd.DataFrame) -> None:
    """
    All four selection groups must be present.  Missing a group means the
    selection logic filtered too aggressively or scores.parquet is incomplete.
    """
    groups = set(selected["selection_group"].unique())
    missing = _SELECTION_GROUPS - groups
    assert not missing, (
        f"Missing selection groups: {missing}.  "
        f"Present: {groups}"
    )
    # Each group should have at least 1 row
    counts = selected["selection_group"].value_counts()
    for grp in _SELECTION_GROUPS:
        assert counts.get(grp, 0) >= 1, f"Group '{grp}' has no rows."


def test_shap_values_shape(
    selected: pd.DataFrame,
    shap_df: pd.DataFrame,
) -> None:
    """
    SHAP output must have exactly n_selected × 24 rows.
    Any fewer means some timestamps were missing from the feature matrix.
    """
    n_selected = len(selected)
    n_features = 24
    expected_rows = n_selected * n_features
    assert len(shap_df) == expected_rows, (
        f"Expected {expected_rows} SHAP rows ({n_selected} timestamps × {n_features} features), "
        f"got {len(shap_df)}.  Check that all selected timestamps exist in the feature matrix."
    )
    assert set(shap_df.columns) >= {"timestamp", "feature_name", "shap_value", "feature_value"}, (
        f"Missing columns in shap_values.parquet.  Got: {list(shap_df.columns)}"
    )


def test_shap_sums_match_score(
    selected: pd.DataFrame,
    shap_df: pd.DataFrame,
    shap_meta: dict,
) -> None:
    """
    TreeSHAP additivity: expected_value + sum(SHAP_i(x)) == shap_model_output(x)
    within 1e-3.

    Note: for IsolationForest, SHAP's TreeExplainer explains raw path lengths,
    not the normalised if_anomaly_score.  expected_value ≈ 12 (average path
    length), not 0.5.  shap_model_output is stored alongside the SHAP values
    so this test remains correct regardless of which output SHAP uses internally.
    """
    assert "shap_model_output" in shap_df.columns, (
        "shap_values.parquet is missing 'shap_model_output' column.  "
        "Re-run build_all_explanations.py."
    )

    expected_value = shap_meta["expected_value"]
    # Sample 10 timestamps for speed
    sample_ts = shap_df["timestamp"].drop_duplicates().head(10)
    tol = 1e-3

    for ts in sample_ts:
        grp = shap_df[shap_df["timestamp"] == ts]
        shap_sum = grp["shap_value"].sum()
        stored_output = grp["shap_model_output"].iloc[0]
        computed = expected_value + shap_sum
        diff = abs(computed - stored_output)
        assert diff < tol, (
            f"SHAP additivity violated at {ts}: "
            f"expected_value({expected_value:.4f}) + shap_sum({shap_sum:.4f}) "
            f"= {computed:.4f}  vs  stored shap_model_output={stored_output:.4f}  "
            f"(diff={diff:.2e}, tol={tol})"
        )


def test_lime_returns_top5_features(
    selected: pd.DataFrame,
    lime_df: pd.DataFrame,
) -> None:
    """
    Every selected timestamp must have exactly 5 LIME feature rows.
    Fewer means LIME returned fewer than num_features=5; more means
    the saving logic duplicated rows.
    """
    counts = lime_df.groupby("timestamp")["feature_name"].count()
    wrong = counts[counts != 5]
    assert len(wrong) == 0, (
        f"{len(wrong)} timestamps do not have exactly 5 LIME features.\n"
        f"Examples:\n{wrong.head(5).to_string()}"
    )


def test_counterfactual_changes_features(cf_df: pd.DataFrame) -> None:
    """
    At least one feature must change in >= 80% of generated counterfactuals.
    If this fails, the DiCE model wrapper always predicts the same class and
    CFs are trivial (no-ops).
    """
    if len(cf_df) == 0:
        pytest.skip("No counterfactuals were generated (all failures).")

    cf_groups = cf_df.groupby(["timestamp", "cf_index"])
    n_cfs = len(cf_groups)
    n_with_change = sum(
        1 for _, grp in cf_groups if (grp["abs_change"] > 1e-6).any()
    )
    pct = n_with_change / n_cfs
    assert pct >= 0.80, (
        f"Only {n_with_change}/{n_cfs} ({pct:.0%}) counterfactuals changed at least "
        "one feature.  Expected >= 80%.  Check the DiCE wrapper's predict() method."
    )


def test_no_xai_for_engineered_features_in_cf(cf_df: pd.DataFrame) -> None:
    """
    Counterfactuals must only modify the 7 raw sensor features.
    Engineered features (lags, rolling, time, ratios) are derived and cannot
    be physically changed independently of the raw values.
    """
    if len(cf_df) == 0:
        pytest.skip("No counterfactuals were generated (all failures).")

    feat_in_cf = set(cf_df["feature_name"].unique())
    allowed = set(RAW_COLS)
    disallowed = feat_in_cf - allowed
    assert not disallowed, (
        f"Counterfactuals modified engineered features: {disallowed}.\n"
        "Only RAW_COLS should appear in feature_name."
    )
