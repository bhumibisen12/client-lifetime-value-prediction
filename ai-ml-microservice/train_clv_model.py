"""
train_clv_model.py — Detagenix CLV XGBoost Training Pipeline v2

Trains the CLV prediction model and saves all required artefacts to
models_registry/. Must be run before starting the API.

Run:
    python train_clv_model.py
    python train_clv_model.py path/to/custom_data.csv

Architecture:
    sklearn Pipeline:
        1. ColumnTransformer
            - StandardScaler on numeric features
            - OneHotEncoder on industry_type (fixed category list)
        2. XGBRegressor with tuned hyperparameters

Artefacts saved:
    models_registry/clv_pipeline_v2.pkl       — Fitted pipeline (preprocessor + model)
    models_registry/training_metrics.json     — RMSE, MAE, R² on held-out test set
    models_registry/feature_ranges.json       — p01/p99 per numeric feature for confidence scoring
    models_registry/industry_clv_ceilings.json— Per-industry 95th-pct CLV/spend ratio ceilings
    models_registry/industry_clv_medians.json — Per-industry median CLV (for cross-sell gating)
    models_registry/upsell_model.pkl          — Secondary XGBClassifier: upsell probability
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
REGISTRY = _HERE / "models_registry"
REGISTRY.mkdir(exist_ok=True)

PIPELINE_PATH       = REGISTRY / "clv_pipeline_v2.pkl"
METRICS_PATH        = REGISTRY / "training_metrics.json"
FEATURE_RANGES_PATH = REGISTRY / "feature_ranges.json"
CEILINGS_PATH       = REGISTRY / "industry_clv_ceilings.json"
MEDIANS_PATH        = REGISTRY / "industry_clv_medians.json"
UPSELL_MODEL_PATH   = REGISTRY / "upsell_model.pkl"

DEFAULT_CSV = _HERE / "clv_synthetic_dataset.csv"
TARGET = "predicted_clv_amount"

# ---------------------------------------------------------------------------
# Feature column lists — must match pipelines/feature_store.py BASE_FEATURE_COLUMNS
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "historical_spend_amount",
    "purchase_frequency_yearly",
    "rolling_engagement_score",
    "contract_tenure_months",
    "spend_per_tenure_year",
    "spend_velocity",
    "engagement_x_tenure",
]
CATEGORICAL_FEATURES: list[str] = ["industry_type"]

INDUSTRY_CATEGORIES: list[str] = [
    "E-commerce",
    "Education",
    "Finance",
    "Healthcare",
    "Hospitality",
    "Logistics & Supply Chain",
    "Manufacturing",
    "Real Estate",
    "Retail",
    "Technology",
]

# ---------------------------------------------------------------------------
# Upsell label thresholds (used to build training labels for upsell classifier)
# A client is a "upsell candidate" if their predicted CLV/spend ratio puts them
# in the top 40% within their industry AND their engagement is above 65.
# ---------------------------------------------------------------------------
UPSELL_ENGAGEMENT_MIN = 65.0
UPSELL_CLV_RATIO_PERCENTILE = 0.60  # top 40% of CLV/spend ratio


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive engineered features on the full dataset DataFrame.

    Used only during training — the live feature_store.py derives these
    identically on per-request payloads.

    Args:
        df: Raw DataFrame from the CSV.

    Returns:
        DataFrame with derived feature columns added.
    """
    df = df.copy()
    safe_tenure = df["contract_tenure_months"].clip(lower=1)
    df["spend_per_tenure_year"] = df["historical_spend_amount"] / (safe_tenure / 12.0)
    df["spend_velocity"]        = df["historical_spend_amount"] / safe_tenure
    df["engagement_x_tenure"]   = df["rolling_engagement_score"] * df["contract_tenure_months"]
    return df


def build_clv_pipeline() -> Pipeline:
    """
    Construct the unfitted sklearn Pipeline for CLV regression.

    Architecture:
        ColumnTransformer → StandardScaler (numeric) + OHE (industry)
        XGBRegressor: n_estimators=300, max_depth=5, learning_rate=0.07

    Returns:
        Unfitted sklearn Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[INDUSTRY_CATEGORIES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def build_upsell_pipeline() -> Pipeline:
    """
    Construct the unfitted XGBClassifier pipeline for upsell probability scoring.

    Uses the same preprocessor architecture as the CLV regressor so that
    identical feature transformations are applied to both models.

    Returns:
        Unfitted sklearn Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[INDUSTRY_CATEGORIES],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.80,
        colsample_bytree=0.80,
        random_state=42,
        tree_method="hist",
        verbosity=0,
        eval_metric="logloss",
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def compute_feature_ranges(df: pd.DataFrame) -> dict:
    """
    Compute 1st and 99th percentile for each numeric feature from training split.

    These bounds are used by CLVPredictor to detect extrapolation (inputs
    outside the training distribution), which feeds the confidence score.

    Args:
        df: Training-split DataFrame (after feature derivation).

    Returns:
        Dict: column → {"p01": float, "p99": float}.
    """
    ranges: dict = {}
    for col in NUMERIC_FEATURES:
        ranges[col] = {
            "p01": float(np.percentile(df[col], 1)),
            "p99": float(np.percentile(df[col], 99)),
        }
    return ranges


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate the training CSV.

    Args:
        csv_path: Path to the CLV training CSV.

    Returns:
        Validated DataFrame.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = [
        "company_id",
        "historical_spend_amount",
        "purchase_frequency_yearly",
        "industry_type",
        "rolling_engagement_score",
        "contract_tenure_months",
        TARGET,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=[TARGET])
    if len(df) < before:
        print(f"[warn] Dropped {before - len(df)} rows with null target.")

    return df


def _build_upsell_labels(
    df: pd.DataFrame,
    clv_col: str = TARGET,
) -> np.ndarray:
    """
    Build binary upsell labels from training data.

    A client is a upsell candidate (label=1) if:
    1. Their engagement score >= UPSELL_ENGAGEMENT_MIN
    2. Their CLV/spend ratio is in the top 40% within their industry

    This produces a data-driven label rather than a hardcoded threshold,
    so the classifier learns what engagement + spend patterns actually
    correlate with high-value expansion.

    Args:
        df: Training DataFrame (must include industry_type and the target column).
        clv_col: Name of the CLV target column.

    Returns:
        np.ndarray of binary labels (0 or 1).
    """
    df = df.copy()
    # CLV/spend ratio per row
    safe_spend = df["historical_spend_amount"].clip(lower=1)
    df["_clv_ratio"] = df[clv_col] / safe_spend

    # Per-industry 60th percentile of ratio
    industry_p60 = df.groupby("industry_type")["_clv_ratio"].transform(
        lambda x: x.quantile(UPSELL_CLV_RATIO_PERCENTILE)
    )

    labels = (
        (df["rolling_engagement_score"] >= UPSELL_ENGAGEMENT_MIN)
        & (df["_clv_ratio"] >= industry_p60)
    ).astype(int)

    return labels.values


def train(csv_path: Path | None = None) -> None:
    """
    Full training run: load → derive features → split → fit → evaluate → save.

    Args:
        csv_path: Optional override for the CSV path.
    """
    path = csv_path or DEFAULT_CSV
    print(f"[train] Loading data from: {path}")
    df = load_data(path)
    df = derive_features(df)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]
    y_upsell = _build_upsell_labels(df)

    X_train, X_test, y_train, y_test, y_up_train, y_up_test = train_test_split(
        X, y, y_upsell, test_size=0.20, random_state=42
    )
    print(f"[train] Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"[train] Upsell positive rate (train): {y_up_train.mean():.2%}")

    # ─── Fit CLV Regressor ───────────────────────────────────────────────────
    print("[train] Fitting CLV XGBoost regressor ...")
    clv_pipeline = build_clv_pipeline()
    clv_pipeline.fit(X_train, y_train)

    y_pred = clv_pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    train_eng_avg = float(X_train["rolling_engagement_score"].mean())

    print("\n" + "=" * 50)
    print("  CLV Regressor — Test-Set Metrics")
    print("=" * 50)
    print(f"  RMSE : {rmse:>14,.2f}")
    print(f"  MAE  : {mae:>14,.2f}")
    print(f"  R²   : {r2:>14.4f}")
    print("=" * 50)

    # ─── Fit Upsell Classifier ───────────────────────────────────────────────
    print("[train] Fitting upsell XGBoost classifier ...")
    upsell_pipeline = build_upsell_pipeline()
    upsell_pipeline.fit(X_train, y_up_train)

    up_probs = upsell_pipeline.predict_proba(X_test)[:, 1]
    up_preds = (up_probs >= 0.50).astype(int)
    up_acc   = float(accuracy_score(y_up_test, up_preds))
    up_auc   = float(roc_auc_score(y_up_test, up_probs))

    print(f"\n  Upsell Classifier — Test-Set Metrics")
    print("=" * 50)
    print(f"  Accuracy : {up_acc:>12.4f}")
    print(f"  ROC-AUC  : {up_auc:>12.4f}")
    print("=" * 50 + "\n")

    # ─── Industry statistics (training split only — no leakage) ──────────────
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values

    # Medians
    industry_medians: dict[str, float] = (
        train_df.groupby("industry_type")[TARGET].median().to_dict()
    )
    overall_median = float(y_train.median())
    for ind in INDUSTRY_CATEGORIES:
        if ind not in industry_medians:
            industry_medians[ind] = overall_median
            print(f"[warn] No training rows for '{ind}'; using overall median.")

    # Per-industry 95th-percentile CLV/spend ratio ceilings
    safe_spend = train_df["historical_spend_amount"].clip(lower=1)
    ratio_series = train_df[TARGET] / safe_spend
    industry_ratio_p95 = (
        ratio_series.groupby(train_df["industry_type"]).quantile(0.95).to_dict()
    )
    for ind in INDUSTRY_CATEGORIES:
        if ind not in industry_ratio_p95:
            industry_ratio_p95[ind] = 9.0
            print(f"[warn] No ceiling data for '{ind}'; defaulting to 9x.")

    # ─── Feature ranges ──────────────────────────────────────────────────────
    feature_ranges = compute_feature_ranges(X_train)

    # ─── Save all artefacts ──────────────────────────────────────────────────
    joblib.dump(clv_pipeline,    PIPELINE_PATH)
    joblib.dump(upsell_pipeline, UPSELL_MODEL_PATH)
    print(f"[train] CLV pipeline saved    → {PIPELINE_PATH}")
    print(f"[train] Upsell model saved    → {UPSELL_MODEL_PATH}")

    metrics = {
        "r2": r2, "rmse": rmse, "mae": mae,
        "upsell_accuracy": up_acc, "upsell_roc_auc": up_auc,
        "training_engagement_avg": train_eng_avg,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Metrics saved         → {METRICS_PATH}")

    with open(FEATURE_RANGES_PATH, "w") as f:
        json.dump(feature_ranges, f, indent=2)
    print(f"[train] Feature ranges saved  → {FEATURE_RANGES_PATH}")

    ceilings_payload = {
        "ceilings": {k: float(v) for k, v in industry_ratio_p95.items()},
        "training_engagement_avg": train_eng_avg,
    }
    with open(CEILINGS_PATH, "w") as f:
        json.dump(ceilings_payload, f, indent=2)
    print(f"[train] Ceilings saved        → {CEILINGS_PATH}")
    print(f"\n  Per-industry CLV/spend ratio ceilings (95th pct):")
    for ind, ceil in sorted(ceilings_payload["ceilings"].items()):
        print(f"    {ind:<32} {ceil:.2f}x")

    medians_out = {k: float(v) for k, v in industry_medians.items()}
    with open(MEDIANS_PATH, "w") as f:
        json.dump(medians_out, f, indent=2)
    print(f"[train] Medians saved         → {MEDIANS_PATH}")
    print(f"\n[train] Training engagement avg: {train_eng_avg:.2f}")
    print("\n[train] ✓ All artefacts saved. Ready to run: uvicorn main:app --reload")


if __name__ == "__main__":
    csv_override = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    train(csv_path=csv_override)
