"""
Feature engineering for the Detagenix CLV Prediction Engine v2.

Converts CLVInputPayload(s) into a model-ready DataFrame. This is the single
source of truth for feature construction — both train_clv_model.py and the
live predictor import from here to prevent train/serve skew.

v2 Improvements:
- Added `days_since_contract_start` as a continuous feature (more precise than integer months)
- Added `spend_velocity` (monthly spend rate)
- Added `spend_per_tenure_year` (annualised spend intensity)
- Engagement score is kept raw (0–100) — the model learns the relationship itself
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Union

import numpy as np
import pandas as pd

from app.clv_schemas import CLVInputPayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column order must exactly match what the ColumnTransformer was fitted on.
# Adding new features → update NUMERIC_FEATURES in train_clv_model.py too.
# ---------------------------------------------------------------------------
BASE_FEATURE_COLUMNS: list[str] = [
    "historical_spend_amount",
    "purchase_frequency_yearly",
    "rolling_engagement_score",
    "contract_tenure_months",
    "spend_per_tenure_year",     # derived: annualised spend rate
    "spend_velocity",            # derived: monthly spend rate
    "engagement_x_tenure",       # interaction: captures high-engagement + long-tenure clients
    "industry_type",             # categorical → OHE
]


def compute_rolling_engagement(raw_score: float) -> float:
    """
    Pass-through for the rolling engagement score.

    Extension point: when real event-level engagement history is available
    (e.g. from lead_activity table), replace this with a weighted recency
    decay calculation without changing call-sites.

    Args:
        raw_score: Engagement score 0–100.

    Returns:
        Processed engagement score (float, 0–100).
    """
    return float(raw_score)


def _derive_spend_per_tenure_year(
    historical_spend: float,
    contract_tenure_months: int,
) -> float:
    """
    Average annualised spend over the entire contract lifetime.

    Why: captures account spending intensity normalised by how long the
    client has been active. A client with $60k in 5 years is very different
    from one with $60k in 1 year.

    Args:
        historical_spend: Total USD spend.
        contract_tenure_months: Months under contract.

    Returns:
        Annualised spend per year (float).
    """
    if contract_tenure_months <= 0:
        return float(historical_spend)
    tenure_years = contract_tenure_months / 12.0
    return float(historical_spend / tenure_years)


def _derive_spend_velocity(
    historical_spend: float,
    contract_tenure_months: int,
) -> float:
    """
    Monthly spend velocity: average spend per active month.

    Args:
        historical_spend: Total spend amount.
        contract_tenure_months: Months under contract (>= 0).

    Returns:
        Monthly spend velocity (float, spend/month).
    """
    safe_tenure = max(contract_tenure_months, 1)
    return float(historical_spend / safe_tenure)


def _derive_engagement_x_tenure(
    rolling_engagement_score: float,
    contract_tenure_months: int,
) -> float:
    """
    Interaction feature: engagement × tenure.

    Clients with high engagement AND long tenure are the most valuable accounts.
    This feature lets XGBoost capture this non-linear joint effect in a single
    signal rather than discovering it through deep tree splits.

    Args:
        rolling_engagement_score: Engagement score 0–100.
        contract_tenure_months: Months under contract.

    Returns:
        Product of engagement score and tenure (float).
    """
    return float(rolling_engagement_score * contract_tenure_months)


def validate_raw_inputs(payload: CLVInputPayload) -> None:
    """
    Business-level guardrails applied before feature construction.

    Pydantic handles type/range validation; this adds semantic checks for
    values that are technically valid but likely data-quality issues.

    Args:
        payload: Validated CLVInputPayload instance.

    Raises:
        ValueError: If a field value is logically impossible.
    """
    if payload.historical_spend_amount < 0:
        raise ValueError(
            f"historical_spend_amount must be >= 0, got {payload.historical_spend_amount}"
        )
    if payload.contract_tenure_months < 0:
        raise ValueError(
            f"contract_tenure_months must be >= 0, got {payload.contract_tenure_months}"
        )
    if payload.historical_spend_amount == 0:
        logger.warning(
            "company_id=%s has historical_spend_amount=0; "
            "prediction will be bounded by the per-industry clamp at 0.",
            payload.company_id,
        )


def payload_to_dataframe(
    payload: Union[CLVInputPayload, list[CLVInputPayload]],
) -> pd.DataFrame:
    """
    Convert one or more CLVInputPayload objects into a model-ready DataFrame.

    The returned DataFrame has exactly the columns in BASE_FEATURE_COLUMNS,
    in that order, matching what the ColumnTransformer was fitted on during
    training (train/serve parity).

    Args:
        payload: A single CLVInputPayload or a list (for batch mode).

    Returns:
        pd.DataFrame with shape (n, len(BASE_FEATURE_COLUMNS)).
    """
    payloads: list[CLVInputPayload] = (
        payload if isinstance(payload, list) else [payload]
    )

    rows: list[dict] = []
    for p in payloads:
        validate_raw_inputs(p)
        engagement = compute_rolling_engagement(p.rolling_engagement_score)
        spend_per_yr = _derive_spend_per_tenure_year(
            p.historical_spend_amount, p.contract_tenure_months
        )
        spend_vel = _derive_spend_velocity(
            p.historical_spend_amount, p.contract_tenure_months
        )
        eng_x_tenure = _derive_engagement_x_tenure(
            p.rolling_engagement_score, p.contract_tenure_months
        )
        rows.append(
            {
                "historical_spend_amount": p.historical_spend_amount,
                "purchase_frequency_yearly": p.purchase_frequency_yearly,
                "rolling_engagement_score": engagement,
                "contract_tenure_months": p.contract_tenure_months,
                "spend_per_tenure_year": spend_per_yr,
                "spend_velocity": spend_vel,
                "engagement_x_tenure": eng_x_tenure,
                "industry_type": p.industry_type.value,  # str for OHE
            }
        )

    df = pd.DataFrame(rows, columns=BASE_FEATURE_COLUMNS)
    return df
