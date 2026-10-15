"""
Pydantic v2 data contracts for the CLV Prediction Engine v2.

Field names align with the documentation spec. All enums and validation
constraints are defined here — this is the single source of truth for
input/output schema.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class IndustryType(str, Enum):
    """B2B industry verticals supported by the CLV engine."""

    LOGISTICS = "Logistics & Supply Chain"
    RETAIL = "Retail"
    HEALTHCARE = "Healthcare"
    MANUFACTURING = "Manufacturing"
    FINANCE = "Finance"
    TECHNOLOGY = "Technology"
    EDUCATION = "Education"
    REAL_ESTATE = "Real Estate"
    HOSPITALITY = "Hospitality"
    ECOMMERCE = "E-commerce"


class CLVInputPayload(BaseModel):
    """Input contract for a single CLV prediction request."""

    company_id: str = Field(..., description="Client identifier string (e.g. 'client-00123' or a UUID)")
    historical_spend_amount: Annotated[
        float, Field(ge=0.0, description="Total historical spend in INR")
    ]
    purchase_frequency_yearly: Annotated[
        int, Field(ge=0, description="Number of purchases per year")
    ]
    industry_type: IndustryType = Field(..., description="Client's B2B industry vertical")
    rolling_engagement_score: Annotated[
        float,
        Field(ge=0.0, le=100.0, description="Rolling engagement score 0–100"),
    ]
    contract_tenure_months: Annotated[
        int, Field(ge=0, description="Months the client has been active")
    ]
    contract_start_date: str | None = Field(
        default=None,
        description="ISO-8601 date string YYYY-MM-DD. Used for dynamic pitch date calculation.",
    )

    @field_validator("contract_start_date")
    @classmethod
    def validate_contract_start_date(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        try:
            parsed = date.fromisoformat(v)
        except ValueError:
            raise ValueError(
                f"Invalid date format '{v}'. Must be a valid ISO-8601 calendar date in YYYY-MM-DD format (e.g., '2024-05-15')."
            )
        if parsed > date.today():
            raise ValueError(
                f"contract_start_date cannot be in the future ({v} > {date.today()}). Contract must have already started."
            )
        return parsed.isoformat()


class SHAPDriverEntry(BaseModel):
    """A single SHAP feature importance entry."""

    feature: str = Field(..., description="Feature name (human-readable label)")
    value: float | str = Field(..., description="Actual input value for this feature")
    shap_contribution: float = Field(..., description="SHAP value (signed impact on CLV)")
    direction: str = Field(..., description="'positive' or 'negative'")
    contribution_pct: float = Field(
        ..., description="This feature's share of total absolute SHAP contribution (%)"
    )


class ExpansionAnalytics(BaseModel):
    """Upsell signal, product recommendation, and top SHAP drivers."""

    upsell_opportunity_detected: bool
    upsell_confidence_score: Annotated[
        float, Field(ge=0.0, le=1.0, description="ML-derived upsell probability 0–1")
    ]
    recommended_target_product: str = Field(
        ..., description="Product or tier recommended for upsell"
    )
    top_prediction_driver_reason: str = Field(
        ..., description="Plain-English summary of top SHAP driver"
    )
    shap_drivers: list[SHAPDriverEntry] = Field(
        default_factory=list,
        description="Top 3 SHAP feature contributors (positive and negative)",
    )


class CrossSellOpportunityWindow(BaseModel):
    """Dynamic cross-sell timing and recommendation."""

    cross_sell_detected: bool
    secondary_product_slug: str = Field(..., description="Product slug for cross-sell")
    secondary_product_name: str = Field(..., description="Human-readable product name")
    optimal_pitch_target_date: str = Field(
        ..., description="ISO-8601 date string YYYY-MM-DD (client-specific, not fixed offset)"
    )
    days_until_optimal_pitch: int = Field(
        ..., description="Days from today to the recommended pitch date"
    )
    timing_confidence_score: Annotated[
        float, Field(ge=0.0, le=1.0, description="Timing confidence 0–1")
    ]
    pitch_rationale: str = Field(
        ..., description="Why this date was chosen for this specific client"
    )


class CLVOutputPayload(BaseModel):
    """Output contract for a single CLV prediction response."""

    company_id: str
    predicted_clv_amount: Annotated[
        float, Field(ge=0.0, description="Predicted lifetime value in INR")
    ]
    confidence_interval_percentage: Annotated[
        float,
        Field(ge=0.0, le=100.0, description="Approximate % confidence in prediction"),
    ]
    expansion_analytics: ExpansionAnalytics
    cross_sell_opportunity_window: CrossSellOpportunityWindow
    prediction_timestamp: str = Field(
        ..., description="ISO-8601 datetime of when prediction was generated"
    )
