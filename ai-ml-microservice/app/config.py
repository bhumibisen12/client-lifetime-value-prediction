"""
CLV Engine — Business configuration and product catalog.

All business-rule thresholds are readable from environment variables (prefix
CLV_) so they can be tuned without code changes or redeployment.

Product Catalog:
    This module defines the platform product names used for upsell
    and cross-sell recommendations, keyed by B2B industry vertical.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Upsell Product Map (industry → recommended upgrade tier)
# ---------------------------------------------------------------------------
UPSELL_PRODUCTS: dict[str, str] = {
    "Technology":               "Enterprise AI Suite — Unlimited Seats + Custom Model Training",
    "Finance":                  "FinOps Intelligence Tier — Compliance Module + Advanced Reporting",
    "Healthcare":               "Health Data Platform — HIPAA-Ready Analytics + Patient Journey Mapping",
    "Retail":                   "Retail 360 — Omnichannel CRM + Inventory Prediction Module",
    "Manufacturing":            "OpsEdge Pro — Supply Chain AI + Predictive Maintenance Alerts",
    "Logistics & Supply Chain": "LogiCore Enterprise — Real-Time Fleet Analytics + Route Intelligence",
    "Education":                "EduInsights Pro — Student Lifecycle Analytics + LMS Integration",
    "Real Estate":              "PropTech Intelligence — Lead Scoring + Market Trend Dashboard",
    "Hospitality":              "GuestPulse Pro — Loyalty CRM + Revenue Management AI",
    "E-commerce":               "Commerce Brain — Buyer Journey AI + Dynamic Pricing Engine",
}

# ---------------------------------------------------------------------------
# Cross-Sell Product Map (industry → cross-sell module slug + name)
# ---------------------------------------------------------------------------
CROSS_SELL_PRODUCTS: dict[str, dict[str, str]] = {
    "Technology": {
        "slug": "ai-workflow-automation",
        "name": "AI Workflow Automation Module",
    },
    "Finance": {
        "slug": "risk-intelligence-add-on",
        "name": "Risk Intelligence Add-On",
    },
    "Healthcare": {
        "slug": "clinical-reporting-suite",
        "name": "Clinical Reporting Suite",
    },
    "Retail": {
        "slug": "demand-forecasting-module",
        "name": "Demand Forecasting Module",
    },
    "Manufacturing": {
        "slug": "quality-control-ai",
        "name": "Quality Control AI Module",
    },
    "Logistics & Supply Chain": {
        "slug": "last-mile-optimizer",
        "name": "Last-Mile Delivery Optimizer",
    },
    "Education": {
        "slug": "engagement-tracker-edu",
        "name": "Student Engagement Tracker",
    },
    "Real Estate": {
        "slug": "valuation-intelligence",
        "name": "Property Valuation Intelligence",
    },
    "Hospitality": {
        "slug": "booking-analytics-pro",
        "name": "Booking Analytics Pro",
    },
    "E-commerce": {
        "slug": "cart-recovery-ai",
        "name": "Cart Recovery & Retention AI",
    },
}

# Fallback for unrecognized industry
UPSELL_DEFAULT = "Enterprise Platform — Full Suite Upgrade"
CROSS_SELL_DEFAULT = {
    "slug": "enterprise-integration",
    "name": "Enterprise Integration Module",
}


# ---------------------------------------------------------------------------
# Business rule thresholds — tunable via env vars
# ---------------------------------------------------------------------------
@dataclass
class CLVConfig:
    """
    Business-rule thresholds for the CLV engine.

    All fields read defaults from environment variables so values can be
    adjusted between deployments without touching code.
    """

    # --- Upsell thresholds ---
    upsell_engagement_threshold: float = float(
        os.getenv("CLV_UPSELL_ENGAGEMENT_THRESHOLD", "72.0")
    )
    upsell_tenure_months_threshold: int = int(
        os.getenv("CLV_UPSELL_TENURE_MONTHS_THRESHOLD", "10")
    )
    # Upsell also fires when upsell_score (ML output) exceeds this floor
    upsell_ml_score_threshold: float = float(
        os.getenv("CLV_UPSELL_ML_SCORE_THRESHOLD", "0.60")
    )

    # --- Cross-sell thresholds ---
    cross_sell_min_purchase_frequency: int = int(
        os.getenv("CLV_CROSS_SELL_MIN_PURCHASE_FREQUENCY", "3")
    )
    # Base offset in days; dynamically adjusted per client (see clv_predictor.py)
    cross_sell_base_offset_days: int = int(
        os.getenv("CLV_CROSS_SELL_BASE_OFFSET_DAYS", "45")
    )

    # --- Drift detection ---
    drift_engagement_threshold_pct: float = float(
        os.getenv("CLV_DRIFT_ENGAGEMENT_THRESHOLD_PCT", "15.0")
    )

    # --- Clamp fallback (when no per-industry ceiling is found) ---
    clamp_fallback_multiplier: float = float(
        os.getenv("CLV_CLAMP_FALLBACK_MULTIPLIER", "9.0")
    )

    # --- Database ---
    db_path: str = os.getenv(
        "CLV_DB_PATH",
        "database/clv_predictions.db"
    )


# Singleton — import this everywhere; do not instantiate CLVConfig again.
clv_config = CLVConfig()
