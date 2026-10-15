"""
clv_predictor.py — CLV Prediction Engine Core v2

Loads trained artefacts and runs the full inference pipeline:
    1. Feature construction (via feature_store.py)
    2. CLV regression (XGBoost)
    3. Per-industry ceiling clamp
    4. SHAP explainability (top 3 drivers with signed contributions)
    5. ML-based upsell scoring (XGBClassifier probability)
    6. Dynamic cross-sell date calculation (client-specific, not a fixed offset)
    7. Confidence interval estimation

v2 Key Changes vs v1:
    - Upsell decision is ML-driven (XGBClassifier probability) not rule-only
    - Cross-sell pitch date is dynamic: based on contract cycle, tenure, engagement
    - SHAP returns top 3 drivers with % attribution (not just #1)
    - All product names come from the platform catalog (config.py)
    - confidence_interval_percentage uses graduated penalty (same as v1 but extended)
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from app.clv_schemas import (
    CLVInputPayload,
    CLVOutputPayload,
    CrossSellOpportunityWindow,
    ExpansionAnalytics,
    SHAPDriverEntry,
)
from app.config import (
    CROSS_SELL_DEFAULT,
    CROSS_SELL_PRODUCTS,
    UPSELL_DEFAULT,
    UPSELL_PRODUCTS,
    clv_config,
)
from pipelines.feature_store import BASE_FEATURE_COLUMNS, payload_to_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — relative to this file's parent (ai-ml-microservice/)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent.parent
REGISTRY        = _HERE / "models_registry"
PIPELINE_PATH   = REGISTRY / "clv_pipeline_v2.pkl"
UPSELL_PATH     = REGISTRY / "upsell_model.pkl"
CEILINGS_PATH   = REGISTRY / "industry_clv_ceilings.json"
MEDIANS_PATH    = REGISTRY / "industry_clv_medians.json"
FEATURE_RANGES  = REGISTRY / "feature_ranges.json"
METRICS_PATH    = REGISTRY / "training_metrics.json"

# Human-readable labels for SHAP output (keyed by DataFrame column name)
FEATURE_LABELS: dict[str, str] = {
    "historical_spend_amount":   "Historical Spend",
    "purchase_frequency_yearly": "Purchase Frequency (yearly)",
    "rolling_engagement_score":  "Engagement Score",
    "contract_tenure_months":    "Contract Tenure (months)",
    "spend_per_tenure_year":     "Annualised Spend per Year",
    "spend_velocity":            "Monthly Spend Velocity",
    "engagement_x_tenure":       "Engagement × Tenure Interaction",
}


class CLVPredictor:
    """
    Production inference class for the CLV Prediction Engine.

    Usage:
        predictor = CLVPredictor()           # loads artefacts on first call
        result    = predictor.predict(payload)   # CLVOutputPayload
    """

    def __init__(self) -> None:
        self._pipeline          = joblib.load(PIPELINE_PATH)
        self._upsell_pipeline   = joblib.load(UPSELL_PATH)

        with open(CEILINGS_PATH)   as f: ceilings_data        = json.load(f)
        with open(MEDIANS_PATH)    as f: self._industry_medians = json.load(f)
        with open(FEATURE_RANGES)  as f: self._feature_ranges   = json.load(f)
        with open(METRICS_PATH)    as f: metrics               = json.load(f)

        self._industry_ceilings      = ceilings_data["ceilings"]
        self.training_engagement_avg = ceilings_data.get("training_engagement_avg", 65.0)
        self._train_r2               = metrics.get("r2", 0.90)

        # Build a SHAP TreeExplainer once (expensive, cache it)
        xgb_model = self._pipeline.named_steps["regressor"]
        self._explainer = shap.TreeExplainer(xgb_model)

        logger.info("CLVPredictor v2 loaded. Training R²=%.4f", self._train_r2)

    def predict(self, payload: CLVInputPayload) -> CLVOutputPayload:
        """
        Run the full CLV inference pipeline for a single account.

        Steps:
            1. Build feature DataFrame from payload
            2. Preprocess → XGBoost regressor → raw CLV prediction
            3. Apply per-industry ceiling clamp
            4. SHAP: compute top-3 feature contributions
            5. Upsell: ML classifier probability + rule gate
            6. Cross-sell: dynamic pitch date calculation
            7. Confidence: graduated penalty based on feature ranges

        Args:
            payload: Validated CLVInputPayload instance.

        Returns:
            CLVOutputPayload with all prediction fields populated.
        """
        feature_df = payload_to_dataframe(payload)

        # ── Step 2: Preprocess + predict raw CLV ─────────────────────────────
        preprocessed  = self._pipeline.named_steps["preprocessor"].transform(feature_df)
        raw_pred       = float(self._pipeline.named_steps["regressor"].predict(preprocessed)[0])
        raw_pred       = max(raw_pred, 0.0)

        # ── Step 3: Per-industry ceiling clamp ───────────────────────────────
        clamped_pred = self._apply_clamp(payload, raw_pred)

        # ── Step 4: SHAP explainability ──────────────────────────────────────
        shap_values   = self._explainer.shap_values(preprocessed)
        shap_drivers  = self._build_shap_drivers(shap_values, feature_df, payload)
        top_reason    = self._build_top_reason(shap_drivers, payload)

        # ── Step 5: Confidence interval ──────────────────────────────────────
        confidence = self._compute_confidence(feature_df)

        # ── Step 6: ML Upsell scoring ────────────────────────────────────────
        expansion = self._evaluate_upsell(payload, feature_df, top_reason, shap_drivers)

        # ── Step 7: Dynamic cross-sell timing ────────────────────────────────
        cross_sell = self._evaluate_cross_sell(payload, clamped_pred)

        return CLVOutputPayload(
            company_id=payload.company_id,
            predicted_clv_amount=round(clamped_pred, 2),
            confidence_interval_percentage=round(confidence, 2),
            expansion_analytics=ExpansionAnalytics(**expansion),
            cross_sell_opportunity_window=CrossSellOpportunityWindow(**cross_sell),
            prediction_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _apply_clamp(self, payload: CLVInputPayload, raw_pred: float) -> float:
        """Apply per-industry 95th-percentile ceiling clamp."""
        industry = payload.industry_type.value
        ceiling_multiplier = self._industry_ceilings.get(
            industry, clv_config.clamp_fallback_multiplier
        )
        ceiling = payload.historical_spend_amount * ceiling_multiplier
        if raw_pred > ceiling:
            logger.debug(
                "company_id=%s: clamped %.2f → %.2f (industry=%s, %.2fx ceiling)",
                payload.company_id, raw_pred, ceiling, industry, ceiling_multiplier,
            )
            return ceiling
        return raw_pred

    def _build_shap_drivers(
        self,
        shap_values: np.ndarray,
        feature_df: pd.DataFrame,
        payload: CLVInputPayload,
    ) -> list[SHAPDriverEntry]:
        """
        Compute top-3 SHAP feature drivers with signed contributions.

        For OHE features (industry_type expands to multiple columns in the
        preprocessed space), we aggregate their SHAP values back to a single
        "industry_type" entry.

        Returns:
            List of up to 3 SHAPDriverEntry objects, sorted by |shap_value| desc.
        """
        shap_row = shap_values[0]  # shape: (n_transformed_features,)
        n_numeric = len([c for c in BASE_FEATURE_COLUMNS if c != "industry_type"])

        # Numeric features: direct 1:1 mapping
        numeric_cols = [c for c in BASE_FEATURE_COLUMNS if c != "industry_type"]
        contributions: list[tuple[str, float]] = []
        for i, col in enumerate(numeric_cols):
            if i < len(shap_row):
                contributions.append((col, float(shap_row[i])))

        # OHE industry columns: sum their SHAP values into one entry
        ohe_shap_sum = float(np.sum(shap_row[n_numeric:]))
        contributions.append(("industry_type", ohe_shap_sum))

        # Sort by absolute value descending, take top 3
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        top3 = contributions[:3]

        total_abs = sum(abs(v) for _, v in contributions) or 1.0

        drivers: list[SHAPDriverEntry] = []
        for col_key, shap_val in top3:
            direction = "positive" if shap_val >= 0 else "negative"
            pct = round(abs(shap_val) / total_abs * 100, 1)

            if col_key == "industry_type":
                val_display: Any = payload.industry_type.value
            else:
                val_display = round(float(feature_df[col_key].iloc[0]), 2)

            label = FEATURE_LABELS.get(col_key, col_key.replace("_", " ").title())
            drivers.append(
                SHAPDriverEntry(
                    feature=label,
                    value=val_display,
                    shap_contribution=round(shap_val, 2),
                    direction=direction,
                    contribution_pct=pct,
                )
            )

        return drivers

    def _build_top_reason(
        self,
        drivers: list[SHAPDriverEntry],
        payload: CLVInputPayload,
    ) -> str:
        """
        Build a plain-English explanation from the top SHAP driver.

        Args:
            drivers: List of SHAPDriverEntry objects (sorted by impact).
            payload: Original input payload.

        Returns:
            Human-readable explanation string.
        """
        if not drivers:
            return "Model prediction based on all provided client metrics."

        top = drivers[0]
        direction_word = "positively" if top.direction == "positive" else "negatively"

        if top.feature == FEATURE_LABELS.get("rolling_engagement_score"):
            return (
                f"Engagement score of {payload.rolling_engagement_score:.0f}/100 is {direction_word} "
                f"driving the CLV prediction, accounting for {top.contribution_pct:.0f}% "
                f"of the model's signal — {'high engagement signals strong retention' if top.direction == 'positive' else 'low engagement signals churn risk'}."
            )
        elif top.feature == FEATURE_LABELS.get("historical_spend_amount"):
            return (
                f"Historical spend of ₹{payload.historical_spend_amount:,.0f} is {direction_word} "
                f"influencing the CLV forecast ({top.contribution_pct:.0f}% of signal), "
                f"reflecting the client's demonstrated spending capacity."
            )
        elif top.feature == FEATURE_LABELS.get("contract_tenure_months"):
            return (
                f"Contract tenure of {payload.contract_tenure_months} months is {direction_word} "
                f"affecting the CLV prediction ({top.contribution_pct:.0f}% of signal) — "
                f"{'long-tenured clients show compounding value growth' if top.direction == 'positive' else 'short tenure limits projected lifetime value'}."
            )
        elif top.feature == FEATURE_LABELS.get("engagement_x_tenure"):
            return (
                f"The combined effect of high engagement ({payload.rolling_engagement_score:.0f}/100) "
                f"over {payload.contract_tenure_months} months of active contract is the strongest "
                f"signal, contributing {top.contribution_pct:.0f}% to the CLV forecast."
            )
        elif top.feature == FEATURE_LABELS.get("spend_velocity"):
            velocity = payload.historical_spend_amount / max(payload.contract_tenure_months, 1)
            return (
                f"Monthly spend velocity of ₹{velocity:,.0f}/month is {direction_word} "
                f"influencing the CLV score ({top.contribution_pct:.0f}% of signal)."
            )
        else:
            return (
                f"'{top.feature}' (value: {top.value}) is the strongest {direction_word} "
                f"driver, contributing {top.contribution_pct:.0f}% of the model's prediction signal."
            )

    def _compute_confidence(self, feature_df: pd.DataFrame) -> float:
        """
        Estimate confidence percentage around the CLV prediction.

        Formula:
            base = train_r2 * 100
            penalty per out-of-range feature = min(overshoot_fraction * 20, 15) pp
            total penalty capped at 30 pp
            confidence = max(5, base - total_penalty)

        Args:
            feature_df: Single-row DataFrame from payload_to_dataframe().

        Returns:
            Confidence interval percentage (float, 5–100).
        """
        numeric_cols = [c for c in BASE_FEATURE_COLUMNS if c != "industry_type"]
        base_confidence = self._train_r2 * 100.0
        total_penalty = 0.0

        for col in numeric_cols:
            if col not in self._feature_ranges:
                continue
            val = float(feature_df[col].iloc[0])
            p01 = self._feature_ranges[col]["p01"]
            p99 = self._feature_ranges[col]["p99"]
            range_width = max(p99 - p01, 1e-6)

            if val < p01:
                overshoot = (p01 - val) / range_width
                total_penalty += min(overshoot * 20.0, 15.0)
            elif val > p99:
                overshoot = (val - p99) / range_width
                total_penalty += min(overshoot * 20.0, 15.0)

        confidence = max(5.0, base_confidence - min(total_penalty, 30.0))
        return float(confidence)

    def _evaluate_upsell(
        self,
        payload: CLVInputPayload,
        feature_df: pd.DataFrame,
        top_driver_reason: str,
        shap_drivers: list[SHAPDriverEntry],
    ) -> dict:
        """
        ML-driven upsell evaluation using the trained XGBClassifier.

        Decision rule (v2):
            upsell_score = XGBClassifier.predict_proba(features)[:, 1]
            upsell_detected = (
                upsell_score >= clv_config.upsell_ml_score_threshold
                AND rolling_engagement_score >= clv_config.upsell_engagement_threshold
                AND contract_tenure_months >= clv_config.upsell_tenure_months_threshold
            )

        The ML score acts as the primary signal; the engagement + tenure gates
        act as sanity filters to prevent low-engagement clients from being
        incorrectly flagged by the model.

        Returns:
            Dict matching the ExpansionAnalytics schema.
        """
        # ML upsell probability
        preprocessed = self._upsell_pipeline.named_steps["preprocessor"].transform(feature_df)
        upsell_prob  = float(
            self._upsell_pipeline.named_steps["classifier"].predict_proba(preprocessed)[0, 1]
        )
        upsell_prob  = round(upsell_prob, 4)

        # Combined rule: ML score + minimum engagement + minimum tenure
        upsell = (
            upsell_prob >= clv_config.upsell_ml_score_threshold
            and payload.rolling_engagement_score >= clv_config.upsell_engagement_threshold
            and payload.contract_tenure_months >= clv_config.upsell_tenure_months_threshold
        )

        industry = payload.industry_type.value
        product = UPSELL_PRODUCTS.get(industry, UPSELL_DEFAULT)

        return {
            "upsell_opportunity_detected": upsell,
            "upsell_confidence_score": upsell_prob,
            "recommended_target_product": product,
            "top_prediction_driver_reason": top_driver_reason,
            "shap_drivers": [d.model_dump() for d in shap_drivers],
        }

    def _evaluate_cross_sell(
        self,
        payload: CLVInputPayload,
        clamped_pred: float,
    ) -> dict:
        """
        Dynamic cross-sell pitch date calculation — client-specific, not a fixed offset.

        Logic:
            1. Base offset = clv_config.cross_sell_base_offset_days (default 45)
            2. Engagement adjustment: high engagement → pitch sooner (up to -15 days)
               low engagement → pitch later (up to +20 days)
            3. Frequency adjustment: high purchase frequency → pitch sooner (up to -10 days)
            4. Tenure adjustment: new clients (<6 months) → add 15 days (not ready yet)
            5. Contract renewal proximity: if contract started ~11-12 months ago → subtract 10 days
               (pitch before annual renewal decision)

        This means each client gets a genuinely different pitch date based on
        their behavioral profile.

        Timing confidence: multi-factor score (not just engagement/100):
            - engagement component (0.5 weight)
            - frequency component (0.3 weight)
            - above-median CLV component (0.2 weight)

        Args:
            payload: Original input payload.
            clamped_pred: Clamped CLV prediction value.

        Returns:
            Dict matching the CrossSellOpportunityWindow schema.
        """
        industry  = payload.industry_type.value
        today     = date.today()
        base_offset = clv_config.cross_sell_base_offset_days

        # ── 1. Engagement adjustment ────────────────────────────────────────
        eng = payload.rolling_engagement_score
        if eng >= 80:
            eng_adj = -15
        elif eng >= 65:
            eng_adj = -8
        elif eng >= 50:
            eng_adj = 0
        elif eng >= 35:
            eng_adj = +10
        else:
            eng_adj = +20

        # ── 2. Purchase frequency adjustment ───────────────────────────────
        freq = payload.purchase_frequency_yearly
        if freq >= 10:
            freq_adj = -10
        elif freq >= 6:
            freq_adj = -5
        elif freq >= 3:
            freq_adj = 0
        else:
            freq_adj = +8

        # ── 3. Tenure adjustment (new clients need more time) ───────────────
        tenure = payload.contract_tenure_months
        if tenure < 6:
            tenure_adj = +15
        elif tenure < 12:
            tenure_adj = +5
        else:
            tenure_adj = 0

        # ── 4. Renewal proximity adjustment ────────────────────────────────
        # If contract start date is provided, check if annual renewal is coming
        renewal_adj = 0
        pitch_rationale_detail = ""
        if payload.contract_start_date:
            try:
                start = date.fromisoformat(payload.contract_start_date)
                months_active = (today - start).days / 30.44
                # Months into current annual cycle
                months_in_cycle = months_active % 12
                if 10 <= months_in_cycle <= 11.5:
                    # Renewal in ~1-2 months — pitch now before decision is made
                    renewal_adj = -10
                    pitch_rationale_detail = " Contract renewal approaching — pitch before annual decision."
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "company_id=%s: Invalid contract_start_date '%s': %s",
                    payload.company_id,
                    payload.contract_start_date,
                    exc,
                )

        # ── Total offset ────────────────────────────────────────────────────
        total_offset = max(7, base_offset + eng_adj + freq_adj + tenure_adj + renewal_adj)
        pitch_date   = today + timedelta(days=total_offset)
        pitch_date_str = pitch_date.strftime("%Y-%m-%d")

        # ── Cross-sell gate ─────────────────────────────────────────────────
        industry_median = self._industry_medians.get(industry)
        if industry_median is not None:
            above_median = clamped_pred > industry_median
        else:
            above_median = clamped_pred > 1.5 * payload.historical_spend_amount
            logger.warning("No industry median for '%s'; using 1.5x fallback.", industry)

        cross_sell = (
            payload.purchase_frequency_yearly >= clv_config.cross_sell_min_purchase_frequency
            and above_median
        )

        # ── Timing confidence (multi-factor, not just engagement/100) ───────
        eng_component  = (eng / 100.0) * 0.50
        freq_cap       = min(freq / 12.0, 1.0)
        freq_component = freq_cap * 0.30
        median_component = 0.20 if above_median else 0.0
        timing_score   = round(eng_component + freq_component + median_component, 4)
        timing_score   = min(timing_score, 1.0)

        # ── Product ─────────────────────────────────────────────────────────
        cs_product = CROSS_SELL_PRODUCTS.get(industry, CROSS_SELL_DEFAULT)

        # ── Rationale ───────────────────────────────────────────────────────
        rationale_parts = []
        if eng_adj < 0:
            rationale_parts.append(f"high engagement ({eng:.0f}/100) suggests client is receptive")
        elif eng_adj > 0:
            rationale_parts.append(f"lower engagement ({eng:.0f}/100) warrants a longer warm-up")
        if freq_adj < 0:
            rationale_parts.append(f"frequent buyer ({freq}/yr) accelerates timing")
        if tenure_adj > 0:
            rationale_parts.append(f"newer client ({tenure} months) needs more runway")
        if renewal_adj < 0:
            rationale_parts.append("annual contract renewal approaching")
        if not rationale_parts:
            rationale_parts.append("standard 45-day pitch window applies")

        rationale = f"Pitch timing adjusted for this client: {'; '.join(rationale_parts)}.{pitch_rationale_detail}"

        return {
            "cross_sell_detected": cross_sell,
            "secondary_product_slug": cs_product["slug"],
            "secondary_product_name": cs_product["name"],
            "optimal_pitch_target_date": pitch_date_str,
            "days_until_optimal_pitch": total_offset,
            "timing_confidence_score": timing_score,
            "pitch_rationale": rationale,
        }
