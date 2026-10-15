"""
test_clv_math.py — Normal-range correctness tests for the CLV pipeline v2.

Verifies:
  - predicted_clv_amount is a positive finite float, never null/NaN.
  - Output validates against CLVOutputPayload schema.
  - SHAP drivers list has up to 3 entries with correct structure.
  - Upsell confidence score is in [0, 1].
  - Cross-sell pitch date is YYYY-MM-DD and > today.
  - Timing confidence is a multi-factor score in [0, 1] (not just eng/100).
  - Product names are non-empty strings.
  - All 10 industries return valid predictions.
"""
from __future__ import annotations

import math
import uuid
from datetime import date

import pytest

from app.clv_schemas import CLVInputPayload, CLVOutputPayload, IndustryType
from pipelines.clv_predictor import CLVPredictor


@pytest.fixture(scope="module")
def predictor() -> CLVPredictor:
    """Module-scoped predictor fixture — loads artefacts once per test session."""
    return CLVPredictor()


def _payload(**overrides) -> CLVInputPayload:
    """Build a normal-range CLVInputPayload with sensible defaults."""
    defaults = dict(
        company_id=str(uuid.uuid4()),
        historical_spend_amount=20_000.0,
        purchase_frequency_yearly=6,
        industry_type=IndustryType.TECHNOLOGY,
        rolling_engagement_score=78.0,
        contract_tenure_months=24,
        contract_start_date="2024-10-01",
    )
    defaults.update(overrides)
    return CLVInputPayload(**defaults)


class TestBasicOutputContracts:
    """Output schema contract tests."""

    def test_predicted_clv_positive_finite(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_payload())
        assert isinstance(result.predicted_clv_amount, float)
        assert result.predicted_clv_amount > 0.0
        assert math.isfinite(result.predicted_clv_amount)

    def test_output_validates_schema(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_payload())
        validated = CLVOutputPayload.model_validate(result.model_dump())
        assert validated.company_id == result.company_id

    def test_confidence_in_range(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_payload())
        assert 0.0 <= result.confidence_interval_percentage <= 100.0

    def test_prediction_timestamp_format(self, predictor: CLVPredictor) -> None:
        from datetime import datetime
        result = predictor.predict(_payload())
        # Should parse as ISO 8601 UTC
        dt = datetime.strptime(result.prediction_timestamp, "%Y-%m-%dT%H:%M:%SZ")
        assert dt is not None


class TestSHAPDrivers:
    """Tests for the SHAP explainability output."""

    def test_shap_drivers_not_empty(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_payload())
        assert len(result.expansion_analytics.shap_drivers) > 0

    def test_shap_drivers_max_three(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_payload())
        assert len(result.expansion_analytics.shap_drivers) <= 3

    def test_shap_driver_fields(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_payload())
        for driver in result.expansion_analytics.shap_drivers:
            # SHAPDriverEntry is a Pydantic model — use attribute access
            assert isinstance(driver.feature, str)
            assert driver.direction in ("positive", "negative")
            assert 0.0 <= driver.contribution_pct <= 100.0

    def test_top_driver_reason_non_empty(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_payload())
        reason = result.expansion_analytics.top_prediction_driver_reason
        assert isinstance(reason, str) and reason.strip()


class TestUpsellLogic:
    """Tests for ML-driven upsell detection."""

    def test_upsell_score_in_range(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_payload())
        score = result.expansion_analytics.upsell_confidence_score
        assert 0.0 <= score <= 1.0

    def test_high_engagement_long_tenure_upsell(self, predictor: CLVPredictor) -> None:
        """High engagement + long tenure should yield a high upsell score."""
        result = predictor.predict(_payload(
            rolling_engagement_score=92.0,
            contract_tenure_months=36,
        ))
        # Score should be reasonably high (ML model trained for this)
        score = result.expansion_analytics.upsell_confidence_score
        assert score > 0.4, f"Expected high upsell score, got {score}"

    def test_low_engagement_upsell_blocked(self, predictor: CLVPredictor) -> None:
        """Very low engagement should block upsell regardless of ML score."""
        result = predictor.predict(_payload(
            rolling_engagement_score=20.0,
            contract_tenure_months=24,
        ))
        assert result.expansion_analytics.upsell_opportunity_detected is False

    def test_short_tenure_upsell_blocked(self, predictor: CLVPredictor) -> None:
        """New clients (< 10 months) must not be flagged for upsell."""
        result = predictor.predict(_payload(
            rolling_engagement_score=95.0,
            contract_tenure_months=3,
        ))
        assert result.expansion_analytics.upsell_opportunity_detected is False

    def test_product_names_non_empty(self, predictor: CLVPredictor) -> None:
        """All industries must return a non-empty product recommendation."""
        for industry in IndustryType:
            result = predictor.predict(_payload(industry_type=industry))
            product = result.expansion_analytics.recommended_target_product
            assert product.strip(), f"Empty product for {industry.value}"


class TestCrossSellLogic:
    """Tests for dynamic cross-sell timing."""

    def test_pitch_date_in_future(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_payload())
        pitch = result.cross_sell_opportunity_window.optimal_pitch_target_date
        pitch_date = date.fromisoformat(pitch)
        assert pitch_date > date.today(), "Pitch date must be in the future"

    def test_timing_confidence_in_range(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_payload())
        score = result.cross_sell_opportunity_window.timing_confidence_score
        assert 0.0 <= score <= 1.0

    def test_high_engagement_shorter_pitch_window(self, predictor: CLVPredictor) -> None:
        """High-engagement client should have shorter pitch offset than low-engagement."""
        r_high = predictor.predict(_payload(rolling_engagement_score=92.0))
        r_low  = predictor.predict(_payload(rolling_engagement_score=25.0))
        days_high = r_high.cross_sell_opportunity_window.days_until_optimal_pitch
        days_low  = r_low.cross_sell_opportunity_window.days_until_optimal_pitch
        assert days_high < days_low, (
            f"High-engagement client should have shorter window: {days_high} vs {days_low}"
        )

    def test_pitch_rationale_non_empty(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_payload())
        rationale = result.cross_sell_opportunity_window.pitch_rationale
        assert rationale.strip(), "pitch_rationale must not be empty"

    def test_zero_frequency_no_cross_sell(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_payload(purchase_frequency_yearly=0))
        assert result.cross_sell_opportunity_window.cross_sell_detected is False

    def test_cross_sell_product_names_non_empty(self, predictor: CLVPredictor) -> None:
        """All industries must return non-empty cross-sell product names and slugs."""
        for industry in IndustryType:
            result = predictor.predict(_payload(industry_type=industry))
            name = result.cross_sell_opportunity_window.secondary_product_name
            slug = result.cross_sell_opportunity_window.secondary_product_slug
            assert name.strip(), f"Empty cross-sell name for {industry.value}"
            assert slug.strip(), f"Empty cross-sell slug for {industry.value}"

    def test_different_clients_different_pitch_dates(self, predictor: CLVPredictor) -> None:
        """Two clients with different profiles should get different pitch dates."""
        r1 = predictor.predict(_payload(
            rolling_engagement_score=90.0,
            purchase_frequency_yearly=12,
            contract_tenure_months=36,
        ))
        r2 = predictor.predict(_payload(
            rolling_engagement_score=30.0,
            purchase_frequency_yearly=1,
            contract_tenure_months=3,
        ))
        d1 = r1.cross_sell_opportunity_window.days_until_optimal_pitch
        d2 = r2.cross_sell_opportunity_window.days_until_optimal_pitch
        assert d1 != d2, (
            f"Different clients should have different pitch offsets: both got {d1} days"
        )


class TestAllIndustries:
    """All 10 industries must produce valid, non-throwing predictions."""

    def test_all_industries_no_exception(self, predictor: CLVPredictor) -> None:
        for industry in IndustryType:
            result = predictor.predict(_payload(industry_type=industry))
            assert result is not None
            assert math.isfinite(result.predicted_clv_amount)
            assert result.predicted_clv_amount >= 0


class TestDateValidation:
    """Verify strict validation of contract_start_date."""

    def test_invalid_date_format_raises_validation_error(self) -> None:
        from pydantic import ValidationError

        invalid_dates = ["invalid-date", "2024-15-40", "15/10/2024", "2024-02-31"]
        for bad_date in invalid_dates:
            with pytest.raises(ValidationError) as exc_info:
                _payload(contract_start_date=bad_date)
            assert "contract_start_date" in str(exc_info.value)

    def test_future_date_raises_validation_error(self) -> None:
        from datetime import timedelta
        from pydantic import ValidationError

        future_date = (date.today() + timedelta(days=5)).isoformat()
        with pytest.raises(ValidationError) as exc_info:
            _payload(contract_start_date=future_date)
        assert "cannot be in the future" in str(exc_info.value)

    def test_valid_past_date_succeeds(self) -> None:
        p = _payload(contract_start_date="2023-05-15")
        assert p.contract_start_date == "2023-05-15"

    def test_empty_or_none_date_succeeds(self) -> None:
        p1 = _payload(contract_start_date=None)
        assert p1.contract_start_date is None

        p2 = _payload(contract_start_date="   ")
        assert p2.contract_start_date is None

