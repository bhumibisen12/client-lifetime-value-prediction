"""
test_clv_outliers.py — Edge-case and outlier tests for the CLV pipeline v2.

Tests:
  - Zero spend → CLV = 0 (ceiling clamp at 0)
  - Per-industry ceilings hold for all 10 industries
  - Zero tenure → no divide-by-zero
  - Extreme engagement boundaries (0 and 100)
  - Very high spend accounts stay within ceiling
  - New clients (< 6 months) get longer pitch window than old clients
"""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

import pytest

from app.clv_schemas import CLVInputPayload, IndustryType
from pipelines.clv_predictor import CLVPredictor

_CEILINGS_PATH = (
    Path(__file__).parent.parent / "models_registry" / "industry_clv_ceilings.json"
)


def _load_ceilings() -> dict[str, float]:
    with open(_CEILINGS_PATH) as f:
        return json.load(f)["ceilings"]


@pytest.fixture(scope="module")
def predictor() -> CLVPredictor:
    """Module-scoped predictor — loaded once per session."""
    return CLVPredictor()


def _edge(**overrides) -> CLVInputPayload:
    """Build a high-value edge-case payload."""
    base = dict(
        company_id=str(uuid.uuid4()),
        historical_spend_amount=10_000.0,
        purchase_frequency_yearly=6,
        industry_type=IndustryType.TECHNOLOGY,
        rolling_engagement_score=95.0,
        contract_tenure_months=60,
        contract_start_date=None,
    )
    base.update(overrides)
    return CLVInputPayload(**base)


def _assert_ceiling_holds(result, payload: CLVInputPayload) -> None:
    """Assert per-industry ceiling is respected."""
    ceilings = _load_ceilings()
    industry = payload.industry_type.value
    ceiling_mult = ceilings.get(industry, 9.0)
    ceiling_val = payload.historical_spend_amount * ceiling_mult
    assert result.predicted_clv_amount <= ceiling_val + 0.01, (
        f"CLV {result.predicted_clv_amount:.2f} exceeds ceiling "
        f"{ceiling_val:.2f} ({ceiling_mult:.2f}x) for {industry}"
    )


class TestZeroSpendEdge:
    """Zero historical_spend_amount must produce CLV = 0."""

    def test_zero_spend_clv_is_zero(self, predictor: CLVPredictor) -> None:
        payload = _edge(
            historical_spend_amount=0.0,
            purchase_frequency_yearly=0,
        )
        result = predictor.predict(payload)
        assert result.predicted_clv_amount == 0.0, (
            f"Zero spend must produce 0 CLV, got {result.predicted_clv_amount}"
        )

    def test_zero_spend_no_exception(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_edge(historical_spend_amount=0.0))
        assert result is not None
        assert math.isfinite(result.predicted_clv_amount)


class TestCeilingClamp:
    """Per-industry ceiling must hold for every industry."""

    def test_ceiling_holds_all_industries(self, predictor: CLVPredictor) -> None:
        for industry in IndustryType:
            payload = _edge(
                historical_spend_amount=50_000.0,
                industry_type=industry,
            )
            result = predictor.predict(payload)
            _assert_ceiling_holds(result, payload)

    def test_ceiling_holds_very_high_spend(self, predictor: CLVPredictor) -> None:
        payload = _edge(
            historical_spend_amount=200_000.0,
            industry_type=IndustryType.FINANCE,
        )
        result = predictor.predict(payload)
        _assert_ceiling_holds(result, payload)

    def test_ceiling_holds_low_spend(self, predictor: CLVPredictor) -> None:
        payload = _edge(
            historical_spend_amount=500.0,
            industry_type=IndustryType.RETAIL,
        )
        result = predictor.predict(payload)
        _assert_ceiling_holds(result, payload)


class TestZeroTenure:
    """Zero contract tenure must not raise divide-by-zero errors."""

    def test_zero_tenure_no_exception(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_edge(contract_tenure_months=0))
        assert result is not None
        assert math.isfinite(result.predicted_clv_amount)

    def test_zero_tenure_ceiling_holds(self, predictor: CLVPredictor) -> None:
        payload = _edge(contract_tenure_months=0, historical_spend_amount=15_000.0)
        result = predictor.predict(payload)
        _assert_ceiling_holds(result, payload)

    def test_zero_tenure_no_upsell(self, predictor: CLVPredictor) -> None:
        """Upsell must be blocked when tenure < threshold (10 months)."""
        result = predictor.predict(_edge(
            contract_tenure_months=0,
            rolling_engagement_score=98.0,
        ))
        assert result.expansion_analytics.upsell_opportunity_detected is False


class TestEngagementBoundaries:
    """Min and max engagement values must not cause errors."""

    def test_zero_engagement_no_exception(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_edge(rolling_engagement_score=0.0))
        assert result is not None
        assert math.isfinite(result.predicted_clv_amount)

    def test_zero_engagement_ceiling_holds(self, predictor: CLVPredictor) -> None:
        payload = _edge(rolling_engagement_score=0.0)
        result  = predictor.predict(payload)
        _assert_ceiling_holds(result, payload)

    def test_zero_engagement_no_upsell(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_edge(
            rolling_engagement_score=0.0,
            contract_tenure_months=48,
        ))
        assert result.expansion_analytics.upsell_opportunity_detected is False

    def test_max_engagement_no_exception(self, predictor: CLVPredictor) -> None:
        result = predictor.predict(_edge(rolling_engagement_score=100.0))
        assert result is not None
        assert math.isfinite(result.predicted_clv_amount)

    def test_max_engagement_timing_score_high(self, predictor: CLVPredictor) -> None:
        """Max engagement should produce a high timing confidence score."""
        result = predictor.predict(_edge(
            rolling_engagement_score=100.0,
            purchase_frequency_yearly=10,
        ))
        score = result.cross_sell_opportunity_window.timing_confidence_score
        assert score >= 0.7, f"Max engagement should produce high timing score, got {score}"


class TestDynamicPitchDate:
    """Pitch date must vary per client — not be a fixed 45-day offset for all."""

    def test_new_client_longer_window_than_veteran(self, predictor: CLVPredictor) -> None:
        """New client (< 6 months) should get a longer pitch offset."""
        new_client = predictor.predict(_edge(contract_tenure_months=2))
        veteran    = predictor.predict(_edge(contract_tenure_months=48))
        days_new = new_client.cross_sell_opportunity_window.days_until_optimal_pitch
        days_vet = veteran.cross_sell_opportunity_window.days_until_optimal_pitch
        assert days_new > days_vet, (
            f"New client should wait longer: {days_new} days vs veteran {days_vet} days"
        )

    def test_pitch_date_always_future(self, predictor: CLVPredictor) -> None:
        """Pitch date must always be in the future."""
        from datetime import date
        for industry in IndustryType:
            result = predictor.predict(_edge(industry_type=industry))
            pitch_str = result.cross_sell_opportunity_window.optimal_pitch_target_date
            pitch_date = date.fromisoformat(pitch_str)
            assert pitch_date > date.today(), (
                f"Pitch date in the past for {industry.value}: {pitch_str}"
            )

    def test_pitch_date_minimum_7_days(self, predictor: CLVPredictor) -> None:
        """Even with maximum engagement, pitch date must be >= 7 days out."""
        result = predictor.predict(_edge(
            rolling_engagement_score=100.0,
            purchase_frequency_yearly=52,
            contract_tenure_months=120,
        ))
        days = result.cross_sell_opportunity_window.days_until_optimal_pitch
        assert days >= 7, f"Pitch date must be >= 7 days away, got {days}"
