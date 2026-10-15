"""
CLV API Routes — Detagenix CLV Prediction Engine v2

Exposes:
    POST /api/v1/clv/predict           Single-account prediction
    POST /api/v1/clv/batch-predict     Batch prediction (list of accounts)
    GET  /api/v1/clv/history/{company_id}  History for one client
    GET  /api/v1/clv/summary           Aggregate stats (for leadership dashboard)
    GET  /api/v1/clv/recent            Last N predictions
"""
from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query, status

from app.clv_schemas import CLVInputPayload, CLVOutputPayload
from app.config import clv_config
from database.db import (
    get_summary_stats,
    init_db,
    query_by_company,
    query_recent_predictions,
    save_prediction,
)
from pipelines.clv_predictor import CLVPredictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/clv", tags=["CLV Prediction Engine"])

# Initialise DB tables on import
init_db()


@lru_cache(maxsize=1)
def get_predictor() -> CLVPredictor:
    """
    Lazy-load the CLVPredictor singleton (model artefacts loaded once).

    Raises:
        HTTPException 503: If model artefacts are missing.
    """
    try:
        return CLVPredictor()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "CLV model artefacts not found. "
                "Run train_clv_model.py first to generate them."
            ),
        ) from exc


# ── Single Prediction ────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=CLVOutputPayload,
    summary="Predict CLV for a single enterprise account",
    description=(
        "Runs the full Detagenix CLV pipeline: XGBoost regression → SHAP "
        "explainability → ML upsell scoring → dynamic cross-sell timing. "
        "Result is persisted to the CLV predictions database."
    ),
)
def predict_clv(payload: CLVInputPayload) -> CLVOutputPayload:
    """
    Single-account CLV prediction.

    Args:
        payload: Validated CLVInputPayload.

    Returns:
        CLVOutputPayload with prediction, confidence, upsell, and cross-sell signals.
    """
    predictor = get_predictor()
    try:
        result = predictor.predict(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during CLV prediction for %s", payload.company_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction.",
        ) from exc

    # Persist to DB
    try:
        row_id = save_prediction(payload.model_dump(mode="json"), result.model_dump(mode="json"))
        logger.info("Prediction saved to DB: row_id=%d company_id=%s", row_id, payload.company_id)
    except Exception as db_exc:
        logger.warning("DB save failed (non-fatal): %s", db_exc)

    return result


# ── Batch Prediction ─────────────────────────────────────────────────────────

@router.post(
    "/batch-predict",
    response_model=list[CLVOutputPayload],
    summary="Predict CLV for a batch of accounts",
    description=(
        "Accepts a list of CLVInputPayload objects. Designed for the weekly "
        "Sunday CRON batch job. Individual failures are caught and logged "
        "without aborting the whole batch. Results are persisted to DB."
    ),
)
def batch_predict_clv(payloads: list[CLVInputPayload]) -> list[CLVOutputPayload]:
    """
    Batch CLV prediction endpoint.

    Args:
        payloads: List of CLVInputPayload objects.

    Returns:
        List of CLVOutputPayload objects (successful predictions only).
    """
    if not payloads:
        return []

    predictor = get_predictor()
    results: list[CLVOutputPayload] = []

    for payload in payloads:
        try:
            result = predictor.predict(payload)
            results.append(result)
            # Persist each result
            try:
                save_prediction(
                    payload.model_dump(mode="json"),
                    result.model_dump(mode="json"),
                )
            except Exception as db_exc:
                logger.warning("DB save failed for %s: %s", payload.company_id, db_exc)
        except Exception as exc:
            logger.error(
                "Batch prediction failed for company_id=%s: %s",
                payload.company_id, exc,
            )

    # ── Drift detection ──────────────────────────────────────────────────────
    if results and predictor.training_engagement_avg > 0:
        batch_avg = sum(p.rolling_engagement_score for p in payloads) / len(payloads)
        diff_pct  = abs(batch_avg - predictor.training_engagement_avg) / predictor.training_engagement_avg * 100.0
        if diff_pct > clv_config.drift_engagement_threshold_pct:
            logger.warning(
                "Data drift detected: batch avg engagement (%.1f) differs from "
                "training baseline (%.1f) by %.1f%% — consider retraining.",
                batch_avg, predictor.training_engagement_avg, diff_pct,
            )

    return results


# ── History & Analytics ──────────────────────────────────────────────────────

@router.get(
    "/history/{company_id}",
    summary="Get prediction history for a client",
    description="Returns all CLV predictions stored in the database for a given company_id.",
)
def get_company_history(company_id: str) -> list[dict]:
    """
    Retrieve all stored CLV predictions for a specific company.

    Args:
        company_id: The client identifier string.

    Returns:
        List of prediction records (newest first).
    """
    rows = query_by_company(company_id)
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No predictions found for company_id: {company_id}",
        )
    return rows


@router.get(
    "/recent",
    summary="Get recent predictions",
    description="Returns the N most recent CLV predictions across all clients.",
)
def get_recent_predictions(
    limit: int = Query(default=10, ge=1, le=100, description="Number of records to return")
) -> list[dict]:
    """Return the most recent N predictions."""
    return query_recent_predictions(limit=limit)


@router.get(
    "/summary",
    summary="Executive summary statistics",
    description=(
        "Returns aggregate statistics across all stored CLV predictions: "
        "total count, average predicted CLV, upsell rate, cross-sell rate, "
        "and average confidence. Designed for leadership dashboard integration."
    ),
)
def get_summary() -> dict:
    """Return aggregate prediction statistics for the executive dashboard."""
    stats = get_summary_stats()
    if not stats or stats.get("total_predictions") == 0:
        return {"message": "No predictions in database yet. Run /predict first."}
    return stats
