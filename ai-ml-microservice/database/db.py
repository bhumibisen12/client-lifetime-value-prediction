"""
database/db.py — Detagenix CLV Prediction Database Layer

SQLite-backed persistence for CLV prediction results. Every prediction is
stored with full output payload + metadata for audit trails, trend analysis,
and leadership reporting.

Schema:
    clv_predictions table — one row per prediction call
    prediction_drivers table — top SHAP drivers per prediction (FK)
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from app.config import clv_config

# Resolve absolute path relative to the microservice root
_HERE = Path(__file__).parent.parent
DB_PATH = _HERE / clv_config.db_path
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─── SQL Definitions ────────────────────────────────────────────────────────
_CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS clv_predictions (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_timestamp        TEXT    NOT NULL,
    company_id                  TEXT    NOT NULL,
    industry_type               TEXT    NOT NULL,
    historical_spend_amount     REAL    NOT NULL,
    purchase_frequency_yearly   INTEGER NOT NULL,
    rolling_engagement_score    REAL    NOT NULL,
    contract_tenure_months      INTEGER NOT NULL,
    predicted_clv_amount        REAL    NOT NULL,
    confidence_interval_pct     REAL    NOT NULL,
    upsell_opportunity_detected INTEGER NOT NULL,   -- 0 or 1
    upsell_confidence_score     REAL    NOT NULL,
    recommended_target_product  TEXT    NOT NULL,
    cross_sell_detected         INTEGER NOT NULL,   -- 0 or 1
    secondary_product_name      TEXT    NOT NULL,
    optimal_pitch_target_date   TEXT    NOT NULL,
    days_until_optimal_pitch    INTEGER NOT NULL,
    timing_confidence_score     REAL    NOT NULL,
    top_prediction_driver_reason TEXT   NOT NULL,
    raw_shap_drivers_json       TEXT    NOT NULL    -- JSON blob of all 3 drivers
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_clv_company
    ON clv_predictions (company_id, prediction_timestamp);
"""

_INSERT_PREDICTION = """
INSERT INTO clv_predictions (
    prediction_timestamp,
    company_id,
    industry_type,
    historical_spend_amount,
    purchase_frequency_yearly,
    rolling_engagement_score,
    contract_tenure_months,
    predicted_clv_amount,
    confidence_interval_pct,
    upsell_opportunity_detected,
    upsell_confidence_score,
    recommended_target_product,
    cross_sell_detected,
    secondary_product_name,
    optimal_pitch_target_date,
    days_until_optimal_pitch,
    timing_confidence_score,
    top_prediction_driver_reason,
    raw_shap_drivers_json
) VALUES (
    :prediction_timestamp,
    :company_id,
    :industry_type,
    :historical_spend_amount,
    :purchase_frequency_yearly,
    :rolling_engagement_score,
    :contract_tenure_months,
    :predicted_clv_amount,
    :confidence_interval_pct,
    :upsell_opportunity_detected,
    :upsell_confidence_score,
    :recommended_target_product,
    :cross_sell_detected,
    :secondary_product_name,
    :optimal_pitch_target_date,
    :days_until_optimal_pitch,
    :timing_confidence_score,
    :top_prediction_driver_reason,
    :raw_shap_drivers_json
);
"""


def init_db() -> None:
    """Create database tables if they don't already exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_CREATE_PREDICTIONS)
        conn.execute(_CREATE_INDEX)
        conn.commit()


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for a short-lived SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_prediction(input_payload: dict, output_payload: dict) -> int:
    """
    Persist a CLV prediction to the database.

    Args:
        input_payload:  dict of CLVInputPayload fields.
        output_payload: dict of CLVOutputPayload fields (from .model_dump()).

    Returns:
        Row ID of the inserted record.
    """
    ea  = output_payload["expansion_analytics"]
    csw = output_payload["cross_sell_opportunity_window"]

    record = {
        "prediction_timestamp":        output_payload["prediction_timestamp"],
        "company_id":                  output_payload["company_id"],
        "industry_type":               input_payload.get("industry_type", ""),
        "historical_spend_amount":     input_payload.get("historical_spend_amount", 0.0),
        "purchase_frequency_yearly":   input_payload.get("purchase_frequency_yearly", 0),
        "rolling_engagement_score":    input_payload.get("rolling_engagement_score", 0.0),
        "contract_tenure_months":      input_payload.get("contract_tenure_months", 0),
        "predicted_clv_amount":        output_payload["predicted_clv_amount"],
        "confidence_interval_pct":     output_payload["confidence_interval_percentage"],
        "upsell_opportunity_detected": int(ea["upsell_opportunity_detected"]),
        "upsell_confidence_score":     ea["upsell_confidence_score"],
        "recommended_target_product":  ea["recommended_target_product"],
        "cross_sell_detected":         int(csw["cross_sell_detected"]),
        "secondary_product_name":      csw["secondary_product_name"],
        "optimal_pitch_target_date":   csw["optimal_pitch_target_date"],
        "days_until_optimal_pitch":    csw["days_until_optimal_pitch"],
        "timing_confidence_score":     csw["timing_confidence_score"],
        "top_prediction_driver_reason": ea["top_prediction_driver_reason"],
        "raw_shap_drivers_json":       json.dumps(ea.get("shap_drivers", [])),
    }

    with get_connection() as conn:
        cursor = conn.execute(_INSERT_PREDICTION, record)
        return cursor.lastrowid


def query_recent_predictions(limit: int = 20) -> list[dict]:
    """
    Retrieve the most recent CLV predictions from the database.

    Args:
        limit: Maximum number of rows to return.

    Returns:
        List of dicts, newest first.
    """
    sql = """
        SELECT * FROM clv_predictions
        ORDER BY prediction_timestamp DESC
        LIMIT ?;
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def query_by_company(company_id: str) -> list[dict]:
    """Retrieve all predictions for a specific company_id."""
    sql = """
        SELECT * FROM clv_predictions
        WHERE company_id = ?
        ORDER BY prediction_timestamp DESC;
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (company_id,)).fetchall()
    return [dict(r) for r in rows]


def get_summary_stats() -> dict:
    """
    Compute aggregate statistics for the executive summary view.

    Returns:
        Dict with count, avg CLV, upsell rate, cross-sell rate.
    """
    sql = """
        SELECT
            COUNT(*)                        AS total_predictions,
            ROUND(AVG(predicted_clv_amount), 2) AS avg_predicted_clv,
            ROUND(AVG(upsell_opportunity_detected) * 100, 1) AS upsell_rate_pct,
            ROUND(AVG(cross_sell_detected) * 100, 1)         AS cross_sell_rate_pct,
            ROUND(AVG(confidence_interval_pct), 2)           AS avg_confidence_pct
        FROM clv_predictions;
    """
    with get_connection() as conn:
        row = conn.execute(sql).fetchone()
    return dict(row) if row else {}
