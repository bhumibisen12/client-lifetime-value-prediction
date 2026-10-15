"""
Entry point for the CLV Prediction Engine API v2.

Run:
    uvicorn main:app --reload

Interactive docs:
    http://127.0.0.1:8000/docs   (Swagger UI)
    http://127.0.0.1:8000/redoc  (ReDoc)
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.clv_routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="CLV Prediction Engine",
    description=(
        "**Module 8 — Client Lifetime Value (CLV) Prediction Engine v2**\n\n"
        "Predicts enterprise account lifetime value using XGBoost + SHAP, "
        "flags upsell opportunities using an ML classifier, and computes "
        "per-client dynamic cross-sell pitch windows.\n\n"
        "**Stack:** XGBoost · SHAP · scikit-learn · FastAPI · SQLite\n\n"
        "**No LLM is used anywhere in this module.**\n\n"
        "**Industries supported:** Technology · Finance · Healthcare · "
        "Retail · Manufacturing · Logistics & Supply Chain · Education · "
        "Real Estate · Hospitality · E-commerce"
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integration across different PCs / origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root URL to interactive Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """
    Liveness check — confirms the API is running.

    Returns:
        dict with status, service name, and version.
    """
    return {
        "status": "ok",
        "service": "CLV Prediction Engine",
        "version": "2.0.0",
    }
