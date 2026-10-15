"""
generate_dataset.py — Detagenix CLV Synthetic Dataset Generator v2

Generates a realistic, industry-calibrated training dataset for the CLV
Prediction Engine. Each industry vertical has its own realistic spending
distribution, engagement profiles, and CLV multiplier range — preventing the
model from learning trivial patterns from homogeneous data.

Run:
    python generate_dataset.py
    python generate_dataset.py --rows 5000 --seed 99 --output my_data.csv

Output: clv_synthetic_dataset.csv (sibling of this script)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Industry-specific profile configuration
# Each entry defines realistic parameter distributions for that B2B vertical.
# ---------------------------------------------------------------------------
INDUSTRY_PROFILES: dict[str, dict] = {
    "Technology": {
        "spend_mean": 18_000, "spend_std": 9_000,
        "freq_min": 4, "freq_max": 12,
        "tenure_mean": 22, "tenure_std": 10,
        "engagement_mean": 74, "engagement_std": 14,
        "clv_multiplier_range": (3.5, 9.0),   # High growth potential
        "weight": 0.14,                         # Share of dataset rows
    },
    "Finance": {
        "spend_mean": 28_000, "spend_std": 12_000,
        "freq_min": 3, "freq_max": 8,
        "tenure_mean": 30, "tenure_std": 12,
        "engagement_mean": 65, "engagement_std": 12,
        "clv_multiplier_range": (3.0, 8.5),
        "weight": 0.13,
    },
    "Healthcare": {
        "spend_mean": 22_000, "spend_std": 10_000,
        "freq_min": 2, "freq_max": 7,
        "tenure_mean": 28, "tenure_std": 11,
        "engagement_mean": 60, "engagement_std": 13,
        "clv_multiplier_range": (2.8, 7.5),
        "weight": 0.12,
    },
    "Retail": {
        "spend_mean": 12_000, "spend_std": 6_000,
        "freq_min": 6, "freq_max": 18,
        "tenure_mean": 15, "tenure_std": 8,
        "engagement_mean": 68, "engagement_std": 16,
        "clv_multiplier_range": (2.5, 7.0),
        "weight": 0.11,
    },
    "Manufacturing": {
        "spend_mean": 32_000, "spend_std": 14_000,
        "freq_min": 2, "freq_max": 6,
        "tenure_mean": 36, "tenure_std": 14,
        "engagement_mean": 55, "engagement_std": 11,
        "clv_multiplier_range": (2.0, 6.5),
        "weight": 0.10,
    },
    "Logistics & Supply Chain": {
        "spend_mean": 25_000, "spend_std": 11_000,
        "freq_min": 4, "freq_max": 14,
        "tenure_mean": 20, "tenure_std": 9,
        "engagement_mean": 62, "engagement_std": 13,
        "clv_multiplier_range": (2.8, 7.8),
        "weight": 0.10,
    },
    "Education": {
        "spend_mean": 9_000, "spend_std": 4_500,
        "freq_min": 1, "freq_max": 5,
        "tenure_mean": 18, "tenure_std": 8,
        "engagement_mean": 70, "engagement_std": 15,
        "clv_multiplier_range": (2.0, 5.5),
        "weight": 0.08,
    },
    "Real Estate": {
        "spend_mean": 15_000, "spend_std": 7_500,
        "freq_min": 2, "freq_max": 7,
        "tenure_mean": 24, "tenure_std": 10,
        "engagement_mean": 58, "engagement_std": 14,
        "clv_multiplier_range": (2.5, 6.5),
        "weight": 0.08,
    },
    "Hospitality": {
        "spend_mean": 10_000, "spend_std": 5_000,
        "freq_min": 3, "freq_max": 10,
        "tenure_mean": 12, "tenure_std": 7,
        "engagement_mean": 72, "engagement_std": 17,
        "clv_multiplier_range": (2.2, 5.8),
        "weight": 0.07,
    },
    "E-commerce": {
        "spend_mean": 14_000, "spend_std": 7_000,
        "freq_min": 8, "freq_max": 24,
        "tenure_mean": 14, "tenure_std": 7,
        "engagement_mean": 76, "engagement_std": 15,
        "clv_multiplier_range": (3.0, 8.0),
        "weight": 0.07,
    },
}

# Contract start dates: realistic range for B2B enterprise accounts
CONTRACT_START_MIN = pd.Timestamp("2021-01-01")
CONTRACT_START_MAX = pd.Timestamp("2025-06-30")


def _clip_positive(val: float, min_val: float = 0.0) -> float:
    return max(val, min_val)


def generate_dataset(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic CLV training dataset with realistic, industry-calibrated
    distributions.

    Key design decisions:
    - CLV is computed as spend * multiplier + noise, where the multiplier is
      driven by engagement, tenure, and frequency — so the model can learn
      real relationships, not just identity mappings.
    - Each industry has its own spending scale and CLV multiplier range,
      creating genuine cross-industry variation.
    - contract_start_date is included so downstream logic can compute actual
      tenure from real dates (not just use the pre-computed integer).

    Args:
        n_rows: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame with all required columns.
    """
    rng = np.random.default_rng(seed)
    industries = list(INDUSTRY_PROFILES.keys())
    weights = [INDUSTRY_PROFILES[i]["weight"] for i in industries]
    # Normalize weights
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    rows = []
    for i in range(n_rows):
        # --- Pick industry ---
        industry = rng.choice(industries, p=weights)
        prof = INDUSTRY_PROFILES[industry]

        # --- Company ID ---
        company_id = f"client-{i+1:05d}"

        # --- Contract start date (realistic B2B range) ---
        total_days = (CONTRACT_START_MAX - CONTRACT_START_MIN).days
        start_offset = int(rng.integers(0, total_days))
        contract_start_date = CONTRACT_START_MIN + pd.Timedelta(days=start_offset)

        # --- Contract tenure (months since start) ---
        reference_date = pd.Timestamp("2026-10-01")
        actual_months = max(
            1,
            int((reference_date - contract_start_date).days / 30.44)
        )
        # Add small jitter to simulate data entry lag
        contract_tenure_months = max(1, actual_months + int(rng.integers(-1, 2)))

        # --- Historical spend ---
        raw_spend = rng.normal(prof["spend_mean"], prof["spend_std"])
        historical_spend_amount = round(_clip_positive(raw_spend, 500.0), 2)

        # --- Purchase frequency ---
        purchase_frequency_yearly = int(
            rng.integers(prof["freq_min"], prof["freq_max"] + 1)
        )

        # --- Rolling engagement score (0–100) ---
        raw_eng = rng.normal(prof["engagement_mean"], prof["engagement_std"])
        rolling_engagement_score = round(
            float(np.clip(raw_eng, 0.0, 100.0)), 2
        )

        # --- CLV target: driven by real behavioral signals + noise ---
        # engagement effect: 0.5x to 1.5x multiplier based on score
        eng_factor = 0.5 + (rolling_engagement_score / 100.0)
        # tenure effect: longer tenure = more lifetime value realized
        tenure_factor = 1.0 + (contract_tenure_months / 60.0)
        # frequency effect: more frequent buyers = higher CLV
        freq_factor = 1.0 + (purchase_frequency_yearly / 20.0)

        # Base multiplier from industry range, skewed by factors
        base_mult_lo, base_mult_hi = prof["clv_multiplier_range"]
        base_mult = rng.uniform(base_mult_lo, base_mult_hi)

        clv_raw = (
            historical_spend_amount
            * base_mult
            * eng_factor
            * tenure_factor
            * freq_factor
        )
        # Add realistic noise (±8%)
        noise = rng.uniform(0.92, 1.08)
        predicted_clv_amount = round(_clip_positive(clv_raw * noise, 0.0), 2)

        rows.append(
            {
                "company_id": company_id,
                "contract_start_date": contract_start_date.strftime("%Y-%m-%d"),
                "historical_spend_amount": historical_spend_amount,
                "purchase_frequency_yearly": purchase_frequency_yearly,
                "industry_type": industry,
                "rolling_engagement_score": rolling_engagement_score,
                "contract_tenure_months": contract_tenure_months,
                "predicted_clv_amount": predicted_clv_amount,
            }
        )

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CLV synthetic training dataset."
    )
    parser.add_argument(
        "--rows", type=int, default=4000,
        help="Number of rows to generate (default: 4000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: clv_synthetic_dataset.csv next to this script)"
    )
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else (
        Path(__file__).parent / "clv_synthetic_dataset.csv"
    )

    print(f"[dataset] Generating {args.rows} rows with seed={args.seed} ...")
    df = generate_dataset(n_rows=args.rows, seed=args.seed)

    df.to_csv(out_path, index=False)
    print(f"[dataset] Saved → {out_path}")
    print(f"[dataset] Shape: {df.shape}")
    print("\nIndustry distribution:")
    print(df["industry_type"].value_counts().to_string())
    print(f"\nCLV stats (predicted_clv_amount):")
    print(df["predicted_clv_amount"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
