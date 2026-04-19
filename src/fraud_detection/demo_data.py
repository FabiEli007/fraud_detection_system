from __future__ import annotations

import math

import numpy as np
import pandas as pd


def generate_demo_transactions(n_samples: int = 15000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    amount = rng.gamma(shape=2.4, scale=95.0, size=n_samples).round(2)
    hour = rng.integers(0, 24, size=n_samples)
    distance_from_home_km = rng.gamma(shape=2.0, scale=18.0, size=n_samples).round(2)
    merchant_risk_score = rng.beta(a=2.0, b=6.0, size=n_samples).round(4)
    avg_amount_30d = np.maximum(amount * rng.uniform(0.55, 1.45, size=n_samples), 10.0).round(2)
    velocity_1h = rng.poisson(0.8, size=n_samples)
    velocity_24h = velocity_1h + rng.poisson(2.2, size=n_samples)
    device_change = rng.binomial(1, 0.09, size=n_samples)
    card_present = rng.binomial(1, 0.74, size=n_samples)
    international = rng.binomial(1, 0.07, size=n_samples)
    account_age_days = rng.integers(20, 3650, size=n_samples)
    failed_pin_attempts = rng.poisson(0.25, size=n_samples)
    num_chargebacks_6m = rng.poisson(0.1, size=n_samples)
    is_weekend = rng.binomial(1, 2 / 7, size=n_samples)
    distance_ratio = np.divide(
        distance_from_home_km,
        np.maximum(rng.normal(8.0, 2.0, size=n_samples), 1.0),
    )
    amount_vs_usual = np.divide(amount, np.maximum(avg_amount_30d, 1.0))

    latent_risk = (
        0.9 * np.log1p(amount)
        + 0.05 * distance_from_home_km
        + 2.4 * merchant_risk_score
        + 0.9 * international
        + 0.8 * device_change
        + 0.7 * (1 - card_present)
        + 0.35 * np.maximum(velocity_1h - 1, 0)
        + 0.10 * velocity_24h
        + 0.22 * failed_pin_attempts
        + 0.25 * num_chargebacks_6m
        + 0.45 * (amount_vs_usual > 2.0)
        + 0.35 * (distance_ratio > 2.5)
        + 0.30 * np.isin(hour, [0, 1, 2, 3, 4, 5, 23]).astype(int)
        + 0.10 * is_weekend
        - 0.0006 * account_age_days
        + rng.normal(0.0, 0.75, size=n_samples)
    )

    cutoff = np.quantile(latent_risk, 0.982)
    is_fraud = (latent_risk >= cutoff).astype(int)

    transaction_ids = [f"TXN-{index:07d}" for index in range(1, n_samples + 1)]

    data = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "amount": amount,
            "hour": hour,
            "distance_from_home_km": distance_from_home_km,
            "merchant_risk_score": merchant_risk_score,
            "avg_amount_30d": avg_amount_30d,
            "velocity_1h": velocity_1h,
            "velocity_24h": velocity_24h,
            "device_change": device_change,
            "card_present": card_present,
            "international": international,
            "account_age_days": account_age_days,
            "failed_pin_attempts": failed_pin_attempts,
            "num_chargebacks_6m": num_chargebacks_6m,
            "is_weekend": is_weekend,
            "distance_ratio": np.round(distance_ratio, 3),
            "amount_vs_usual": np.round(amount_vs_usual, 3),
            "is_fraud": is_fraud,
        }
    )

    # Ensure the generated target keeps a realistic positive count even on small samples.
    if data["is_fraud"].sum() < max(12, math.ceil(0.01 * n_samples)):
        ranked_index = np.argsort(latent_risk)[-max(12, math.ceil(0.015 * n_samples)) :]
        data["is_fraud"] = 0
        data.iloc[ranked_index, data.columns.get_loc("is_fraud")] = 1

    return data
