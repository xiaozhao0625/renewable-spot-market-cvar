# -*- coding: utf-8 -*-
"""
Generate the common source dataset used in the analysis:
multi_scenario_power_market_dataset.csv

Uses:
1. Multi-scenario modeling
2. Electricity spot-market optimization
3. Risk-control analysis
4. Revenue, deviation, price, scenario, and sensitivity analyses

Design:
- Period: 2025-01-01 00:00:00 to 2025-12-31 23:00:00
- Resolution: one hour
- Scenarios: five
- Scenario definitions:
    1 = Baseline
    2 = High wind
    3 = High solar
    4 = High load and high price
    5 = Low output risk

Output:
- multi_scenario_power_market_dataset.csv

Dependencies:
- pandas
- numpy
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# Global parameters
# =========================
SEED = 20260417
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PACKAGE_ROOT / "data" / "multi_scenario_power_market_dataset.csv"

START_TIME = "2025-01-01 00:00:00"
END_TIME = "2025-12-31 23:00:00"
FREQ = "h"

SCENARIO_NAMES = {
    1: "Baseline",
    2: "High_Wind",
    3: "High_Solar",
    4: "High_Load_High_Price",
    5: "Low_Output_Risk",
}

SCENARIO_PROB = {
    1: 0.30,
    2: 0.20,
    3: 0.20,
    4: 0.15,
    5: 0.15,
}


# =========================
# Configuration
# =========================
@dataclass
class CapacityConfig:
    wind_capacity_mw: float = 320.0
    solar_capacity_mw: float = 260.0
    load_base_mw: float = 420.0
    thermal_support_mw: float = 180.0


# =========================
# Utilities
# =========================
def clip(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip an array to the specified interval."""
    return np.clip(arr, low, high)


def safe_round(series: np.ndarray | pd.Series, digits: int = 4) -> np.ndarray:
    """Round numeric values consistently."""
    return np.round(series, digits)


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct calendar and time features."""
    dt = df["timestamp"]
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek  # Monday=0
    df["dayofyear"] = dt.dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Seasonal indicators.
    def month_to_season(m: int) -> str:
        if m in (12, 1, 2):
            return "Winter"
        if m in (3, 4, 5):
            return "Spring"
        if m in (6, 7, 8):
            return "Summer"
        return "Autumn"

    df["season"] = df["month"].map(month_to_season)
    return df


def solar_shape(hour: np.ndarray, dayofyear: np.ndarray) -> np.ndarray:
    """
    Construct the photovoltaic daylight profile.
    Output is concentrated in daylight hours, with longer days and higher
    peaks in summer.
    """
    # Seasonal daylight duration ranges from approximately 10 to 14 hours.
    daylight_length = 12 + 2 * np.sin(2 * np.pi * (dayofyear - 80) / 365.0)
    sunrise = 12 - daylight_length / 2
    sunset = 12 + daylight_length / 2

    shape = np.zeros_like(hour, dtype=float)
    mask = (hour >= sunrise) & (hour <= sunset)

    # Map daylight hours to [0, pi].
    x = np.zeros_like(hour, dtype=float)
    x[mask] = (hour[mask] - sunrise[mask]) / np.maximum(
        sunset[mask] - sunrise[mask], 1e-6
    ) * np.pi

    shape[mask] = np.sin(x[mask]) ** 1.6

    # Seasonal gain is higher in summer and lower in winter.
    seasonal_gain = 0.85 + 0.25 * np.sin(2 * np.pi * (dayofyear - 80) / 365.0)
    shape *= seasonal_gain

    return clip(shape, 0.0, 1.25)


def wind_shape(hour: np.ndarray, dayofyear: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Construct a wind capacity factor with seasonal, intraday, and random
    components.
    """
    seasonal = 0.48 + 0.12 * np.cos(2 * np.pi * (dayofyear - 15) / 365.0)
    diurnal = 0.06 * np.sin(2 * np.pi * (hour + 3) / 24.0) + 0.04 * np.cos(
        2 * np.pi * hour / 12.0
    )
    noise = rng.normal(0, 0.08, size=len(hour))
    cf = seasonal + diurnal + noise
    return clip(cf, 0.03, 0.92)


def load_shape(
    hour: np.ndarray,
    dayofyear: np.ndarray,
    dayofweek: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Construct system load with winter and summer effects, weekday uplift,
    and morning and evening peaks.
    """
    # Cooling and heating effects.
    annual = (
        1.0
        + 0.10 * np.cos(2 * np.pi * (dayofyear - 15) / 365.0)   # Winter uplift
        + 0.08 * np.cos(2 * np.pi * (dayofyear - 200) / 365.0)  # Summer uplift
    )

    morning_peak = np.exp(-((hour - 10) / 3.0) ** 2)
    evening_peak = 1.35 * np.exp(-((hour - 19) / 3.2) ** 2)
    night_valley = 0.25 * np.exp(-((hour - 4) / 2.8) ** 2)
    diurnal = 0.82 + 0.12 * morning_peak + 0.18 * evening_peak - 0.08 * night_valley

    weekday_factor = np.where(dayofweek < 5, 1.03, 0.95)
    noise = rng.normal(0, 0.015, size=len(hour))

    shape = annual * diurnal * weekday_factor + noise
    return clip(shape, 0.72, 1.35)


def scenario_adjustment(
    scenario_id: int,
    wind_cf: np.ndarray,
    solar_cf: np.ndarray,
    load_mw: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the predefined scenario adjustments.
    """
    wind = wind_cf.copy()
    solar = solar_cf.copy()
    load = load_mw.copy()

    if scenario_id == 1:
        # Baseline scenario with small perturbations.
        wind *= 1.00 + rng.normal(0, 0.01, len(wind))
        solar *= 1.00 + rng.normal(0, 0.01, len(solar))
        load *= 1.00 + rng.normal(0, 0.005, len(load))

    elif scenario_id == 2:
        # High-wind scenario.
        wind *= 1.18 + rng.normal(0, 0.015, len(wind))
        solar *= 0.98 + rng.normal(0, 0.01, len(solar))
        load *= 0.99 + rng.normal(0, 0.006, len(load))

    elif scenario_id == 3:
        # High-solar scenario.
        wind *= 0.97 + rng.normal(0, 0.015, len(wind))
        solar *= 1.20 + rng.normal(0, 0.02, len(solar))
        load *= 1.00 + rng.normal(0, 0.006, len(load))

    elif scenario_id == 4:
        # High-load, high-price scenario.
        wind *= 0.94 + rng.normal(0, 0.02, len(wind))
        solar *= 0.95 + rng.normal(0, 0.02, len(solar))
        load *= 1.12 + rng.normal(0, 0.008, len(load))

    elif scenario_id == 5:
        # Low-output-risk scenario.
        wind *= 0.78 + rng.normal(0, 0.02, len(wind))
        solar *= 0.82 + rng.normal(0, 0.02, len(solar))
        load *= 1.05 + rng.normal(0, 0.008, len(load))

    wind = clip(wind, 0.0, 0.98)
    solar = clip(solar, 0.0, 1.10)
    load = clip(load, 0.0, None)

    return wind, solar, load


def price_model(
    load_mw: np.ndarray,
    wind_mw: np.ndarray,
    solar_mw: np.ndarray,
    hour: np.ndarray,
    dayofweek: np.ndarray,
    rng: np.random.Generator,
    scenario_id: int,
) -> np.ndarray:
    """
    Construct spot prices as a function of net load, time of day, weekday,
    renewable output, and scenario-specific premiums.
    """
    renewable = wind_mw + solar_mw
    net_load = load_mw - renewable

    evening_peak = np.exp(-((hour - 19) / 2.8) ** 2)
    morning_peak = np.exp(-((hour - 10) / 3.0) ** 2)
    valley = np.exp(-((hour - 3) / 2.5) ** 2)

    price = (
        180
        + 0.32 * net_load
        + 28 * evening_peak
        + 10 * morning_peak
        - 12 * valley
        - 0.03 * renewable
    )

    # Weekend prices are slightly lower.
    price *= np.where(dayofweek < 5, 1.00, 0.94)

    # Scenario-specific price premium.
    if scenario_id == 4:
        price *= 1.18
    elif scenario_id == 5:
        price *= 1.12
    elif scenario_id == 2:
        price *= 0.96
    elif scenario_id == 3:
        price *= 0.95

    # Add random variation and occasional price spikes.
    noise = rng.normal(0, 12, size=len(price))
    spike_flag = rng.random(len(price)) < 0.015
    spike_value = rng.uniform(50, 160, size=len(price)) * spike_flag

    price = price + noise + spike_value

    # Add occasional low or negative prices.
    oversupply_flag = (renewable > 0.95 * load_mw) & (rng.random(len(price)) < 0.06)
    price[oversupply_flag] -= rng.uniform(80, 180, size=oversupply_flag.sum())

    return clip(price, -80, 1200)


def day_ahead_forecast(actual: np.ndarray, rng: np.random.Generator, error_std: float) -> np.ndarray:
    """
    Construct a day-ahead forecast or submitted bid.
    """
    forecast = actual * (1 + rng.normal(0, error_std, len(actual)))
    return clip(forecast, 0.0, None)


def build_single_scenario(
    scenario_id: int,
    base_df: pd.DataFrame,
    cfg: CapacityConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate one complete scenario.
    """
    df = base_df.copy()

    hour = df["hour"].to_numpy()
    dayofyear = df["dayofyear"].to_numpy()
    dayofweek = df["dayofweek"].to_numpy()

    # Base profiles.
    wind_cf_base = wind_shape(hour, dayofyear, rng)
    solar_cf_base = solar_shape(hour, dayofyear)
    load_shape_base = load_shape(hour, dayofyear, dayofweek, rng)

    load_mw_base = cfg.load_base_mw * load_shape_base

    # Scenario adjustment.
    wind_cf, solar_cf, load_mw = scenario_adjustment(
        scenario_id, wind_cf_base, solar_cf_base, load_mw_base, rng
    )

    # Realized output.
    wind_actual_mw = cfg.wind_capacity_mw * wind_cf
    solar_actual_mw = cfg.solar_capacity_mw * solar_cf

    # Day-ahead forecast and bid.
    wind_da_bid_mw = day_ahead_forecast(wind_actual_mw, rng, error_std=0.10)
    solar_da_bid_mw = day_ahead_forecast(solar_actual_mw, rng, error_std=0.08)

    total_renew_actual_mw = wind_actual_mw + solar_actual_mw
    total_renew_da_bid_mw = wind_da_bid_mw + solar_da_bid_mw

    # Real-time price.
    spot_price_yuan_per_mwh = price_model(
        load_mw=load_mw,
        wind_mw=wind_actual_mw,
        solar_mw=solar_actual_mw,
        hour=hour,
        dayofweek=dayofweek,
        rng=rng,
        scenario_id=scenario_id,
    )

    # Deviations and settlement penalties.
    deviation_mw = total_renew_actual_mw - total_renew_da_bid_mw
    abs_deviation_mw = np.abs(deviation_mw)

    # Deviation penalties are more sensitive during high-price periods.
    penalty_coeff = 18 + 0.06 * np.maximum(spot_price_yuan_per_mwh, 0)
    deviation_penalty_yuan = abs_deviation_mw * penalty_coeff

    # Curtailment occurs when renewable output is high and load is low.
    curtailment_flag = (total_renew_actual_mw > 0.92 * load_mw) & (spot_price_yuan_per_mwh < 120)
    renewable_curtailment_mw = np.where(
        curtailment_flag,
        np.minimum(total_renew_actual_mw - 0.92 * load_mw, total_renew_actual_mw * 0.12),
        0.0,
    )
    renewable_curtailment_mw = clip(renewable_curtailment_mw, 0.0, None)

    delivered_energy_mw = np.maximum(total_renew_actual_mw - renewable_curtailment_mw, 0.0)

    # Revenue.
    market_revenue_yuan = delivered_energy_mw * spot_price_yuan_per_mwh
    net_revenue_yuan = market_revenue_yuan - deviation_penalty_yuan

    # Shortfall risk increases under high net load.
    net_load_mw = load_mw - delivered_energy_mw
    supply_pressure_index = net_load_mw / np.maximum(load_mw, 1e-6)

    # Assemble output fields.
    df["scenario_id"] = scenario_id
    df["scenario_name"] = SCENARIO_NAMES[scenario_id]
    df["scenario_probability"] = SCENARIO_PROB[scenario_id]

    df["wind_capacity_mw"] = cfg.wind_capacity_mw
    df["solar_capacity_mw"] = cfg.solar_capacity_mw

    df["wind_actual_mw"] = safe_round(wind_actual_mw)
    df["solar_actual_mw"] = safe_round(solar_actual_mw)
    df["total_renewable_actual_mw"] = safe_round(total_renew_actual_mw)

    df["wind_day_ahead_bid_mw"] = safe_round(wind_da_bid_mw)
    df["solar_day_ahead_bid_mw"] = safe_round(solar_da_bid_mw)
    df["total_day_ahead_bid_mw"] = safe_round(total_renew_da_bid_mw)

    df["system_load_mw"] = safe_round(load_mw)
    df["net_load_mw"] = safe_round(net_load_mw)

    df["spot_price_yuan_per_mwh"] = safe_round(spot_price_yuan_per_mwh)
    df["deviation_mw"] = safe_round(deviation_mw)
    df["abs_deviation_mw"] = safe_round(abs_deviation_mw)
    df["deviation_penalty_yuan"] = safe_round(deviation_penalty_yuan, 2)

    df["renewable_curtailment_mw"] = safe_round(renewable_curtailment_mw)
    df["delivered_energy_mw"] = safe_round(delivered_energy_mw)

    df["market_revenue_yuan"] = safe_round(market_revenue_yuan, 2)
    df["net_revenue_yuan"] = safe_round(net_revenue_yuan, 2)

    df["supply_pressure_index"] = safe_round(supply_pressure_index)
    df["is_price_spike"] = (df["spot_price_yuan_per_mwh"] >= 500).astype(int)
    df["is_negative_price"] = (df["spot_price_yuan_per_mwh"] < 0).astype(int)
    df["is_curtailment"] = (df["renewable_curtailment_mw"] > 0).astype(int)

    # Risk levels used for grouped summaries.
    risk_score = (
        0.35 * (df["abs_deviation_mw"] / (df["total_day_ahead_bid_mw"] + 1e-6))
        + 0.35 * (df["supply_pressure_index"])
        + 0.30 * (np.maximum(df["spot_price_yuan_per_mwh"], 0) / 1000.0)
    )
    df["risk_score"] = safe_round(risk_score)

    def risk_level(x: float) -> str:
        if x < 0.18:
            return "Low"
        if x < 0.35:
            return "Medium"
        return "High"

    df["risk_level"] = df["risk_score"].map(risk_level)

    return df


def generate_dataset() -> pd.DataFrame:
    """Generate the complete source dataset."""
    rng = np.random.default_rng(SEED)
    cfg = CapacityConfig()

    # Hourly time index.
    time_index = pd.date_range(start=START_TIME, end=END_TIME, freq=FREQ)
    base_df = pd.DataFrame({"timestamp": time_index})
    base_df = build_time_features(base_df)

    scenario_frames: list[pd.DataFrame] = []
    for scenario_id in sorted(SCENARIO_NAMES.keys()):
        scenario_rng = np.random.default_rng(SEED + scenario_id * 100)
        df_s = build_single_scenario(scenario_id, base_df, cfg, scenario_rng)
        scenario_frames.append(df_s)

    df_all = pd.concat(scenario_frames, ignore_index=True)

    # Apply the final column order.
    ordered_cols = [
        "timestamp",
        "year", "month", "day", "hour", "dayofweek", "dayofyear", "is_weekend", "season",
        "scenario_id", "scenario_name", "scenario_probability",
        "wind_capacity_mw", "solar_capacity_mw",
        "wind_actual_mw", "solar_actual_mw", "total_renewable_actual_mw",
        "wind_day_ahead_bid_mw", "solar_day_ahead_bid_mw", "total_day_ahead_bid_mw",
        "system_load_mw", "net_load_mw",
        "spot_price_yuan_per_mwh",
        "deviation_mw", "abs_deviation_mw", "deviation_penalty_yuan",
        "renewable_curtailment_mw", "delivered_energy_mw",
        "market_revenue_yuan", "net_revenue_yuan",
        "supply_pressure_index", "risk_score", "risk_level",
        "is_price_spike", "is_negative_price", "is_curtailment",
    ]
    df_all = df_all[ordered_cols]

    return df_all


def save_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save the source dataset."""
    output_path = Path(output_path)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def print_summary(df: pd.DataFrame) -> None:
    """Print a concise dataset summary."""
    print("=" * 70)
    print("Dataset generation completed")
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Period: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Number of scenarios: {df['scenario_id'].nunique()}")
    print(f"Scenario names: {sorted(df['scenario_name'].unique())}")
    print("-" * 70)

    summary = df.groupby("scenario_name").agg(
        rows=("timestamp", "count"),
        avg_wind=("wind_actual_mw", "mean"),
        avg_solar=("solar_actual_mw", "mean"),
        avg_load=("system_load_mw", "mean"),
        avg_price=("spot_price_yuan_per_mwh", "mean"),
        avg_net_revenue=("net_revenue_yuan", "mean"),
        avg_risk=("risk_score", "mean"),
    )
    print(summary.round(3))
    print("-" * 70)

    print("Overall statistics:")
    print(df[[
        "wind_actual_mw",
        "solar_actual_mw",
        "system_load_mw",
        "spot_price_yuan_per_mwh",
        "net_revenue_yuan",
        "risk_score",
    ]].describe().round(3))
    print("=" * 70)


def main() -> None:
    df = generate_dataset()
    save_dataset(df, OUTPUT_FILE)
    print_summary(df)
    print(f"\nSaved to: {Path(OUTPUT_FILE).resolve()}")


if __name__ == "__main__":
    main()
