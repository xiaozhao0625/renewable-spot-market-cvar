from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    training_scenarios: int = 200
    retained_scenarios: int = 5
    out_of_sample_scenarios: int = 1000
    alpha: float = 0.95
    risk_aversion: float = 0.20
    bid_grid_step_mw: float = 1.0
    random_seed_training: int = 2026092601
    random_seed_evaluation: int = 2026092602
    renewable_error_std: float = 0.16
    load_error_std: float = 0.035
    price_error_std: float = 0.10
    ar1_coefficient: float = 0.72


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
SOURCE_DATA = DATA_DIR / "multi_scenario_power_market_dataset.csv"


def ensure_directories() -> None:
    for directory in (DATA_DIR, RESULT_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _svg_polyline(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    x_scaled = left + (x - x_min) / max(x_max - x_min, 1e-12) * width
    y_scaled = top + height - (y - y_min) / max(y_max - y_min, 1e-12) * height
    return " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in zip(x_scaled, y_scaled))


def write_svg_line_chart(
    path: Path,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    x_label: str,
    y_label: str,
    x_ticks: list[float] | None = None,
) -> None:
    canvas_w, canvas_h = 1000, 580
    left, top, plot_w, plot_h = 145, 55, 790, 420
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    all_x = np.concatenate([item[1] for item in series])
    all_y = np.concatenate([item[2] for item in series])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    y_pad = 0.06 * max(y_max - y_min, 1.0)
    y_min -= y_pad
    y_max += y_pad

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Times New Roman" font-size="18" fill="#000">',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#000" stroke-width="1.4"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#000" stroke-width="1.4"/>',
    ]
    y_range = y_max - y_min

    def format_y_tick(value: float) -> str:
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.1f}k"
        if y_range < 0.5:
            return f"{value:.3f}"
        return f"{value:.2f}"

    for fraction in np.linspace(0.0, 1.0, 6):
        y = top + plot_h - fraction * plot_h
        value = y_min + fraction * (y_max - y_min)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 6:.2f}" text-anchor="end">{format_y_tick(value)}</text>')
    if x_ticks is None:
        x_ticks = list(np.linspace(x_min, x_max, 6))
    for value in x_ticks:
        x = left + (value - x_min) / max(x_max - x_min, 1e-12) * plot_w
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="#000"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 29}" text-anchor="middle">{value:g}</text>')
    for index, (name, x_values, y_values) in enumerate(series):
        points = _svg_polyline(x_values, y_values, x_min, x_max, y_min, y_max, left, top, plot_w, plot_h)
        color = colors[index % len(colors)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        legend_x = left + 18 + (index % 2) * 385
        legend_y = 22 + (index // 2) * 24
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 38}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 48}" y="{legend_y + 6}">{name}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{canvas_h - 25}" text-anchor="middle">{x_label}</text>')
    parts.append(f'<text x="28" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 28 {top + plot_h / 2})">{y_label}</text>')
    parts.append("</g></svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_path_ensemble_svg(
    path: Path,
    background_paths: np.ndarray,
    highlighted_paths: list[tuple[str, np.ndarray]],
    y_label: str,
) -> None:
    canvas_w, canvas_h = 1000, 560
    left, top, plot_w, plot_h = 115, 45, 820, 420
    x_values = np.arange(background_paths.shape[1], dtype=float)
    all_values = [background_paths.ravel()] + [values.ravel() for _, values in highlighted_paths]
    y_min = float(min(np.min(values) for values in all_values))
    y_max = float(max(np.max(values) for values in all_values))
    padding = 0.06 * max(y_max - y_min, 1.0)
    y_min -= padding
    y_max += padding
    colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#000000"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}" viewBox="0 0 {canvas_w} {canvas_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Times New Roman" font-size="18" fill="#000">',
    ]
    for fraction in np.linspace(0.0, 1.0, 6):
        y = top + plot_h - fraction * plot_h
        value = y_min + fraction * (y_max - y_min)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e1e1e1"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 6:.2f}" text-anchor="end">{value:.0f}</text>')
    for hour in (0, 4, 8, 12, 16, 20, 23):
        x = left + hour / 23.0 * plot_w
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 29}" text-anchor="middle">{hour}</text>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#000" stroke-width="1.4"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#000" stroke-width="1.4"/>')
    for values in background_paths:
        points = _svg_polyline(x_values, values, 0.0, 23.0, y_min, y_max, left, top, plot_w, plot_h)
        parts.append(f'<polyline points="{points}" fill="none" stroke="#bfbfbf" stroke-opacity="0.22" stroke-width="1"/>')
    for index, (name, values) in enumerate(highlighted_paths):
        color = colors[index % len(colors)]
        points = _svg_polyline(x_values, values, 0.0, 23.0, y_min, y_max, left, top, plot_w, plot_h)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        legend_x = left + 10 + (index % 3) * 275
        legend_y = 20 + (index // 3) * 23
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 34}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 42}" y="{legend_y + 6}">{name}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{canvas_h - 24}" text-anchor="middle">Hour</text>')
    parts.append(f'<text x="27" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 27 {top + plot_h / 2})">{y_label}</text>')
    parts.append("</g></svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def ar1_errors(
    rng: np.random.Generator,
    n_paths: int,
    n_hours: int,
    std: float,
    phi: float,
) -> np.ndarray:
    innovations = rng.normal(0.0, std * np.sqrt(1.0 - phi**2), size=(n_paths, n_hours))
    errors = np.empty_like(innovations)
    errors[:, 0] = rng.normal(0.0, std, size=n_paths)
    for hour in range(1, n_hours):
        errors[:, hour] = phi * errors[:, hour - 1] + innovations[:, hour]
    return errors


def generate_daily_paths(
    day: pd.DataFrame,
    n_paths: int,
    rng: np.random.Generator,
    cfg: Config,
) -> dict[str, np.ndarray]:
    renewable_forecast = day["total_day_ahead_bid_mw"].to_numpy(float)
    load_forecast = day["system_load_mw"].to_numpy(float)
    price_forecast = day["spot_price_yuan_per_mwh"].to_numpy(float)
    n_hours = len(day)

    renewable_error = ar1_errors(
        rng, n_paths, n_hours, cfg.renewable_error_std, cfg.ar1_coefficient
    )
    load_error = ar1_errors(rng, n_paths, n_hours, cfg.load_error_std, 0.60)
    price_error = ar1_errors(rng, n_paths, n_hours, cfg.price_error_std, 0.50)

    available = np.clip(
        renewable_forecast[None, :] * (1.0 + renewable_error), 0.0, 580.0
    )
    load = np.clip(load_forecast[None, :] * (1.0 + load_error), 1.0, None)

    # Renewable delivery cannot exceed either available power or the scenario-specific
    # market absorption limit. The difference is explicit renewable curtailment.
    absorption_limit = 0.92 * load
    delivered = np.minimum(available, absorption_limit)
    curtailment = np.maximum(available - delivered, 0.0)

    price = (
        price_forecast[None, :] * (1.0 + price_error)
        + 0.22 * (load - load_forecast[None, :])
        - 0.09 * (delivered - renewable_forecast[None, :])
    )
    price = np.clip(price, -80.0, 1200.0)

    return {
        "available": available,
        "delivered": delivered,
        "curtailment": curtailment,
        "load": load,
        "price": price,
    }


def standardized_features(paths: dict[str, np.ndarray]) -> np.ndarray:
    matrices = [paths["delivered"], paths["load"], paths["price"]]
    features = []
    for matrix in matrices:
        scale = np.std(matrix, axis=0, ddof=1)
        scale = np.where(scale < 1e-8, 1.0, scale)
        features.append((matrix - np.mean(matrix, axis=0)) / scale)
    return np.concatenate(features, axis=1)


def fast_forward_selection(features: np.ndarray, retained: int) -> tuple[np.ndarray, np.ndarray, float]:
    n = features.shape[0]
    weights = np.full(n, 1.0 / n)
    squared_norm = np.sum(features**2, axis=1)
    distances_sq = np.maximum(
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * features @ features.T,
        0.0,
    )
    distances = np.sqrt(distances_sq)

    selected: list[int] = []
    current = np.full(n, np.inf)
    available = np.ones(n, dtype=bool)
    for _ in range(retained):
        candidates = np.flatnonzero(available)
        objectives = np.sum(
            weights[:, None] * np.minimum(current[:, None], distances[:, candidates]),
            axis=0,
        )
        chosen = int(candidates[int(np.argmin(objectives))])
        selected.append(chosen)
        available[chosen] = False
        current = np.minimum(current, distances[:, chosen])

    selected_array = np.asarray(selected, dtype=int)
    assignment = np.argmin(distances[:, selected_array], axis=1)
    probabilities = np.bincount(assignment, minlength=retained).astype(float) / n
    reduction_distance = float(np.sum(weights * current))
    return selected_array, probabilities, reduction_distance


def profit_components(
    bid: np.ndarray,
    delivered: np.ndarray,
    price: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    under_delivery = np.maximum(bid - delivered, 0.0)
    over_delivery = np.maximum(delivered - bid, 0.0)
    shortage_rate = 20.0 + 0.18 * np.maximum(price, 0.0)
    surplus_rate = 8.0 + 0.07 * np.maximum(price, 0.0)
    penalty = shortage_rate * under_delivery + surplus_rate * over_delivery
    gross_revenue = price * delivered
    profit = gross_revenue - penalty
    return profit, penalty, np.abs(delivered - bid)


def weighted_cvar(loss: np.ndarray, probabilities: np.ndarray, alpha: float) -> np.ndarray:
    # loss shape: grid x retained scenarios
    order = np.argsort(loss, axis=1)
    sorted_loss = np.take_along_axis(loss, order, axis=1)
    weights = np.broadcast_to(probabilities, loss.shape)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    cumulative = np.cumsum(sorted_weights, axis=1)
    quantile_index = np.argmax(cumulative >= alpha, axis=1)
    eta = sorted_loss[np.arange(loss.shape[0]), quantile_index]
    excess = np.maximum(loss - eta[:, None], 0.0)
    return eta + np.sum(probabilities[None, :] * excess, axis=1) / (1.0 - alpha)


def optimize_hour_bids(
    delivered: np.ndarray,
    price: np.ndarray,
    probabilities: np.ndarray,
    forecast_bid: float,
    forecast_price: float,
    cfg: Config,
    risk_aversion_values: tuple[float, ...],
) -> dict[float, float]:
    upper = max(float(np.max(delivered)), forecast_bid) + 3.0
    grid = np.arange(0.0, upper + cfg.bid_grid_step_mw, cfg.bid_grid_step_mw)
    bid = grid[:, None]
    profit, _, _ = profit_components(bid, delivered[None, :], price[None, :])
    expected_profit = np.sum(profit * probabilities[None, :], axis=1)

    target = max(forecast_price, 0.0) * forecast_bid
    downside_shortfall = np.maximum(target - profit, 0.0)
    cvar = weighted_cvar(downside_shortfall, probabilities, cfg.alpha)

    return {
        value: float(grid[int(np.argmax(expected_profit - value * cvar))])
        for value in risk_aversion_values
    }


def summarize_strategy(
    name: str,
    profit: np.ndarray,
    penalty: np.ndarray,
    abs_deviation: np.ndarray,
    shortfall: np.ndarray,
    alpha: float,
) -> dict[str, float | str]:
    flat_profit = profit.ravel()
    flat_penalty = penalty.ravel()
    flat_deviation = abs_deviation.ravel()
    flat_shortfall = shortfall.ravel()
    var = float(np.quantile(flat_shortfall, alpha))
    tail = flat_shortfall[flat_shortfall >= var]
    cvar = float(np.mean(tail)) if len(tail) else var
    lower_profit_quantile = float(np.quantile(flat_profit, 1.0 - alpha))
    lower_tail_profit = flat_profit[flat_profit <= lower_profit_quantile]
    return {
        "Strategy": name,
        "Average hourly net revenue (Yuan)": float(np.mean(flat_profit)),
        "Expected annual net revenue (Yuan)": float(np.mean(flat_profit) * 8760.0),
        "Revenue standard deviation (Yuan)": float(np.std(flat_profit, ddof=1)),
        "Average deviation penalty (Yuan)": float(np.mean(flat_penalty)),
        "Mean absolute deviation (MW)": float(np.mean(flat_deviation)),
        "VaR95 of downside shortfall (Yuan)": var,
        "CVaR95 of downside shortfall (Yuan)": cvar,
        "Mean worst-5% hourly profit (Yuan)": float(np.mean(lower_tail_profit)),
    }


def main() -> None:
    started = time.perf_counter()
    cfg = Config()
    ensure_directories()

    source = pd.read_csv(SOURCE_DATA, parse_dates=["timestamp"])
    base = source[source["scenario_id"] == 1].copy().sort_values("timestamp")
    if len(base) != 8760:
        raise ValueError(f"Expected 8760 baseline hours, found {len(base)}")

    training_rng = np.random.default_rng(cfg.random_seed_training)
    evaluation_rng = np.random.default_rng(cfg.random_seed_evaluation)

    hourly_rows: list[dict[str, float | int | str]] = []
    reduction_rows: list[dict[str, float | int | str]] = []
    computation_rows: list[dict[str, float | int | str]] = []
    monthly_accumulator: list[pd.DataFrame] = []

    strategy_store: dict[str, dict[str, list[np.ndarray]]] = {
        key: {metric: [] for metric in ("profit", "penalty", "deviation", "shortfall")}
        for key in ("Deterministic forecast baseline", "Risk-neutral stochastic", "Proposed CVaR")
    }
    sensitivity_lambdas = (0.0, 0.05, 0.10, cfg.risk_aversion, 0.40, 0.80)
    sensitivity_profit: dict[float, list[np.ndarray]] = {value: [] for value in sensitivity_lambdas}
    sensitivity_shortfall: dict[float, list[np.ndarray]] = {value: [] for value in sensitivity_lambdas}

    for date, day in base.groupby(base["timestamp"].dt.date, sort=True):
        day_started = time.perf_counter()
        day = day.reset_index(drop=True)
        if len(day) != 24:
            raise ValueError(f"Date {date} has {len(day)} hours, expected 24")

        training = generate_daily_paths(day, cfg.training_scenarios, training_rng, cfg)
        selected, probabilities, reduction_distance = fast_forward_selection(
            standardized_features(training), cfg.retained_scenarios
        )
        reduced = {key: value[selected] for key, value in training.items()}

        baseline_bid = day["total_day_ahead_bid_mw"].to_numpy(float)
        risk_neutral_bid = np.zeros(24)
        cvar_bid = np.zeros(24)
        sensitivity_bids = {value: np.zeros(24) for value in sensitivity_lambdas}
        for hour in range(24):
            optimized = optimize_hour_bids(
                reduced["delivered"][:, hour],
                reduced["price"][:, hour],
                probabilities,
                baseline_bid[hour],
                float(day.loc[hour, "spot_price_yuan_per_mwh"]),
                cfg,
                sensitivity_lambdas,
            )
            for value, bid_value in optimized.items():
                sensitivity_bids[value][hour] = bid_value
            risk_neutral_bid[hour] = optimized[0.0]
            cvar_bid[hour] = optimized[cfg.risk_aversion]

        evaluation_paths = generate_daily_paths(
            day, cfg.out_of_sample_scenarios, evaluation_rng, cfg
        )
        target = (
            np.maximum(day["spot_price_yuan_per_mwh"].to_numpy(float), 0.0)
            * baseline_bid
        )[None, :]

        day_outputs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for name, bids in (
            ("Deterministic forecast baseline", baseline_bid),
            ("Risk-neutral stochastic", risk_neutral_bid),
            ("Proposed CVaR", cvar_bid),
        ):
            profit, penalty, deviation = profit_components(
                bids[None, :], evaluation_paths["delivered"], evaluation_paths["price"]
            )
            shortfall = np.maximum(target - profit, 0.0)
            day_outputs[name] = (profit, penalty, deviation, shortfall)
            strategy_store[name]["profit"].append(profit)
            strategy_store[name]["penalty"].append(penalty)
            strategy_store[name]["deviation"].append(deviation)
            strategy_store[name]["shortfall"].append(shortfall)

        if len(reduction_rows) == 0:
            write_path_ensemble_svg(
                FIGURE_DIR / "scenario_reduction.svg",
                training["delivered"],
                [(f"Representative {index + 1}", reduced["delivered"][index]) for index in range(cfg.retained_scenarios)],
                "Delivered renewable power (MW)",
            )
            write_path_ensemble_svg(
                FIGURE_DIR / "out_of_sample_example.svg",
                evaluation_paths["delivered"][:100],
                [
                    ("Deterministic bid", baseline_bid),
                    ("Risk-neutral bid", risk_neutral_bid),
                    ("Proposed CVaR bid", cvar_bid),
                ],
                "Power (MW)",
            )

        # Use the first 200 independent evaluation paths for the risk-aversion
        # sensitivity analysis without changing the main 1000-path evaluation.
        sensitivity_slice = slice(0, min(200, cfg.out_of_sample_scenarios))
        for value, bids in sensitivity_bids.items():
            profit, _, _ = profit_components(
                bids[None, :],
                evaluation_paths["delivered"][sensitivity_slice],
                evaluation_paths["price"][sensitivity_slice],
            )
            shortfall = np.maximum(target - profit, 0.0)
            sensitivity_profit[value].append(profit)
            sensitivity_shortfall[value].append(shortfall)

        for hour in range(24):
            row: dict[str, float | int | str] = {
                "timestamp": str(day.loc[hour, "timestamp"]),
                "baseline_bid_mw": baseline_bid[hour],
                "risk_neutral_bid_mw": risk_neutral_bid[hour],
                "cvar_bid_mw": cvar_bid[hour],
                "training_scenarios": cfg.training_scenarios,
                "retained_scenarios": cfg.retained_scenarios,
                "out_of_sample_scenarios": cfg.out_of_sample_scenarios,
            }
            for name, prefix in (
                ("Deterministic forecast baseline", "baseline"),
                ("Risk-neutral stochastic", "risk_neutral"),
                ("Proposed CVaR", "cvar"),
            ):
                profit, penalty, deviation, shortfall = day_outputs[name]
                row[f"{prefix}_oos_mean_profit_yuan"] = float(np.mean(profit[:, hour]))
                row[f"{prefix}_oos_mean_penalty_yuan"] = float(np.mean(penalty[:, hour]))
                row[f"{prefix}_oos_mean_abs_deviation_mw"] = float(np.mean(deviation[:, hour]))
                row[f"{prefix}_oos_mean_shortfall_yuan"] = float(np.mean(shortfall[:, hour]))
            hourly_rows.append(row)

        month_frame = pd.DataFrame(
            {
                "date": pd.Timestamp(date),
                "baseline": np.mean(day_outputs["Deterministic forecast baseline"][0], axis=0),
                "risk_neutral": np.mean(day_outputs["Risk-neutral stochastic"][0], axis=0),
                "cvar": np.mean(day_outputs["Proposed CVaR"][0], axis=0),
            }
        )
        monthly_accumulator.append(month_frame)

        reduction_rows.append(
            {
                "date": str(date),
                "initial_scenarios": cfg.training_scenarios,
                "retained_scenarios": cfg.retained_scenarios,
                "selected_zero_based_indices": ";".join(str(int(i)) for i in selected),
                "retained_probabilities": ";".join(f"{p:.6f}" for p in probabilities),
                "probability_sum": float(np.sum(probabilities)),
                "weighted_reduction_distance": reduction_distance,
            }
        )
        computation_rows.append(
            {
                "date": str(date),
                "wall_time_seconds": time.perf_counter() - day_started,
                "candidate_bid_step_mw": cfg.bid_grid_step_mw,
            }
        )

    metrics = []
    concatenated: dict[str, dict[str, np.ndarray]] = {}
    for name, store in strategy_store.items():
        concatenated[name] = {
            metric: np.concatenate(arrays, axis=1) for metric, arrays in store.items()
        }
        metrics.append(
            summarize_strategy(
                name,
                concatenated[name]["profit"],
                concatenated[name]["penalty"],
                concatenated[name]["deviation"],
                concatenated[name]["shortfall"],
                cfg.alpha,
            )
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(RESULT_DIR / "out_of_sample_performance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(hourly_rows).to_csv(
        DATA_DIR / "hourly_out_of_sample_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(reduction_rows).to_csv(
        RESULT_DIR / "scenario_reduction_audit.csv", index=False, encoding="utf-8-sig"
    )
    computation_df = pd.DataFrame(computation_rows)
    computation_df.to_csv(
        RESULT_DIR / "daily_computation_times.csv", index=False, encoding="utf-8-sig"
    )

    monthly = pd.concat(monthly_accumulator, ignore_index=True)
    monthly["month"] = monthly["date"].dt.month
    monthly_result = monthly.groupby("month", as_index=False)[
        ["baseline", "risk_neutral", "cvar"]
    ].sum()
    monthly_result.to_csv(
        RESULT_DIR / "monthly_out_of_sample_revenue.csv", index=False, encoding="utf-8-sig"
    )

    sensitivity_rows = []
    for value in sensitivity_lambdas:
        profit = np.concatenate(sensitivity_profit[value], axis=1).ravel()
        shortfall = np.concatenate(sensitivity_shortfall[value], axis=1).ravel()
        var = float(np.quantile(shortfall, cfg.alpha))
        tail = shortfall[shortfall >= var]
        sensitivity_rows.append(
            {
                "Risk aversion coefficient": value,
                "Average hourly net revenue (Yuan)": float(np.mean(profit)),
                "CVaR95 of downside shortfall (Yuan)": float(np.mean(tail)),
            }
        )
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(
        RESULT_DIR / "risk_aversion_sensitivity.csv", index=False, encoding="utf-8-sig"
    )

    write_svg_line_chart(
        FIGURE_DIR / "monthly_out_of_sample_revenue.svg",
        [
            ("Deterministic baseline", monthly_result["month"].to_numpy(float), monthly_result["baseline"].to_numpy(float)),
            ("Risk-neutral stochastic", monthly_result["month"].to_numpy(float), monthly_result["risk_neutral"].to_numpy(float)),
            ("Proposed CVaR", monthly_result["month"].to_numpy(float), monthly_result["cvar"].to_numpy(float)),
        ],
        "Month",
        "Out-of-sample monthly net revenue (Yuan)",
        list(range(1, 13)),
    )

    max_shortfall = max(
        np.quantile(concatenated[name]["shortfall"], 0.995) for name in concatenated
    )
    bins = np.linspace(0.0, max_shortfall, 100)
    risk_series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, label in (
        ("Deterministic forecast baseline", "Deterministic baseline"),
        ("Risk-neutral stochastic", "Risk-neutral stochastic"),
        ("Proposed CVaR", "Proposed CVaR"),
    ):
        values = concatenated[name]["shortfall"].ravel()
        hist, edges = np.histogram(values, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        risk_series.append((label, centers, np.log10(np.maximum(hist, 1e-12))))
    write_svg_line_chart(
        FIGURE_DIR / "downside_risk.svg",
        risk_series,
        "Downside revenue shortfall (Yuan)",
        "Log10 density",
    )

    # Normalize the two quantities to the risk-neutral case so that both can be
    # compared on one axis without a misleading dual-axis presentation.
    normalized_revenue = (
        sensitivity_df["Average hourly net revenue (Yuan)"]
        / sensitivity_df.loc[0, "Average hourly net revenue (Yuan)"]
    )
    normalized_cvar = (
        sensitivity_df["CVaR95 of downside shortfall (Yuan)"]
        / sensitivity_df.loc[0, "CVaR95 of downside shortfall (Yuan)"]
    )
    write_svg_line_chart(
        FIGURE_DIR / "risk_aversion_sensitivity.svg",
        [
            ("Normalized revenue", sensitivity_df["Risk aversion coefficient"].to_numpy(float), normalized_revenue.to_numpy(float)),
            ("Normalized CVaR95", sensitivity_df["Risk aversion coefficient"].to_numpy(float), normalized_cvar.to_numpy(float)),
        ],
        "Risk aversion coefficient",
        "Normalized out-of-sample metric",
        list(sensitivity_lambdas),
    )

    total_seconds = time.perf_counter() - started
    environment = {
        "config": asdict(cfg),
        "source_data": SOURCE_DATA.relative_to(ROOT).as_posix(),
        "source_rows": int(len(source)),
        "baseline_rows": int(len(base)),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "total_wall_time_seconds": total_seconds,
        "mean_daily_wall_time_seconds": float(computation_df["wall_time_seconds"].mean()),
        "max_daily_wall_time_seconds": float(computation_df["wall_time_seconds"].max()),
    }
    (RESULT_DIR / "environment_and_runtime.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics_df.to_string(index=False))
    print(json.dumps(environment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
