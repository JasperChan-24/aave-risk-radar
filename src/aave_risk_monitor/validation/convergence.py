"""Time-grid, Monte Carlo sample-size, and calibration-window validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from aave_risk_monitor.simulation import (
    PortfolioRevaluation,
    calibrate_log_returns,
)

from ._common import (
    bool_checks,
    ensure_strictly_increasing,
    relative_difference,
    simulation_risk_point,
)


def time_step_convergence(
    *,
    spot_prices_usd: ArrayLike,
    price_history: pd.DataFrame,
    revalue_fn: PortfolioRevaluation,
    step_sizes_days: Sequence[float] = (1.0, 0.5, 0.25),
    horizon_days: float = 30.0,
    n_paths: int = 20_000,
    seed: int = 20_260_721,
    confidence_level: float = 0.95,
    probability_tolerance: float = 0.025,
    var_relative_tolerance: float = 0.05,
    es_relative_tolerance: float = 0.075,
) -> dict[str, Any]:
    """Compare the two finest monitoring grids over one fixed horizon.

    This is a discrete first-passage convergence experiment, not proof of a
    continuous-time limit.  The finest configured grid is the reference.
    """

    steps = tuple(float(value) for value in step_sizes_days)
    if len(steps) < 3:
        raise ValueError("step_sizes_days must include at least three grids")
    if any(not np.isfinite(value) or value <= 0.0 for value in steps):
        raise ValueError("step sizes must be positive and finite")
    if any(right >= left for left, right in zip(steps, steps[1:], strict=False)):
        raise ValueError("step sizes must be strictly decreasing")
    if not np.isfinite(horizon_days) or horizon_days <= 0.0:
        raise ValueError("horizon_days must be positive and finite")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    calibration = calibrate_log_returns(price_history)
    rows: list[dict[str, Any]] = []
    for step_size in steps:
        horizon_steps = round(horizon_days / step_size)
        if not np.isclose(horizon_steps * step_size, horizon_days, atol=1e-12):
            raise ValueError("each step size must divide the horizon exactly")
        point = simulation_risk_point(
            spot_prices_usd=spot_prices_usd,
            calibration=calibration,
            revalue_fn=revalue_fn,
            n_paths=n_paths,
            horizon_steps=horizon_steps,
            step_size_days=step_size,
            seed=seed,
            confidence_level=confidence_level,
        )
        rows.append(
            {
                "step_size_days": step_size,
                "horizon_steps": horizon_steps,
                **point.to_dict(),
            }
        )

    candidate, reference = rows[-2], rows[-1]
    differences = {
        "liquidation_probability_absolute": abs(
            candidate["liquidation_probability"] - reference["liquidation_probability"]
        ),
        "value_at_risk_relative": relative_difference(
            candidate["value_at_risk_usd"], reference["value_at_risk_usd"]
        ),
        "expected_shortfall_relative": relative_difference(
            candidate["expected_shortfall_usd"],
            reference["expected_shortfall_usd"],
        ),
    }
    thresholds = {
        "liquidation_probability_absolute": probability_tolerance,
        "value_at_risk_relative": var_relative_tolerance,
        "expected_shortfall_relative": es_relative_tolerance,
    }
    status = bool_checks({name: differences[name] <= limit for name, limit in thresholds.items()})
    return {
        "method": "production correlated-GBM simulator; finest grid is reference",
        "horizon_days": horizon_days,
        "n_paths_per_grid": n_paths,
        "seed_per_grid": seed,
        "results": rows,
        "comparison": {
            "candidate_step_size_days": candidate["step_size_days"],
            "reference_step_size_days": reference["step_size_days"],
            "differences": differences,
            "thresholds": thresholds,
            **status,
        },
    }


def path_count_convergence(
    *,
    spot_prices_usd: ArrayLike,
    price_history: pd.DataFrame,
    revalue_fn: PortfolioRevaluation,
    path_counts: Sequence[int] = (2_000, 8_000, 32_000),
    horizon_days: int = 30,
    seed: int = 20_260_722,
    confidence_level: float = 0.95,
    probability_tolerance: float = 0.02,
    var_relative_tolerance: float = 0.05,
    es_relative_tolerance: float = 0.075,
) -> dict[str, Any]:
    """Check sampling stability and Wilson-interval contraction."""

    counts = tuple(int(value) for value in path_counts)
    ensure_strictly_increasing(tuple(float(value) for value in counts), name="path_counts")
    if any(
        float(original) != converted
        for original, converted in zip(path_counts, counts, strict=True)
    ):
        raise ValueError("path counts must be integers")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    calibration = calibrate_log_returns(price_history)
    rows: list[dict[str, Any]] = []
    for n_paths in counts:
        point = simulation_risk_point(
            spot_prices_usd=spot_prices_usd,
            calibration=calibration,
            revalue_fn=revalue_fn,
            n_paths=n_paths,
            horizon_steps=horizon_days,
            step_size_days=1.0,
            seed=seed,
            confidence_level=confidence_level,
        )
        rows.append({"n_paths": n_paths, **point.to_dict()})

    candidate, reference = rows[-2], rows[-1]
    differences = {
        "liquidation_probability_absolute": abs(
            candidate["liquidation_probability"] - reference["liquidation_probability"]
        ),
        "value_at_risk_relative": relative_difference(
            candidate["value_at_risk_usd"], reference["value_at_risk_usd"]
        ),
        "expected_shortfall_relative": relative_difference(
            candidate["expected_shortfall_usd"],
            reference["expected_shortfall_usd"],
        ),
    }
    thresholds = {
        "liquidation_probability_absolute": probability_tolerance,
        "value_at_risk_relative": var_relative_tolerance,
        "expected_shortfall_relative": es_relative_tolerance,
    }
    interval_widths = [row["liquidation_probability_ci_width"] for row in rows]
    checks = {name: differences[name] <= limit for name, limit in thresholds.items()}
    checks["liquidation_probability_ci_contracts"] = all(
        right < left for left, right in zip(interval_widths, interval_widths[1:], strict=False)
    )
    return {
        "method": "nested seeded path prefixes; largest path count is reference",
        "horizon_days": horizon_days,
        "seed_per_run": seed,
        "results": rows,
        "comparison": {
            "candidate_n_paths": candidate["n_paths"],
            "reference_n_paths": reference["n_paths"],
            "differences": differences,
            "thresholds": thresholds,
            **bool_checks(checks),
        },
    }


def calibration_window_convergence(
    *,
    spot_prices_usd: ArrayLike,
    price_history: pd.DataFrame,
    revalue_fn: PortfolioRevaluation,
    calibration_windows_days: Sequence[int] = (90, 180, 365),
    horizon_days: int = 30,
    n_paths: int = 20_000,
    seed: int = 20_260_723,
    confidence_level: float = 0.95,
    annualized_volatility_relative_tolerance: float = 0.30,
    correlation_absolute_tolerance: float = 0.25,
    probability_tolerance: float = 0.10,
    var_relative_tolerance: float = 0.25,
    es_relative_tolerance: float = 0.25,
) -> dict[str, Any]:
    """Measure parameter and output sensitivity to trailing history length."""

    windows = tuple(int(value) for value in calibration_windows_days)
    ensure_strictly_increasing(tuple(float(value) for value in windows), name="windows")
    if any(
        float(original) != converted
        for original, converted in zip(calibration_windows_days, windows, strict=True)
    ):
        raise ValueError("calibration windows must be integers")
    if len(price_history) < windows[-1] + 1:
        raise ValueError("price history is too short for the largest calibration window")
    if horizon_days <= 0 or n_paths <= 0:
        raise ValueError("horizon_days and n_paths must be positive")

    rows: list[dict[str, Any]] = []
    for window in windows:
        calibration = calibrate_log_returns(price_history.iloc[-(window + 1) :])
        point = simulation_risk_point(
            spot_prices_usd=spot_prices_usd,
            calibration=calibration,
            revalue_fn=revalue_fn,
            n_paths=n_paths,
            horizon_steps=horizon_days,
            step_size_days=1.0,
            seed=seed,
            confidence_level=confidence_level,
        )
        rows.append(
            {
                "calibration_window_days": window,
                "observations": calibration.observations,
                "annualized_volatility": calibration.annualized_volatility.tolist(),
                "correlation": calibration.correlation.tolist(),
                "shrinkage": calibration.shrinkage,
                **point.to_dict(),
            }
        )

    candidate, reference = rows[-2], rows[-1]
    candidate_volatility = np.asarray(candidate["annualized_volatility"])
    reference_volatility = np.asarray(reference["annualized_volatility"])
    positive_reference = reference_volatility > 0.0
    volatility_differences = np.zeros_like(reference_volatility)
    volatility_differences[positive_reference] = (
        np.abs(candidate_volatility[positive_reference] - reference_volatility[positive_reference])
        / reference_volatility[positive_reference]
    )
    volatility_differences[~positive_reference] = np.abs(candidate_volatility[~positive_reference])
    correlation_difference = np.max(
        np.abs(np.asarray(candidate["correlation"]) - np.asarray(reference["correlation"]))
    )
    differences = {
        "max_annualized_volatility_relative": float(np.max(volatility_differences, initial=0.0)),
        "max_correlation_absolute": float(correlation_difference),
        "liquidation_probability_absolute": abs(
            candidate["liquidation_probability"] - reference["liquidation_probability"]
        ),
        "value_at_risk_relative": relative_difference(
            candidate["value_at_risk_usd"], reference["value_at_risk_usd"]
        ),
        "expected_shortfall_relative": relative_difference(
            candidate["expected_shortfall_usd"],
            reference["expected_shortfall_usd"],
        ),
    }
    thresholds = {
        "max_annualized_volatility_relative": (annualized_volatility_relative_tolerance),
        "max_correlation_absolute": correlation_absolute_tolerance,
        "liquidation_probability_absolute": probability_tolerance,
        "value_at_risk_relative": var_relative_tolerance,
        "expected_shortfall_relative": es_relative_tolerance,
    }
    return {
        "method": "trailing daily log-return windows; longest window is reference",
        "asset_names": list(calibrate_log_returns(price_history.iloc[-3:]).asset_names),
        "horizon_days": horizon_days,
        "n_paths_per_window": n_paths,
        "seed_per_window": seed,
        "results": rows,
        "comparison": {
            "candidate_window_days": candidate["calibration_window_days"],
            "reference_window_days": reference["calibration_window_days"],
            "differences": differences,
            "thresholds": thresholds,
            **bool_checks({name: differences[name] <= limit for name, limit in thresholds.items()}),
        },
    }
