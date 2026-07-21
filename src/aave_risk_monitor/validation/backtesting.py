"""Rolling, strictly out-of-sample Value-at-Risk backtesting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erfc, log, sqrt

import numpy as np
import pandas as pd

from aave_risk_monitor.simulation import (
    PortfolioRevaluation,
    SimulationConfig,
    calibrate_log_returns,
    simulate_from_calibration,
)

from ._common import coerce_validation_valuation


@dataclass(frozen=True, slots=True)
class CoverageTest:
    statistic: float
    p_value: float
    passed_at_5pct: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _bernoulli_log_likelihood(successes: int, trials: int, probability: float) -> float:
    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    failures = trials - successes
    value = 0.0
    if successes:
        if probability == 0.0:
            return -np.inf
        value += successes * log(probability)
    if failures:
        if probability == 1.0:
            return -np.inf
        value += failures * log(1.0 - probability)
    return value


def kupiec_unconditional_coverage_test(
    *,
    breaches: int,
    observations: int,
    expected_breach_probability: float,
) -> CoverageTest:
    """Kupiec likelihood-ratio test with a chi-square(1) p-value."""

    if observations <= 0:
        raise ValueError("observations must be positive")
    if not 0 <= breaches <= observations:
        raise ValueError("breaches must be between zero and observations")
    if not 0.0 < expected_breach_probability < 1.0:
        raise ValueError("expected breach probability must be between zero and one")
    observed_probability = breaches / observations
    null_log_likelihood = _bernoulli_log_likelihood(
        breaches, observations, expected_breach_probability
    )
    fitted_log_likelihood = _bernoulli_log_likelihood(breaches, observations, observed_probability)
    statistic = max(0.0, -2.0 * (null_log_likelihood - fitted_log_likelihood))
    p_value = erfc(sqrt(statistic / 2.0))
    return CoverageTest(
        statistic=statistic,
        p_value=p_value,
        passed_at_5pct=p_value >= 0.05,
    )


def christoffersen_independence_test(exceedances: np.ndarray) -> CoverageTest | None:
    """Test whether consecutive VaR breaches are first-order independent."""

    values = np.asarray(exceedances, dtype=np.bool_)
    if values.ndim != 1 or values.size < 2:
        return None
    previous = values[:-1]
    current = values[1:]
    n00 = int(np.count_nonzero(~previous & ~current))
    n01 = int(np.count_nonzero(~previous & current))
    n10 = int(np.count_nonzero(previous & ~current))
    n11 = int(np.count_nonzero(previous & current))
    non_breach_origins = n00 + n01
    breach_origins = n10 + n11
    transitions = non_breach_origins + breach_origins
    total_breaches = n01 + n11
    if not non_breach_origins or not breach_origins:
        return None

    common_probability = total_breaches / transitions
    p01 = n01 / non_breach_origins
    p11 = n11 / breach_origins
    null_log_likelihood = _bernoulli_log_likelihood(total_breaches, transitions, common_probability)
    alternative_log_likelihood = _bernoulli_log_likelihood(
        n01, non_breach_origins, p01
    ) + _bernoulli_log_likelihood(n11, breach_origins, p11)
    statistic = max(0.0, -2.0 * (null_log_likelihood - alternative_log_likelihood))
    p_value = erfc(sqrt(statistic / 2.0))
    return CoverageTest(
        statistic=statistic,
        p_value=p_value,
        passed_at_5pct=p_value >= 0.05,
    )


def rolling_var_backtest(
    *,
    price_history: pd.DataFrame,
    revalue_fn: PortfolioRevaluation,
    calibration_window_days: int = 180,
    confidence_level: float = 0.95,
    n_paths_per_forecast: int = 4_000,
    seed: int = 20_260_724,
) -> dict[str, object]:
    """Forecast one-day VaR using only information available at each origin.

    At origin ``t`` the calibration contains returns ending at ``t``.  The
    realized loss from ``t`` to ``t+1`` is never used in that forecast.
    """

    if calibration_window_days < 2:
        raise ValueError("calibration_window_days must be at least 2")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if n_paths_per_forecast <= 0:
        raise ValueError("n_paths_per_forecast must be positive")
    if len(price_history) < calibration_window_days + 2:
        raise ValueError("price history is too short for an out-of-sample forecast")
    if not isinstance(price_history.index, pd.DatetimeIndex):
        raise ValueError("price history must use a DatetimeIndex")
    values = price_history.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("price history must contain finite positive prices")

    forecast_dates: list[str] = []
    var_forecasts: list[float] = []
    actual_losses: list[float] = []
    exceedances: list[bool] = []
    for origin in range(calibration_window_days, len(price_history) - 1):
        calibration_frame = price_history.iloc[origin - calibration_window_days : origin + 1]
        calibration = calibrate_log_returns(calibration_frame)
        result = simulate_from_calibration(
            spot_prices_usd=values[origin],
            calibration=calibration,
            revalue_fn=revalue_fn,
            config=SimulationConfig(
                n_paths=n_paths_per_forecast,
                horizon_steps=1,
                step_size_days=1.0,
                seed=seed + origin,
                chunk_size=min(2_000, n_paths_per_forecast),
                stored_paths=0,
                tail_confidence_levels=(confidence_level,),
                bootstrap_resamples=0,
            ),
        )
        forecast = result.summary.tail_risk[0].value_at_risk_usd.value
        realized_prices = np.stack((values[origin], values[origin + 1]))[None, :, :]
        realized_equity = coerce_validation_valuation(revalue_fn(realized_prices)).net_equity_usd[0]
        actual_loss = float(realized_equity[0] - realized_equity[1])
        exceeded = actual_loss > forecast

        forecast_dates.append(price_history.index[origin + 1].date().isoformat())
        var_forecasts.append(forecast)
        actual_losses.append(actual_loss)
        exceedances.append(exceeded)

    exceedance_array = np.asarray(exceedances, dtype=np.bool_)
    observations = int(exceedance_array.size)
    breaches = int(np.count_nonzero(exceedance_array))
    expected_probability = 1.0 - confidence_level
    kupiec = kupiec_unconditional_coverage_test(
        breaches=breaches,
        observations=observations,
        expected_breach_probability=expected_probability,
    )
    independence = christoffersen_independence_test(exceedance_array)
    forecasts = np.asarray(var_forecasts, dtype=np.float64)
    losses = np.asarray(actual_losses, dtype=np.float64)
    breach_rows = [
        {
            "date": forecast_dates[index],
            "forecast_var_usd": var_forecasts[index],
            "actual_loss_usd": actual_losses[index],
        }
        for index in np.flatnonzero(exceedance_array)
    ]
    forecast_rows = [
        {
            "date": forecast_dates[index],
            "forecast_var_usd": var_forecasts[index],
            "actual_loss_usd": actual_losses[index],
            "breach": exceedances[index],
        }
        for index in range(observations)
    ]
    checks = {
        "kupiec_unconditional_coverage_p_value_at_least_0_05": (kupiec.passed_at_5pct),
        "christoffersen_independence_p_value_at_least_0_05": (
            independence is not None and independence.passed_at_5pct
        ),
    }
    return {
        "method": (
            "rolling one-day production-simulator VaR; calibration ends at forecast "
            "origin; strict breach when actual loss > VaR"
        ),
        "calibration_window_days": calibration_window_days,
        "confidence_level": confidence_level,
        "n_paths_per_forecast": n_paths_per_forecast,
        "base_seed": seed,
        "forecast_start_date": forecast_dates[0],
        "forecast_end_date": forecast_dates[-1],
        "observations": observations,
        "breaches": breaches,
        "expected_breaches": observations * expected_probability,
        "breach_rate": breaches / observations,
        "expected_breach_rate": expected_probability,
        "mean_forecast_var_usd": float(np.mean(forecasts)),
        "median_forecast_var_usd": float(np.median(forecasts)),
        "mean_actual_loss_usd": float(np.mean(losses)),
        "maximum_actual_loss_usd": float(np.max(losses)),
        "kupiec_unconditional_coverage": kupiec.to_dict(),
        "christoffersen_independence": (None if independence is None else independence.to_dict()),
        "forecast_observations": forecast_rows,
        "breach_observations": breach_rows,
        "checks": checks,
        "passed": all(checks.values()),
    }
