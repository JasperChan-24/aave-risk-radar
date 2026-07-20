"""Chunked, reproducible multi-asset Monte Carlo simulation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .calibration import MarketCalibration
from .metrics import (
    ConfidenceInterval,
    MetricEstimate,
    TailRiskEstimate,
    bootstrap_interval,
    estimate_tail_risk,
    wilson_interval,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PortfolioValuation:
    """Pathwise USD valuation returned by a portfolio revaluation callback.

    Each field must have the same leading shape as ``prices_usd[..., asset]``.
    A callback can alternatively return a mapping with these four field names,
    which keeps the simulation package independent of application domain types.
    """

    collateral_usd: FloatArray
    debt_usd: FloatArray
    health_factor: FloatArray
    net_equity_usd: FloatArray


type ValuationOutput = PortfolioValuation | Mapping[str, ArrayLike]
type PortfolioRevaluation = Callable[[FloatArray], ValuationOutput]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    n_paths: int = 20_000
    horizon_steps: int = 30
    step_size_days: float = 1.0
    seed: int = 20_260_720
    chunk_size: int = 2_000
    stored_paths: int = 200
    liquidation_barrier: float = 1.0
    liquidation_boundary_guard_ulps: float = 8.0
    drift_mode: Literal["zero", "historical"] = "zero"
    tail_confidence_levels: tuple[float, ...] = (0.95, 0.99)
    bootstrap_resamples: int = 500
    interval_confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive")
        if not np.isfinite(self.step_size_days) or self.step_size_days <= 0.0:
            raise ValueError("step_size_days must be positive and finite")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.stored_paths < 0:
            raise ValueError("stored_paths cannot be negative")
        if not np.isfinite(self.liquidation_barrier) or self.liquidation_barrier <= 0.0:
            raise ValueError("liquidation_barrier must be positive and finite")
        if (
            not np.isfinite(self.liquidation_boundary_guard_ulps)
            or self.liquidation_boundary_guard_ulps < 0.0
        ):
            raise ValueError("liquidation_boundary_guard_ulps must be non-negative and finite")
        if self.drift_mode not in {"zero", "historical"}:
            raise ValueError("drift_mode must be 'zero' or 'historical'")
        if self.bootstrap_resamples < 0:
            raise ValueError("bootstrap_resamples cannot be negative")
        if not 0.0 < self.interval_confidence_level < 1.0:
            raise ValueError("interval_confidence_level must be between zero and one")
        if not self.tail_confidence_levels:
            raise ValueError("tail_confidence_levels must not be empty")
        if len(set(self.tail_confidence_levels)) != len(self.tail_confidence_levels):
            raise ValueError("tail_confidence_levels must be unique")
        if any(not 0.0 < level < 1.0 for level in self.tail_confidence_levels):
            raise ValueError("tail confidence levels must be between zero and one")

    @property
    def horizon_days(self) -> float:
        return self.horizon_steps * self.step_size_days


@dataclass(frozen=True, slots=True)
class TimeToLiquidationSummary:
    """Conditional TTL statistics; non-liquidated paths are right-censored."""

    observed_liquidations: int
    censored_paths: int
    conditional_mean_days: MetricEstimate | None
    conditional_median_days: MetricEstimate | None
    conditional_p05_days: float | None
    conditional_p95_days: float | None


@dataclass(frozen=True, slots=True)
class SimulationSummary:
    liquidation_probability: MetricEstimate
    time_to_liquidation: TimeToLiquidationSummary
    tail_risk: tuple[TailRiskEstimate, ...]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Full path outcomes plus a bounded path sample for plotting.

    ``time_to_liquidation_days`` uses NaN for right-censored paths.
    ``equity_loss_usd`` is initial net equity minus terminal net equity, so a
    negative value represents a gain and positive values represent losses.
    """

    config: SimulationConfig
    summary: SimulationSummary
    first_liquidation_step: IntArray
    time_to_liquidation_days: FloatArray
    terminal_equity_usd: FloatArray
    equity_loss_usd: FloatArray
    cumulative_liquidation_probability: FloatArray
    cumulative_liquidation_probability_lower: FloatArray
    cumulative_liquidation_probability_upper: FloatArray
    sampled_price_paths_usd: FloatArray
    sampled_valuation: PortfolioValuation


def make_linear_portfolio_revaluer(
    collateral_units: ArrayLike,
    debt_units: ArrayLike,
    liquidation_thresholds: ArrayLike,
) -> PortfolioRevaluation:
    """Build an asset-level mark-to-market callback.

    Assets not enabled as collateral should have a zero liquidation threshold.
    Variable and stable debt balances should be combined in ``debt_units``.
    Thresholds are fractions in ``[0, 1]``, not basis points.
    """

    collateral = np.asarray(collateral_units, dtype=np.float64)
    debt = np.asarray(debt_units, dtype=np.float64)
    thresholds = np.asarray(liquidation_thresholds, dtype=np.float64)
    if collateral.ndim != 1 or debt.ndim != 1 or thresholds.ndim != 1:
        raise ValueError("portfolio vectors must be one-dimensional")
    if not collateral.size or collateral.shape != debt.shape or debt.shape != thresholds.shape:
        raise ValueError("portfolio vectors must be non-empty and have matching shapes")
    if not (
        np.all(np.isfinite(collateral))
        and np.all(np.isfinite(debt))
        and np.all(np.isfinite(thresholds))
    ):
        raise ValueError("portfolio vectors must be finite")
    if np.any(collateral < 0.0) or np.any(debt < 0.0):
        raise ValueError("collateral and debt units cannot be negative")
    if np.any((thresholds < 0.0) | (thresholds > 1.0)):
        raise ValueError("liquidation thresholds must be fractions in [0, 1]")

    # Do not retain mutable caller-owned arrays in the closure.
    collateral = collateral.copy()
    debt = debt.copy()
    thresholds = thresholds.copy()

    def revalue(prices_usd: FloatArray) -> PortfolioValuation:
        prices = np.asarray(prices_usd, dtype=np.float64)
        if prices.ndim < 2 or prices.shape[-1] != collateral.size:
            raise ValueError("prices must end with the configured asset dimension")
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0.0):
            raise ValueError("prices must be finite and strictly positive")

        supplied_values = prices * collateral
        collateral_usd = np.sum(supplied_values, axis=-1, dtype=np.float64)
        debt_usd = np.sum(prices * debt, axis=-1, dtype=np.float64)
        adjusted_collateral = np.sum(
            supplied_values * thresholds,
            axis=-1,
            dtype=np.float64,
        )
        health_factor = np.divide(
            adjusted_collateral,
            debt_usd,
            out=np.full_like(adjusted_collateral, np.inf),
            where=debt_usd > 0.0,
        )
        return PortfolioValuation(
            collateral_usd=collateral_usd,
            debt_usd=debt_usd,
            health_factor=health_factor,
            net_equity_usd=collateral_usd - debt_usd,
        )

    return revalue


def _coerce_valuation(
    output: ValuationOutput, expected_shape: tuple[int, ...]
) -> PortfolioValuation:
    if isinstance(output, PortfolioValuation):
        values = output
    elif isinstance(output, Mapping):
        required = {
            "collateral_usd",
            "debt_usd",
            "health_factor",
            "net_equity_usd",
        }
        missing = required.difference(output)
        if missing:
            raise ValueError(f"revaluation output is missing fields: {sorted(missing)}")
        values = PortfolioValuation(
            collateral_usd=np.asarray(output["collateral_usd"], dtype=np.float64),
            debt_usd=np.asarray(output["debt_usd"], dtype=np.float64),
            health_factor=np.asarray(output["health_factor"], dtype=np.float64),
            net_equity_usd=np.asarray(output["net_equity_usd"], dtype=np.float64),
        )
    else:
        raise TypeError("revaluation callback must return PortfolioValuation or a mapping")

    arrays = {
        "collateral_usd": np.asarray(values.collateral_usd, dtype=np.float64),
        "debt_usd": np.asarray(values.debt_usd, dtype=np.float64),
        "health_factor": np.asarray(values.health_factor, dtype=np.float64),
        "net_equity_usd": np.asarray(values.net_equity_usd, dtype=np.float64),
    }
    for name, array in arrays.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"revaluation field {name!r} has shape {array.shape}; expected {expected_shape}"
            )
    if np.any(~np.isfinite(arrays["collateral_usd"])) or np.any(arrays["collateral_usd"] < 0.0):
        raise ValueError("collateral_usd must be finite and non-negative")
    if np.any(~np.isfinite(arrays["debt_usd"])) or np.any(arrays["debt_usd"] < 0.0):
        raise ValueError("debt_usd must be finite and non-negative")
    if np.any(np.isnan(arrays["health_factor"])) or np.any(arrays["health_factor"] < 0.0):
        raise ValueError("health_factor must be non-negative and cannot be NaN")
    if np.any(np.isneginf(arrays["health_factor"])):
        raise ValueError("health_factor cannot be negative infinity")
    if np.any(~np.isfinite(arrays["net_equity_usd"])):
        raise ValueError("net_equity_usd must be finite")

    return PortfolioValuation(**arrays)


def _covariance_factor(covariance: FloatArray) -> FloatArray:
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    if float(np.min(eigenvalues, initial=0.0)) < -1e-12 * scale:
        raise ValueError("covariance must be positive semidefinite")
    return eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))


def _validate_market_inputs(
    spot_prices_usd: ArrayLike,
    mean_log_returns: ArrayLike,
    covariance: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    spot = np.asarray(spot_prices_usd, dtype=np.float64)
    mean = np.asarray(mean_log_returns, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if spot.ndim != 1 or not spot.size:
        raise ValueError("spot_prices_usd must be a non-empty one-dimensional vector")
    if mean.shape != spot.shape:
        raise ValueError("mean_log_returns must match spot_prices_usd")
    if cov.shape != (spot.size, spot.size):
        raise ValueError("covariance must have shape (assets, assets)")
    if not np.all(np.isfinite(spot)) or np.any(spot <= 0.0):
        raise ValueError("spot prices must be finite and strictly positive")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
        raise ValueError("market parameters must be finite")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    if np.any(np.diag(cov) < 0.0):
        raise ValueError("covariance diagonal cannot be negative")
    _covariance_factor(cov)  # PSD validation before allocating path arrays.
    return spot, mean, cov


def _summarize_time_to_liquidation(
    times_days: FloatArray,
    *,
    total_paths: int,
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
) -> TimeToLiquidationSummary:
    observed = times_days[np.isfinite(times_days)]
    observed_count = int(observed.size)
    if not observed_count:
        return TimeToLiquidationSummary(
            observed_liquidations=0,
            censored_paths=total_paths,
            conditional_mean_days=None,
            conditional_median_days=None,
            conditional_p05_days=None,
            conditional_p95_days=None,
        )

    mean_interval = None
    median_interval = None
    if bootstrap_resamples:
        mean_interval = bootstrap_interval(
            observed,
            lambda values: float(np.mean(values, dtype=np.float64)),
            n_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            seed=seed,
        )
        median_interval = bootstrap_interval(
            observed,
            lambda values: float(np.median(values)),
            n_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            seed=seed + 1,
        )
    p05, p95 = np.quantile(observed, [0.05, 0.95])
    return TimeToLiquidationSummary(
        observed_liquidations=observed_count,
        censored_paths=total_paths - observed_count,
        conditional_mean_days=MetricEstimate(
            float(np.mean(observed, dtype=np.float64)),
            mean_interval,
        ),
        conditional_median_days=MetricEstimate(float(np.median(observed)), median_interval),
        conditional_p05_days=float(p05),
        conditional_p95_days=float(p95),
    )


def simulate_correlated_gbm(
    *,
    spot_prices_usd: ArrayLike,
    mean_log_returns: ArrayLike,
    covariance: ArrayLike,
    revalue_fn: PortfolioRevaluation,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """Simulate correlated prices and compute liquidation and loss metrics.

    Market inputs are explicit daily log-return parameters.  For a step of
    ``dt`` days, log-return mean scales by ``dt`` and covariance by ``dt``.
    Paths are generated in chunks and only ``stored_paths`` complete paths are
    retained; all scalar outcomes needed for statistics are retained.  This
    low-level function always uses the supplied log-return mean directly;
    ``config.drift_mode`` is applied by :func:`simulate_from_calibration`.
    """

    simulation_config = config or SimulationConfig()
    spot, mean, cov = _validate_market_inputs(
        spot_prices_usd,
        mean_log_returns,
        covariance,
    )
    factor = _covariance_factor(cov)
    random = np.random.default_rng(simulation_config.seed)
    time_count = simulation_config.horizon_steps + 1

    first_liquidation_step = np.empty(simulation_config.n_paths, dtype=np.int64)
    terminal_equity = np.empty(simulation_config.n_paths, dtype=np.float64)
    equity_loss = np.empty(simulation_config.n_paths, dtype=np.float64)
    sampled_prices: list[FloatArray] = []
    sampled_collateral: list[FloatArray] = []
    sampled_debt: list[FloatArray] = []
    sampled_health_factor: list[FloatArray] = []
    sampled_equity: list[FloatArray] = []
    stored_count = 0

    for start in range(0, simulation_config.n_paths, simulation_config.chunk_size):
        stop = min(start + simulation_config.chunk_size, simulation_config.n_paths)
        current_paths = stop - start
        independent_shocks = random.standard_normal(
            (current_paths, simulation_config.horizon_steps, spot.size)
        )
        correlated_shocks = (
            independent_shocks @ factor.T * np.sqrt(simulation_config.step_size_days)
        )
        log_increments = correlated_shocks + mean * simulation_config.step_size_days
        cumulative_log_returns = np.cumsum(log_increments, axis=1)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            future_prices = spot * np.exp(cumulative_log_returns)
        if not np.all(np.isfinite(future_prices)) or np.any(future_prices <= 0.0):
            raise FloatingPointError("simulated prices overflowed or underflowed")

        prices = np.empty((current_paths, time_count, spot.size), dtype=np.float64)
        prices[:, 0, :] = spot
        prices[:, 1:, :] = future_prices
        valuation = _coerce_valuation(revalue_fn(prices), (current_paths, time_count))

        # Portfolio summation in binary floating point can place an
        # algebraically exact HF=1 a handful of ULPs below one. Treat that
        # numerical neighbourhood as the boundary; economically meaningful
        # breaches remain strict and comfortably below it.
        boundary_tolerance = simulation_config.liquidation_boundary_guard_ulps * abs(
            np.spacing(np.float64(simulation_config.liquidation_barrier))
        )
        breaches = valuation.health_factor < (
            simulation_config.liquidation_barrier - boundary_tolerance
        )
        has_liquidation = np.any(breaches, axis=1)
        first_steps = np.argmax(breaches, axis=1).astype(np.int64)
        first_steps[~has_liquidation] = -1
        first_liquidation_step[start:stop] = first_steps
        terminal_equity[start:stop] = valuation.net_equity_usd[:, -1]
        equity_loss[start:stop] = valuation.net_equity_usd[:, 0] - valuation.net_equity_usd[:, -1]

        remaining_to_store = min(
            simulation_config.stored_paths - stored_count,
            current_paths,
        )
        if remaining_to_store > 0:
            selection = slice(0, remaining_to_store)
            sampled_prices.append(prices[selection].copy())
            sampled_collateral.append(valuation.collateral_usd[selection].copy())
            sampled_debt.append(valuation.debt_usd[selection].copy())
            sampled_health_factor.append(valuation.health_factor[selection].copy())
            sampled_equity.append(valuation.net_equity_usd[selection].copy())
            stored_count += remaining_to_store

    liquidated = first_liquidation_step >= 0
    liquidation_count = int(np.count_nonzero(liquidated))
    liquidation_probability = liquidation_count / simulation_config.n_paths
    initial_state_liquidated = bool(np.all(first_liquidation_step == 0))
    probability_interval = (
        ConfidenceInterval(
            lower=1.0,
            upper=1.0,
            confidence_level=simulation_config.interval_confidence_level,
            method="deterministic-initial-state",
        )
        if initial_state_liquidated
        else wilson_interval(
            liquidation_count,
            simulation_config.n_paths,
            confidence_level=simulation_config.interval_confidence_level,
        )
    )
    time_to_liquidation = np.full(simulation_config.n_paths, np.nan, dtype=np.float64)
    time_to_liquidation[liquidated] = (
        first_liquidation_step[liquidated] * simulation_config.step_size_days
    )
    bootstrap_seed = simulation_config.seed + 10_000_019
    ttl_summary = _summarize_time_to_liquidation(
        time_to_liquidation,
        total_paths=simulation_config.n_paths,
        bootstrap_resamples=simulation_config.bootstrap_resamples,
        confidence_level=simulation_config.interval_confidence_level,
        seed=bootstrap_seed,
    )
    tail_risk = estimate_tail_risk(
        equity_loss,
        confidence_levels=simulation_config.tail_confidence_levels,
        bootstrap_resamples=simulation_config.bootstrap_resamples,
        bootstrap_confidence_level=simulation_config.interval_confidence_level,
        seed=bootstrap_seed + 2,
    )
    cumulative_counts = np.array(
        [
            np.count_nonzero((first_liquidation_step >= 0) & (first_liquidation_step <= step))
            for step in range(time_count)
        ],
        dtype=np.int64,
    )
    cumulative_probability = cumulative_counts.astype(np.float64) / simulation_config.n_paths
    cumulative_intervals = tuple(
        wilson_interval(
            int(count),
            simulation_config.n_paths,
            confidence_level=simulation_config.interval_confidence_level,
        )
        for count in cumulative_counts
    )
    cumulative_probability_lower = np.asarray(
        [interval.lower for interval in cumulative_intervals], dtype=np.float64
    )
    cumulative_probability_upper = np.asarray(
        [interval.upper for interval in cumulative_intervals], dtype=np.float64
    )
    # The t=0 state is fixed by the snapshot, not estimated from independent
    # Monte Carlo draws. If it is already liquidatable, every later cumulative
    # first-passage probability is deterministically one as well.
    if initial_state_liquidated:
        cumulative_probability_lower.fill(1.0)
        cumulative_probability_upper.fill(1.0)
    else:
        cumulative_probability_lower[0] = cumulative_probability[0]
        cumulative_probability_upper[0] = cumulative_probability[0]

    price_shape = (0, time_count, spot.size)
    valuation_shape = (0, time_count)
    sampled_valuation = PortfolioValuation(
        collateral_usd=(
            np.concatenate(sampled_collateral, axis=0)
            if sampled_collateral
            else np.empty(valuation_shape, dtype=np.float64)
        ),
        debt_usd=(
            np.concatenate(sampled_debt, axis=0)
            if sampled_debt
            else np.empty(valuation_shape, dtype=np.float64)
        ),
        health_factor=(
            np.concatenate(sampled_health_factor, axis=0)
            if sampled_health_factor
            else np.empty(valuation_shape, dtype=np.float64)
        ),
        net_equity_usd=(
            np.concatenate(sampled_equity, axis=0)
            if sampled_equity
            else np.empty(valuation_shape, dtype=np.float64)
        ),
    )
    return SimulationResult(
        config=simulation_config,
        summary=SimulationSummary(
            liquidation_probability=MetricEstimate(
                liquidation_probability,
                probability_interval,
            ),
            time_to_liquidation=ttl_summary,
            tail_risk=tail_risk,
        ),
        first_liquidation_step=first_liquidation_step,
        time_to_liquidation_days=time_to_liquidation,
        terminal_equity_usd=terminal_equity,
        equity_loss_usd=equity_loss,
        cumulative_liquidation_probability=cumulative_probability,
        cumulative_liquidation_probability_lower=cumulative_probability_lower,
        cumulative_liquidation_probability_upper=cumulative_probability_upper,
        sampled_price_paths_usd=(
            np.concatenate(sampled_prices, axis=0)
            if sampled_prices
            else np.empty(price_shape, dtype=np.float64)
        ),
        sampled_valuation=sampled_valuation,
    )


def simulate_from_calibration(
    *,
    spot_prices_usd: ArrayLike,
    calibration: MarketCalibration,
    revalue_fn: PortfolioRevaluation,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """Run the calibrated model using the configured arithmetic-drift policy.

    The default ``drift_mode='zero'`` avoids extrapolating a noisy historical
    return mean over a short risk horizon.  Under GBM, zero arithmetic drift is
    represented by daily log-return mean ``-0.5 * variance``.  Select
    ``'historical'`` explicitly to use the calibrated historical log-return
    mean instead.
    """

    if np.asarray(spot_prices_usd).size != len(calibration.asset_names):
        raise ValueError("spot price count must match the calibrated asset order")
    simulation_config = config or SimulationConfig()
    if simulation_config.drift_mode == "zero":
        mean_log_returns = -0.5 * np.diag(calibration.covariance)
    else:
        mean_log_returns = calibration.mean_log_returns
    return simulate_correlated_gbm(
        spot_prices_usd=spot_prices_usd,
        mean_log_returns=mean_log_returns,
        covariance=calibration.covariance,
        revalue_fn=revalue_fn,
        config=simulation_config,
    )
