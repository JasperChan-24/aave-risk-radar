"""Shared, deterministic primitives for empirical model validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aave_risk_monitor.simulation import (
    MarketCalibration,
    PortfolioRevaluation,
    PortfolioValuation,
    SimulationConfig,
    simulate_from_calibration,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RiskPoint:
    """A compact set of comparable Monte Carlo risk outputs."""

    liquidation_probability: float
    liquidation_probability_ci_lower: float
    liquidation_probability_ci_upper: float
    value_at_risk_usd: float
    expected_shortfall_usd: float

    @property
    def liquidation_probability_ci_width(self) -> float:
        return self.liquidation_probability_ci_upper - self.liquidation_probability_ci_lower

    def to_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["liquidation_probability_ci_width"] = self.liquidation_probability_ci_width
        return payload


def coerce_validation_valuation(output: object) -> PortfolioValuation:
    """Normalize the public revaluation callback union for validation code."""

    if isinstance(output, PortfolioValuation):
        return output
    if not isinstance(output, Mapping):
        raise TypeError("revaluation callback returned an unsupported value")
    required = (
        "collateral_usd",
        "debt_usd",
        "health_factor",
        "net_equity_usd",
    )
    missing = [name for name in required if name not in output]
    if missing:
        raise ValueError(f"revaluation output is missing fields: {missing}")
    return PortfolioValuation(
        collateral_usd=np.asarray(output["collateral_usd"], dtype=np.float64),
        debt_usd=np.asarray(output["debt_usd"], dtype=np.float64),
        health_factor=np.asarray(output["health_factor"], dtype=np.float64),
        net_equity_usd=np.asarray(output["net_equity_usd"], dtype=np.float64),
    )


def validate_market_vectors(
    spot_prices_usd: ArrayLike,
    calibration: MarketCalibration,
) -> FloatArray:
    spot = np.asarray(spot_prices_usd, dtype=np.float64)
    if spot.ndim != 1 or spot.size != len(calibration.asset_names):
        raise ValueError("spot prices must match the calibrated asset order")
    if not np.all(np.isfinite(spot)) or np.any(spot <= 0.0):
        raise ValueError("spot prices must be finite and strictly positive")
    return spot


def simulation_risk_point(
    *,
    spot_prices_usd: ArrayLike,
    calibration: MarketCalibration,
    revalue_fn: PortfolioRevaluation,
    n_paths: int,
    horizon_steps: int,
    step_size_days: float,
    seed: int,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 0,
) -> RiskPoint:
    """Run the production simulator and extract one comparable risk point."""

    validate_market_vectors(spot_prices_usd, calibration)
    result = simulate_from_calibration(
        spot_prices_usd=spot_prices_usd,
        calibration=calibration,
        revalue_fn=revalue_fn,
        config=SimulationConfig(
            n_paths=n_paths,
            horizon_steps=horizon_steps,
            step_size_days=step_size_days,
            seed=seed,
            chunk_size=min(2_000, n_paths),
            stored_paths=0,
            tail_confidence_levels=(confidence_level,),
            bootstrap_resamples=bootstrap_resamples,
        ),
    )
    probability = result.summary.liquidation_probability
    interval = probability.confidence_interval
    if interval is None:  # pragma: no cover - the simulator always supplies it
        raise AssertionError("liquidation probability interval is missing")
    tail = result.summary.tail_risk[0]
    return RiskPoint(
        liquidation_probability=probability.value,
        liquidation_probability_ci_lower=interval.lower,
        liquidation_probability_ci_upper=interval.upper,
        value_at_risk_usd=tail.value_at_risk_usd.value,
        expected_shortfall_usd=tail.expected_shortfall_usd.value,
    )


def relative_difference(value: float, reference: float) -> float:
    """Symmetric only at zero: report absolute error when reference is zero."""

    if not np.isfinite(value) or not np.isfinite(reference):
        raise ValueError("comparison values must be finite")
    if reference == 0.0:
        return abs(value)
    return abs(value - reference) / abs(reference)


def ensure_strictly_increasing(values: tuple[float, ...], *, name: str) -> None:
    if len(values) < 2:
        raise ValueError(f"{name} must include at least two settings")
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{name} settings must be positive and finite")
    if any(right <= left for left, right in zip(values, values[1:], strict=False)):
        raise ValueError(f"{name} settings must be strictly increasing")


def bool_checks(checks: Mapping[str, bool]) -> dict[str, Any]:
    normalized = {name: bool(value) for name, value in checks.items()}
    return {"checks": normalized, "passed": all(normalized.values())}
