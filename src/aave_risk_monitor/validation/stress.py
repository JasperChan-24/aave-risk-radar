"""Stablecoin-jump and covariance-matched heavy-tail stress experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from aave_risk_monitor.simulation import (
    MarketCalibration,
    PortfolioRevaluation,
    empirical_expected_shortfall,
    empirical_value_at_risk,
)

from ._common import (
    RiskPoint,
    coerce_validation_valuation,
    simulation_risk_point,
    validate_market_vectors,
)


def _covariance_factor(covariance: np.ndarray) -> np.ndarray:
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if float(np.min(eigenvalues, initial=0.0)) < -1e-12:
        raise ValueError("covariance must be positive semidefinite")
    return eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))


def covariance_matched_student_t_risk(
    *,
    spot_prices_usd: ArrayLike,
    calibration: MarketCalibration,
    revalue_fn: PortfolioRevaluation,
    degrees_of_freedom: float = 5.0,
    n_paths: int = 20_000,
    horizon_steps: int = 30,
    step_size_days: float = 1.0,
    seed: int = 20_260_725,
    confidence_level: float = 0.99,
) -> RiskPoint:
    """Simulate multivariate Student-t innovations with matched covariance.

    One chi-square scale is shared across assets within each time step, which
    creates a multivariate t innovation. Multiplication by ``sqrt((df-2)/X)``
    makes its covariance equal to the calibrated Gaussian covariance for
    ``df > 2``.  The production model's zero-arithmetic-drift log location is
    retained solely to isolate the innovation-tail change.
    """

    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 2.0:
        raise ValueError("degrees_of_freedom must exceed 2 for finite covariance")
    if n_paths <= 0 or horizon_steps <= 0:
        raise ValueError("n_paths and horizon_steps must be positive")
    if not np.isfinite(step_size_days) or step_size_days <= 0.0:
        raise ValueError("step_size_days must be positive and finite")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    spot = validate_market_vectors(spot_prices_usd, calibration)
    factor = _covariance_factor(calibration.covariance)
    random = np.random.default_rng(seed)
    normals = random.standard_normal((n_paths, horizon_steps, spot.size))
    chi_square = random.chisquare(degrees_of_freedom, size=(n_paths, horizon_steps, 1))
    scales = np.sqrt((degrees_of_freedom - 2.0) / chi_square)
    shocks = normals @ factor.T
    shocks *= scales * np.sqrt(step_size_days)
    log_location = -0.5 * np.diag(calibration.covariance) * step_size_days
    cumulative = np.cumsum(shocks + log_location, axis=1)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        future_prices = spot * np.exp(cumulative)
    if not np.all(np.isfinite(future_prices)) or np.any(future_prices <= 0.0):
        raise FloatingPointError("heavy-tail stress generated invalid prices")
    prices = np.empty((n_paths, horizon_steps + 1, spot.size), dtype=np.float64)
    prices[:, 0, :] = spot
    prices[:, 1:, :] = future_prices
    valuation = coerce_validation_valuation(revalue_fn(prices))
    boundary_tolerance = 8.0 * abs(np.spacing(np.float64(1.0)))
    breached = np.any(valuation.health_factor < (1.0 - boundary_tolerance), axis=1)
    losses = valuation.net_equity_usd[:, 0] - valuation.net_equity_usd[:, -1]
    probability = float(np.mean(breached, dtype=np.float64))

    # A Wilson interval is computed by the Gaussian helper in the production
    # path. Import locally to keep this stress engine independent of internals.
    from aave_risk_monitor.simulation import wilson_interval

    interval = wilson_interval(int(np.count_nonzero(breached)), n_paths)
    return RiskPoint(
        liquidation_probability=probability,
        liquidation_probability_ci_lower=interval.lower,
        liquidation_probability_ci_upper=interval.upper,
        value_at_risk_usd=empirical_value_at_risk(losses, confidence_level),
        expected_shortfall_usd=empirical_expected_shortfall(losses, confidence_level),
    )


def run_stress_tests(
    *,
    spot_prices_usd: ArrayLike,
    calibration: MarketCalibration,
    revalue_fn: PortfolioRevaluation,
    asset_symbols: Sequence[str],
    stablecoin_indices: Sequence[int],
    stablecoin_roles: Mapping[int, Sequence[str]],
    depeg_shocks_pct: Sequence[float] = (-10.0, 10.0),
    degrees_of_freedom: float = 5.0,
    n_paths: int = 20_000,
    horizon_days: int = 30,
    seed: int = 20_260_725,
    confidence_level: float = 0.99,
) -> dict[str, Any]:
    """Compare Gaussian/t innovations and permanent stablecoin price jumps."""

    spot = validate_market_vectors(spot_prices_usd, calibration)
    symbols = tuple(str(symbol) for symbol in asset_symbols)
    if len(symbols) != spot.size:
        raise ValueError("asset symbols must match spot prices")
    indices = tuple(int(index) for index in stablecoin_indices)
    if len(set(indices)) != len(indices):
        raise ValueError("stablecoin indices must be unique")
    if any(index < 0 or index >= spot.size for index in indices):
        raise ValueError("stablecoin index is out of range")
    if not indices:
        raise ValueError("at least one stablecoin exposure is required")

    gaussian = simulation_risk_point(
        spot_prices_usd=spot,
        calibration=calibration,
        revalue_fn=revalue_fn,
        n_paths=n_paths,
        horizon_steps=horizon_days,
        step_size_days=1.0,
        seed=seed,
        confidence_level=confidence_level,
    )
    student_t = covariance_matched_student_t_risk(
        spot_prices_usd=spot,
        calibration=calibration,
        revalue_fn=revalue_fn,
        degrees_of_freedom=degrees_of_freedom,
        n_paths=n_paths,
        horizon_steps=horizon_days,
        seed=seed,
        confidence_level=confidence_level,
    )

    scenarios: list[dict[str, Any]] = []
    for shock_pct in depeg_shocks_pct:
        if not np.isfinite(shock_pct) or shock_pct <= -100.0:
            raise ValueError("depeg shocks must be finite and greater than -100%")
        stressed_spot = spot.copy()
        stressed_spot[list(indices)] *= 1.0 + shock_pct / 100.0
        initial_valuation = coerce_validation_valuation(revalue_fn(stressed_spot[None, :]))
        initial_health_factor = float(initial_valuation.health_factor[0])
        risk = simulation_risk_point(
            spot_prices_usd=stressed_spot,
            calibration=calibration,
            revalue_fn=revalue_fn,
            n_paths=n_paths,
            horizon_steps=horizon_days,
            step_size_days=1.0,
            seed=seed,
            confidence_level=confidence_level,
        )
        scenarios.append(
            {
                "stablecoin_price_shock_pct": float(shock_pct),
                "initial_health_factor": initial_health_factor,
                **risk.to_dict(),
            }
        )

    exposures = [
        {
            "index": index,
            "symbol": symbols[index],
            "roles": list(stablecoin_roles.get(index, ("unclassified",))),
        }
        for index in indices
    ]
    return {
        "method": (
            "permanent stablecoin spot jump plus calibrated diffusion; multivariate "
            "Student-t innovations covariance-matched to Gaussian baseline"
        ),
        "horizon_days": horizon_days,
        "n_paths_per_scenario": n_paths,
        "seed_per_scenario": seed,
        "tail_confidence_level": confidence_level,
        "stablecoin_exposures": exposures,
        "gaussian_baseline": gaussian.to_dict(),
        "student_t_stress": {
            "degrees_of_freedom": degrees_of_freedom,
            **student_t.to_dict(),
        },
        "student_t_amplification": {
            "liquidation_probability_difference": (
                student_t.liquidation_probability - gaussian.liquidation_probability
            ),
            "value_at_risk_ratio": (
                student_t.value_at_risk_usd / gaussian.value_at_risk_usd
                if gaussian.value_at_risk_usd != 0.0
                else None
            ),
            "expected_shortfall_ratio": (
                student_t.expected_shortfall_usd / gaussian.expected_shortfall_usd
                if gaussian.expected_shortfall_usd != 0.0
                else None
            ),
        },
        "stablecoin_depeg_scenarios": scenarios,
    }
