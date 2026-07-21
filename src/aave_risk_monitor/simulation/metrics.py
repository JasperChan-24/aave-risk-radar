"""Pure statistical estimators used by the simulation engine."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    lower: float
    upper: float
    confidence_level: float
    method: str


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    value: float
    confidence_interval: ConfidenceInterval | None = None


@dataclass(frozen=True, slots=True)
class TailRiskEstimate:
    """Value at Risk and Expected Shortfall of a USD loss distribution."""

    confidence_level: float
    value_at_risk_usd: MetricEstimate
    expected_shortfall_usd: MetricEstimate


def _finite_vector(values: ArrayLike, *, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _validate_confidence_level(confidence_level: float) -> None:
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Wilson score interval for a binomial probability.

    Unlike the Wald interval, this remains within ``[0, 1]`` for zero or all
    successes and is useful for rare liquidation events.
    """

    _validate_confidence_level(confidence_level)
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")

    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    probability = successes / trials
    z_squared = z_score**2
    denominator = 1.0 + z_squared / trials
    center = (probability + z_squared / (2.0 * trials)) / denominator
    half_width = (
        z_score
        * np.sqrt(probability * (1.0 - probability) / trials + z_squared / (4.0 * trials**2))
        / denominator
    )
    return ConfidenceInterval(
        lower=max(0.0, float(center - half_width)),
        upper=min(1.0, float(center + half_width)),
        confidence_level=confidence_level,
        method="wilson",
    )


def bootstrap_interval(
    values: ArrayLike,
    estimator: Callable[[FloatArray], float],
    *,
    n_resamples: int = 500,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap confidence interval for a one-sample estimator."""

    vector = _finite_vector(values, name="values")
    _validate_confidence_level(confidence_level)
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    random = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = vector[random.integers(0, vector.size, size=vector.size)]
        estimates[index] = estimator(sample)
    if not np.all(np.isfinite(estimates)):
        raise ValueError("bootstrap estimator returned a non-finite value")

    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return ConfidenceInterval(
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence_level,
        method="percentile-bootstrap",
    )


def empirical_value_at_risk(losses_usd: ArrayLike, confidence_level: float) -> float:
    """Empirical VaR of the loss variable (positive values are losses)."""

    _validate_confidence_level(confidence_level)
    losses = _finite_vector(losses_usd, name="losses_usd")
    return float(np.quantile(losses, confidence_level))


def empirical_expected_shortfall(losses_usd: ArrayLike, confidence_level: float) -> float:
    """Average the worst empirical ``1 - confidence_level`` loss mass.

    A fractional boundary observation is weighted when the requested tail mass
    is not an integer.  This is preferable to selecting every observation at
    or above VaR: the latter can include far more than the intended tail when
    many paths have the same loss.
    """

    losses = _finite_vector(losses_usd, name="losses_usd")
    _validate_confidence_level(confidence_level)
    ordered = np.sort(losses)
    tail_mass = (1.0 - confidence_level) * losses.size
    nearest_integer = round(tail_mass)
    if np.isclose(tail_mass, nearest_integer, rtol=0.0, atol=1e-12):
        tail_mass = float(nearest_integer)
    full_observations = int(np.floor(tail_mass))
    fractional_observation = tail_mass - full_observations
    tail_sum = (
        float(np.sum(ordered[-full_observations:], dtype=np.float64)) if full_observations else 0.0
    )
    if fractional_observation:
        tail_sum += fractional_observation * float(ordered[-full_observations - 1])
    return tail_sum / tail_mass


def estimate_tail_risk(
    losses_usd: ArrayLike,
    *,
    confidence_levels: Iterable[float] = (0.95, 0.99),
    bootstrap_resamples: int = 500,
    bootstrap_confidence_level: float = 0.95,
    seed: int = 0,
) -> tuple[TailRiskEstimate, ...]:
    """Estimate VaR/ES and deterministic percentile-bootstrap intervals."""

    losses = _finite_vector(losses_usd, name="losses_usd")
    levels = tuple(float(level) for level in confidence_levels)
    if not levels:
        raise ValueError("at least one confidence level is required")
    if len(set(levels)) != len(levels):
        raise ValueError("confidence levels must be unique")
    for level in levels:
        _validate_confidence_level(level)
    if bootstrap_resamples < 0:
        raise ValueError("bootstrap_resamples cannot be negative")

    estimates: list[TailRiskEstimate] = []
    for index, level in enumerate(levels):
        var_value = empirical_value_at_risk(losses, level)
        es_value = empirical_expected_shortfall(losses, level)

        def var_estimator(sample: FloatArray, selected_level: float = level) -> float:
            return empirical_value_at_risk(sample, selected_level)

        def es_estimator(sample: FloatArray, selected_level: float = level) -> float:
            return empirical_expected_shortfall(sample, selected_level)

        var_interval = None
        es_interval = None
        if bootstrap_resamples:
            # Separate reproducible streams prevent metric ordering from changing
            # any already-published interval.
            var_interval = bootstrap_interval(
                losses,
                var_estimator,
                n_resamples=bootstrap_resamples,
                confidence_level=bootstrap_confidence_level,
                seed=seed + index * 2,
            )
            es_interval = bootstrap_interval(
                losses,
                es_estimator,
                n_resamples=bootstrap_resamples,
                confidence_level=bootstrap_confidence_level,
                seed=seed + index * 2 + 1,
            )

        estimates.append(
            TailRiskEstimate(
                confidence_level=level,
                value_at_risk_usd=MetricEstimate(var_value, var_interval),
                expected_shortfall_usd=MetricEstimate(es_value, es_interval),
            )
        )
    return tuple(estimates)
