"""Historical calibration for the multi-asset market model.

The calibration unit is deliberately explicit: one row is one calendar day and
the returned means/covariance describe *daily log returns*.  Simulation code can
therefore scale both by an explicit ``step_size_days`` without hiding a time
convention in a volatility parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.covariance import LedoitWolf

FloatArray = NDArray[np.float64]
_PSD_TOLERANCE: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class MarketCalibration:
    """Calibrated daily joint log-return model.

    ``mean_log_returns`` and ``covariance`` are daily.  Annualized fields are
    supplied for reporting only.  The high-level simulator defaults to zero
    arithmetic drift and uses this historical mean only when explicitly asked.
    """

    asset_names: tuple[str, ...]
    mean_log_returns: FloatArray
    covariance: FloatArray
    correlation: FloatArray
    annualized_log_returns: FloatArray
    annualized_covariance: FloatArray
    annualized_volatility: FloatArray
    observations: int
    periods_per_year: float
    estimator: str
    shrinkage: float


def _as_price_frame(prices: pd.DataFrame | ArrayLike) -> pd.DataFrame:
    if isinstance(prices, pd.DataFrame):
        if prices.columns.has_duplicates:
            raise ValueError("price-history asset names must be unique")
        frame = prices.copy()
        frame.columns = [str(column) for column in frame.columns]
    else:
        values = np.asarray(prices, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("prices must have shape (dates, assets)")
        frame = pd.DataFrame(values, columns=[f"asset_{index}" for index in range(values.shape[1])])

    if frame.shape[1] == 0:
        raise ValueError("prices must include at least one asset")
    if len(frame) < 3:
        raise ValueError("at least three daily price observations are required")

    try:
        frame = frame.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("all prices must be numeric") from exc

    values = frame.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("prices must be finite; align and clean missing history upstream")
    if np.any(values <= 0.0):
        raise ValueError("prices must be strictly positive before taking log returns")

    if isinstance(frame.index, pd.DatetimeIndex):
        if frame.index.has_duplicates:
            raise ValueError("daily price-history timestamps must be unique")
        if not frame.index.is_monotonic_increasing:
            raise ValueError("daily price-history timestamps must be increasing")
        index_nanoseconds = frame.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        gaps = np.diff(index_nanoseconds)
        one_day_ns = 86_400 * 1_000_000_000
        if gaps.size and not np.all(gaps == one_day_ns):
            raise ValueError(
                "price history must have exactly one calendar day between rows; "
                "resample it in the data layer first"
            )

    return frame


def daily_log_returns(prices: pd.DataFrame | ArrayLike) -> pd.DataFrame:
    """Return aligned one-calendar-day log returns.

    Missing values and irregular timestamps are rejected rather than silently
    converted into returns with different horizons.
    """

    frame = _as_price_frame(prices)
    values = np.diff(np.log(frame.to_numpy(dtype=np.float64)), axis=0)
    return pd.DataFrame(values, columns=frame.columns, index=frame.index[1:])


def _correlation_from_covariance(covariance: FloatArray) -> FloatArray:
    standard_deviation = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(standard_deviation, standard_deviation)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0.0,
    )
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, np.where(standard_deviation > 0.0, 1.0, 0.0))
    return correlation


def _validate_psd(covariance: FloatArray) -> FloatArray:
    """Remove only numerical negative eigenvalues from an estimated covariance."""

    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
    if float(np.min(eigenvalues, initial=0.0)) < -_PSD_TOLERANCE * scale:
        raise ValueError("covariance estimator returned a non-PSD matrix")
    clipped = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * clipped) @ eigenvectors.T


def calibrate_log_returns(
    prices: pd.DataFrame | ArrayLike,
    *,
    periods_per_year: float = 365.0,
) -> MarketCalibration:
    """Estimate sample volatilities and a shrunk return correlation matrix.

    Applying a spherical Ledoit-Wolf target directly to heteroskedastic raw
    returns transfers variance from volatile assets into stable assets. We
    therefore standardize each non-constant return series, shrink only their
    correlation structure, and scale it back with each asset's own sample
    volatility. The resulting covariance is positive semidefinite without
    erasing economically important cross-asset volatility differences.
    """

    if not np.isfinite(periods_per_year) or periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be a positive finite number")

    returns = daily_log_returns(prices)
    values = returns.to_numpy(dtype=np.float64)
    if values.shape[0] < 2:
        raise ValueError("at least two daily log-return observations are required")

    mean = np.mean(values, axis=0, dtype=np.float64)
    centered = values - mean
    sample_volatility = np.std(values, axis=0, ddof=1, dtype=np.float64)
    non_constant = np.flatnonzero(sample_volatility > 0.0)
    correlation = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    shrinkage = 0.0
    if non_constant.size == 1:
        correlation[non_constant[0], non_constant[0]] = 1.0
    elif non_constant.size > 1:
        standardized = centered[:, non_constant] / sample_volatility[non_constant]
        estimator = LedoitWolf(assume_centered=True).fit(standardized)
        shrunk_correlation = _correlation_from_covariance(
            np.asarray(estimator.covariance_, dtype=np.float64)
        )
        correlation[np.ix_(non_constant, non_constant)] = shrunk_correlation
        shrinkage = float(estimator.shrinkage_)

    covariance = _validate_psd(correlation * np.outer(sample_volatility, sample_volatility))
    annualized_covariance = covariance * periods_per_year

    return MarketCalibration(
        asset_names=tuple(returns.columns),
        mean_log_returns=mean,
        covariance=covariance,
        correlation=_correlation_from_covariance(covariance),
        annualized_log_returns=mean * periods_per_year,
        annualized_covariance=annualized_covariance,
        annualized_volatility=np.sqrt(np.clip(np.diag(annualized_covariance), 0.0, None)),
        observations=values.shape[0],
        periods_per_year=float(periods_per_year),
        estimator="sample-volatility-ledoit-wolf-correlation",
        shrinkage=shrinkage,
    )
