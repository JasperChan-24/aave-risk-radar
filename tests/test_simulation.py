from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aave_risk_monitor.simulation import (
    PortfolioValuation,
    SimulationConfig,
    bootstrap_interval,
    calibrate_log_returns,
    daily_log_returns,
    empirical_expected_shortfall,
    empirical_value_at_risk,
    estimate_tail_risk,
    make_linear_portfolio_revaluer,
    simulate_correlated_gbm,
    simulate_from_calibration,
    wilson_interval,
)


def test_calibration_uses_daily_log_returns_and_returns_psd_covariance() -> None:
    daily_returns = np.array(
        [
            [0.01, 0.002, -0.004],
            [-0.02, -0.001, 0.006],
            [0.015, 0.003, -0.002],
            [0.005, -0.002, 0.001],
            [-0.01, 0.001, 0.003],
        ]
    )
    prices = 100.0 * np.exp(np.vstack([np.zeros(3), np.cumsum(daily_returns, axis=0)]))
    history = pd.DataFrame(
        prices,
        columns=["WETH", "WBTC", "USDC"],
        index=pd.date_range("2026-01-01", periods=prices.shape[0], freq="D"),
    )

    returns = daily_log_returns(history)
    calibration = calibrate_log_returns(history, periods_per_year=365.0)

    np.testing.assert_allclose(returns.to_numpy(), daily_returns, atol=1e-14)
    np.testing.assert_allclose(
        calibration.mean_log_returns,
        daily_returns.mean(axis=0),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        calibration.annualized_covariance,
        calibration.covariance * 365.0,
    )
    np.testing.assert_allclose(
        calibration.annualized_volatility,
        daily_returns.std(axis=0, ddof=1) * np.sqrt(365.0),
        rtol=1e-12,
        atol=1e-15,
    )
    assert calibration.asset_names == ("WETH", "WBTC", "USDC")
    assert calibration.observations == len(daily_returns)
    assert calibration.estimator == "sample-volatility-ledoit-wolf-correlation"
    assert 0.0 <= calibration.shrinkage <= 1.0
    assert np.linalg.eigvalsh(calibration.covariance).min() >= -1e-14


def test_calibration_rejects_irregular_or_invalid_daily_prices() -> None:
    irregular = pd.DataFrame(
        {"WETH": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-04"]),
    )
    with pytest.raises(ValueError, match="exactly one calendar day"):
        calibrate_log_returns(irregular)
    with pytest.raises(ValueError, match="strictly positive"):
        calibrate_log_returns([[100.0], [0.0], [101.0]])
    with pytest.raises(ValueError, match="at least three"):
        calibrate_log_returns([[100.0], [101.0]])
    with pytest.raises(ValueError, match="positive finite"):
        calibrate_log_returns([[100.0], [101.0], [102.0]], periods_per_year=0.0)


def test_calibration_accepts_one_dimensional_array_with_generated_asset_name() -> None:
    calibration = calibrate_log_returns([100.0, 101.0, 99.0, 102.0])
    assert calibration.asset_names == ("asset_0",)
    assert calibration.covariance.shape == (1, 1)


def test_correlation_shrinkage_preserves_low_volatility_asset_scale() -> None:
    returns = np.array(
        [
            [0.05, 0.0001],
            [-0.04, -0.0002],
            [0.03, 0.0002],
            [-0.02, -0.0001],
            [0.01, 0.0001],
            [-0.03, -0.0001],
        ]
    )
    prices = 100 * np.exp(np.vstack([np.zeros(2), np.cumsum(returns, axis=0)]))

    calibration = calibrate_log_returns(prices)

    np.testing.assert_allclose(
        calibration.annualized_volatility,
        np.std(returns, axis=0, ddof=1) * np.sqrt(365.0),
        rtol=1e-12,
    )
    assert calibration.annualized_volatility[0] > (100 * calibration.annualized_volatility[1])


def test_linear_revaluer_marks_each_asset_and_applies_liquidation_threshold() -> None:
    revalue = make_linear_portfolio_revaluer(
        collateral_units=[2.0, 50.0, 3.0],
        debt_units=[0.0, 25.0, 1.0],
        liquidation_thresholds=[0.8, 0.0, 0.5],
    )
    prices = np.array([[[100.0, 1.0, 10.0], [80.0, 1.0, 20.0]]])

    valuation = revalue(prices)

    np.testing.assert_allclose(valuation.collateral_usd, [[280.0, 270.0]])
    np.testing.assert_allclose(valuation.debt_usd, [[35.0, 45.0]])
    np.testing.assert_allclose(valuation.health_factor, [[175.0 / 35.0, 158.0 / 45.0]])
    np.testing.assert_allclose(valuation.net_equity_usd, [[245.0, 225.0]])


def test_linear_revaluer_handles_no_debt_and_rejects_invalid_portfolios() -> None:
    no_debt = make_linear_portfolio_revaluer([1.0], [0.0], [0.8])
    assert np.isinf(no_debt(np.array([[100.0]])).health_factor).all()
    with pytest.raises(ValueError, match="matching shapes"):
        make_linear_portfolio_revaluer([1.0], [0.0, 1.0], [0.8])
    with pytest.raises(ValueError, match="cannot be negative"):
        make_linear_portfolio_revaluer([-1.0], [0.0], [0.8])
    with pytest.raises(ValueError, match=r"fractions in \[0, 1\]"):
        make_linear_portfolio_revaluer([1.0], [0.0], [80.0])
    with pytest.raises(ValueError, match="asset dimension"):
        no_debt(np.ones((2, 2)))


def _risky_two_asset_revaluer():
    return make_linear_portfolio_revaluer(
        collateral_units=[2.0, 0.0],
        debt_units=[0.0, 100.0],
        liquidation_thresholds=[0.8, 0.0],
    )


def test_seeded_simulation_is_identical_across_chunk_sizes() -> None:
    common = dict(
        n_paths=257,
        horizon_steps=8,
        step_size_days=0.5,
        seed=1977,
        stored_paths=17,
        bootstrap_resamples=30,
    )
    inputs = dict(
        spot_prices_usd=[100.0, 1.0],
        mean_log_returns=[-0.005, 0.0],
        covariance=[[0.003, 0.0001], [0.0001, 0.00005]],
        revalue_fn=_risky_two_asset_revaluer(),
    )

    small_chunks = simulate_correlated_gbm(
        **inputs,
        config=SimulationConfig(**common, chunk_size=13),
    )
    large_chunks = simulate_correlated_gbm(
        **inputs,
        config=SimulationConfig(**common, chunk_size=257),
    )

    np.testing.assert_array_equal(
        small_chunks.sampled_price_paths_usd,
        large_chunks.sampled_price_paths_usd,
    )
    np.testing.assert_array_equal(
        small_chunks.first_liquidation_step,
        large_chunks.first_liquidation_step,
    )
    np.testing.assert_array_equal(small_chunks.equity_loss_usd, large_chunks.equity_loss_usd)
    assert small_chunks.summary == large_chunks.summary


def test_multi_asset_paths_reproduce_requested_covariance() -> None:
    covariance = np.array([[0.04, 0.03], [0.03, 0.09]])
    n_paths = 30_000
    result = simulate_correlated_gbm(
        spot_prices_usd=[100.0, 50.0],
        mean_log_returns=[0.01, -0.02],
        covariance=covariance,
        revalue_fn=make_linear_portfolio_revaluer([1.0, 1.0], [0.0, 0.0], [0.8, 0.7]),
        config=SimulationConfig(
            n_paths=n_paths,
            horizon_steps=1,
            step_size_days=0.25,
            seed=123,
            chunk_size=3_001,
            stored_paths=n_paths,
            bootstrap_resamples=0,
        ),
    )
    sampled_log_returns = np.log(
        result.sampled_price_paths_usd[:, 1, :] / result.sampled_price_paths_usd[:, 0, :]
    )

    np.testing.assert_allclose(sampled_log_returns.mean(axis=0), [0.0025, -0.005], atol=0.002)
    np.testing.assert_allclose(
        np.cov(sampled_log_returns, rowvar=False, ddof=0),
        covariance * 0.25,
        rtol=0.04,
        atol=0.0005,
    )


def test_calibrated_simulation_defaults_to_zero_arithmetic_drift() -> None:
    log_returns = np.array([0.04, -0.02, 0.03, 0.01, -0.01, 0.05])
    price_history = pd.DataFrame(
        {"WETH": 100.0 * np.exp(np.r_[0.0, np.cumsum(log_returns)])},
        index=pd.date_range("2026-01-01", periods=log_returns.size + 1, freq="D"),
    )
    calibration = calibrate_log_returns(price_history)
    revalue = make_linear_portfolio_revaluer([1.0], [0.0], [0.8])
    common = dict(
        n_paths=50,
        horizon_steps=2,
        step_size_days=0.5,
        seed=42,
        chunk_size=11,
        stored_paths=50,
        bootstrap_resamples=0,
    )

    zero_drift = simulate_from_calibration(
        spot_prices_usd=[100.0],
        calibration=calibration,
        revalue_fn=revalue,
        config=SimulationConfig(**common, drift_mode="zero"),
    )
    historical_drift = simulate_from_calibration(
        spot_prices_usd=[100.0],
        calibration=calibration,
        revalue_fn=revalue,
        config=SimulationConfig(**common, drift_mode="historical"),
    )

    zero_log_mean = -0.5 * calibration.covariance[0, 0]
    expected_log_difference = (calibration.mean_log_returns[0] - zero_log_mean) * common[
        "step_size_days"
    ]
    actual_log_difference = np.log(
        historical_drift.sampled_price_paths_usd[:, 1, 0]
        / zero_drift.sampled_price_paths_usd[:, 1, 0]
    )
    np.testing.assert_allclose(actual_log_difference, expected_log_difference, atol=1e-14)


def test_research_defaults_are_reproducible_and_bounded_for_plotting() -> None:
    config = SimulationConfig()
    assert config.n_paths == 20_000
    assert config.seed == 20_260_720
    assert config.drift_mode == "zero"
    assert config.stored_paths <= 200


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_paths": 0}, "n_paths"),
        ({"horizon_steps": 0}, "horizon_steps"),
        ({"step_size_days": 0.0}, "step_size_days"),
        ({"seed": -1}, "seed"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"stored_paths": -1}, "stored_paths"),
        ({"liquidation_barrier": 0.0}, "liquidation_barrier"),
        ({"liquidation_boundary_guard_ulps": -1.0}, "boundary_guard_ulps"),
        ({"drift_mode": "forecast"}, "drift_mode"),
        ({"bootstrap_resamples": -1}, "bootstrap_resamples"),
        ({"interval_confidence_level": 1.0}, "interval_confidence_level"),
        ({"tail_confidence_levels": ()}, "tail_confidence_levels"),
        ({"tail_confidence_levels": (0.95, 0.95)}, "unique"),
        ({"tail_confidence_levels": (0.0,)}, "between zero and one"),
    ],
)
def test_simulation_config_rejects_invalid_contracts(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SimulationConfig(**overrides)


def test_first_barrier_breach_ttl_wilson_interval_and_loss_metrics() -> None:
    result = simulate_correlated_gbm(
        spot_prices_usd=[100.0, 1.0],
        mean_log_returns=[-0.30, 0.0],
        covariance=np.zeros((2, 2)),
        revalue_fn=_risky_two_asset_revaluer(),
        config=SimulationConfig(
            n_paths=100,
            horizon_steps=3,
            step_size_days=1.0,
            seed=1,
            chunk_size=17,
            stored_paths=5,
            bootstrap_resamples=40,
        ),
    )

    np.testing.assert_array_equal(result.first_liquidation_step, np.full(100, 2))
    np.testing.assert_array_equal(result.time_to_liquidation_days, np.full(100, 2.0))
    np.testing.assert_allclose(result.cumulative_liquidation_probability, [0.0, 0.0, 1.0, 1.0])
    assert result.cumulative_liquidation_probability_lower.shape == (4,)
    assert result.cumulative_liquidation_probability_upper.shape == (4,)
    assert np.all(
        result.cumulative_liquidation_probability_lower <= result.cumulative_liquidation_probability
    )
    assert np.all(
        result.cumulative_liquidation_probability <= result.cumulative_liquidation_probability_upper
    )
    probability = result.summary.liquidation_probability
    assert probability.value == 1.0
    assert probability.confidence_interval is not None
    assert probability.confidence_interval.method == "wilson"
    assert 0.96 < probability.confidence_interval.lower < 1.0
    ttl = result.summary.time_to_liquidation
    assert ttl.observed_liquidations == 100
    assert ttl.censored_paths == 0
    assert ttl.conditional_mean_days is not None
    assert ttl.conditional_mean_days.value == 2.0
    assert ttl.conditional_mean_days.confidence_interval is not None
    assert ttl.conditional_median_days is not None
    assert ttl.conditional_median_days.value == 2.0
    assert result.equity_loss_usd[0] > 0.0
    assert [estimate.confidence_level for estimate in result.summary.tail_risk] == [0.95, 0.99]
    for estimate in result.summary.tail_risk:
        assert estimate.value_at_risk_usd.confidence_interval is not None
        assert estimate.expected_shortfall_usd.confidence_interval is not None
        assert estimate.expected_shortfall_usd.value >= estimate.value_at_risk_usd.value


def test_non_liquidated_paths_are_explicitly_right_censored() -> None:
    result = simulate_correlated_gbm(
        spot_prices_usd=[100.0, 1.0],
        mean_log_returns=[0.0, 0.0],
        covariance=np.zeros((2, 2)),
        revalue_fn=_risky_two_asset_revaluer(),
        config=SimulationConfig(
            n_paths=12,
            horizon_steps=2,
            seed=4,
            chunk_size=5,
            stored_paths=0,
            bootstrap_resamples=0,
        ),
    )

    np.testing.assert_array_equal(result.first_liquidation_step, np.full(12, -1))
    assert np.isnan(result.time_to_liquidation_days).all()
    ttl = result.summary.time_to_liquidation
    assert ttl.observed_liquidations == 0
    assert ttl.censored_paths == 12
    assert ttl.conditional_mean_days is None
    assert result.sampled_price_paths_usd.shape == (0, 3, 2)
    assert result.sampled_valuation.health_factor.shape == (0, 3)
    interval = result.summary.liquidation_probability.confidence_interval
    assert interval is not None
    assert interval.lower == 0.0
    assert interval.upper > 0.0


def test_initially_liquidatable_snapshot_has_deterministic_probability_interval() -> None:
    initially_liquidatable = make_linear_portfolio_revaluer(
        collateral_units=[0.9, 0.0],
        debt_units=[0.0, 1.0],
        liquidation_thresholds=[1.0, 0.0],
    )
    result = simulate_correlated_gbm(
        spot_prices_usd=[1.0, 1.0],
        mean_log_returns=[0.0, 0.0],
        covariance=np.zeros((2, 2)),
        revalue_fn=initially_liquidatable,
        config=SimulationConfig(
            n_paths=10,
            horizon_steps=2,
            stored_paths=1,
            bootstrap_resamples=0,
        ),
    )

    np.testing.assert_array_equal(result.first_liquidation_step, np.zeros(10))
    interval = result.summary.liquidation_probability.confidence_interval
    assert interval is not None
    assert interval.lower == interval.upper == 1.0
    assert interval.method == "deterministic-initial-state"
    np.testing.assert_array_equal(result.cumulative_liquidation_probability, 1.0)
    np.testing.assert_array_equal(result.cumulative_liquidation_probability_lower, 1.0)
    np.testing.assert_array_equal(result.cumulative_liquidation_probability_upper, 1.0)


def test_health_factor_exactly_one_is_not_liquidatable() -> None:
    exactly_one = make_linear_portfolio_revaluer(
        collateral_units=[1.25, 0.0],
        debt_units=[0.0, 100.0],
        liquidation_thresholds=[0.8, 0.0],
    )
    result = simulate_correlated_gbm(
        spot_prices_usd=[100.0, 1.0],
        mean_log_returns=[0.0, 0.0],
        covariance=np.zeros((2, 2)),
        revalue_fn=exactly_one,
        config=SimulationConfig(
            n_paths=20,
            horizon_steps=2,
            seed=9,
            stored_paths=2,
            bootstrap_resamples=0,
        ),
    )

    np.testing.assert_allclose(result.sampled_valuation.health_factor, 1.0)
    assert result.summary.liquidation_probability.value == 0.0
    np.testing.assert_array_equal(result.first_liquidation_step, -1)


def test_multi_asset_float_summation_does_not_turn_exact_hf_one_into_a_breach() -> None:
    # Exact decimal accounting is 0.3 / (0.1 + 0.2) == 1, while the binary
    # denominator rounds slightly above 0.3.
    exactly_one = make_linear_portfolio_revaluer(
        collateral_units=[0.3, 0.0, 0.0],
        debt_units=[0.0, 0.1, 0.2],
        liquidation_thresholds=[1.0, 0.0, 0.0],
    )
    result = simulate_correlated_gbm(
        spot_prices_usd=[1.0, 1.0, 1.0],
        mean_log_returns=[0.0, 0.0, 0.0],
        covariance=np.zeros((3, 3)),
        revalue_fn=exactly_one,
        config=SimulationConfig(
            n_paths=10,
            horizon_steps=1,
            stored_paths=1,
            bootstrap_resamples=0,
        ),
    )

    assert result.sampled_valuation.health_factor[0, 0] < 1.0
    assert result.summary.liquidation_probability.value == 0.0
    np.testing.assert_array_equal(result.first_liquidation_step, -1)
    assert result.cumulative_liquidation_probability_upper[0] == 0.0


def test_var_es_and_bootstrap_confidence_intervals_are_reproducible() -> None:
    losses = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 100.0])
    assert empirical_value_at_risk(losses, 0.75) == pytest.approx(3.25)
    assert empirical_expected_shortfall(losses, 0.75) == pytest.approx(52.0)

    first = estimate_tail_risk(
        losses,
        confidence_levels=(0.75,),
        bootstrap_resamples=100,
        seed=99,
    )
    second = estimate_tail_risk(
        losses,
        confidence_levels=(0.75,),
        bootstrap_resamples=100,
        seed=99,
    )
    assert first == second
    assert first[0].value_at_risk_usd.confidence_interval is not None
    assert first[0].expected_shortfall_usd.confidence_interval is not None

    tied_losses = np.r_[np.zeros(99), 100.0]
    assert empirical_value_at_risk(tied_losses, 0.95) == 0.0
    assert empirical_expected_shortfall(tied_losses, 0.95) == pytest.approx(20.0)

    with pytest.raises(ValueError, match="positive"):
        bootstrap_interval(losses, np.mean, n_resamples=0)
    with pytest.raises(ValueError, match="unique"):
        estimate_tail_risk(losses, confidence_levels=(0.95, 0.95))


def test_wilson_interval_handles_rare_event_boundaries() -> None:
    no_events = wilson_interval(0, 100)
    all_events = wilson_interval(100, 100)
    assert no_events.lower == 0.0
    assert 0.0 < no_events.upper < 0.05
    assert 0.95 < all_events.lower < 1.0
    assert all_events.upper == 1.0


def test_simulation_accepts_mapping_callback_and_rejects_bad_shapes() -> None:
    def mapping_callback(prices: np.ndarray):
        leading = prices.shape[:-1]
        return {
            "collateral_usd": np.full(leading, 2.0),
            "debt_usd": np.full(leading, 1.0),
            "health_factor": np.full(leading, 1.5),
            "net_equity_usd": np.full(leading, 1.0),
        }

    result = simulate_correlated_gbm(
        spot_prices_usd=[1.0],
        mean_log_returns=[0.0],
        covariance=[[0.0]],
        revalue_fn=mapping_callback,
        config=SimulationConfig(n_paths=3, horizon_steps=1, bootstrap_resamples=0),
    )
    assert result.summary.liquidation_probability.value == 0.0

    def bad_callback(prices: np.ndarray) -> PortfolioValuation:
        bad_shape = np.ones(prices.shape[:1])
        return PortfolioValuation(bad_shape, bad_shape, bad_shape, bad_shape)

    with pytest.raises(ValueError, match="expected"):
        simulate_correlated_gbm(
            spot_prices_usd=[1.0],
            mean_log_returns=[0.0],
            covariance=[[0.0]],
            revalue_fn=bad_callback,
            config=SimulationConfig(n_paths=3, horizon_steps=1, bootstrap_resamples=0),
        )


def test_simulation_rejects_non_psd_covariance() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        simulate_correlated_gbm(
            spot_prices_usd=[100.0, 1.0],
            mean_log_returns=[0.0, 0.0],
            covariance=[[1.0, 2.0], [2.0, 1.0]],
            revalue_fn=_risky_two_asset_revaluer(),
            config=SimulationConfig(n_paths=10, horizon_steps=1, bootstrap_resamples=0),
        )
