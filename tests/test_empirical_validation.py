from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aave_risk_monitor.simulation import (
    calibrate_log_returns,
    make_linear_portfolio_revaluer,
)
from aave_risk_monitor.validation import (
    calibration_window_convergence,
    christoffersen_independence_test,
    covariance_matched_student_t_risk,
    kupiec_unconditional_coverage_test,
    path_count_convergence,
    rolling_var_backtest,
    run_stress_tests,
    time_step_convergence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "reports" / "empirical_validation.json"


def _history(days: int = 81) -> pd.DataFrame:
    index = np.arange(days - 1, dtype=np.float64)
    returns = np.column_stack(
        (
            0.018 * np.sin(index * 0.71) + 0.006 * np.cos(index * 0.13),
            0.014 * np.sin(index * 0.59 + 0.4) - 0.004 * np.cos(index * 0.17),
            0.0015 * np.sin(index * 0.83) + 0.0004 * np.cos(index * 0.23),
        )
    )
    # Fixed, disclosed shocks ensure the backtest exercises breach accounting.
    returns[35, 0] = -0.085
    returns[52, 1] = 0.065
    prices = np.asarray([100.0, 100.0, 1.0]) * np.exp(
        np.vstack((np.zeros(3), np.cumsum(returns, axis=0)))
    )
    return pd.DataFrame(
        prices,
        columns=["COLL", "DEBT", "USD"],
        index=pd.date_range("2025-01-01", periods=days, freq="D"),
    )


def _near_boundary_revaluer(spot: np.ndarray):
    base_debt_units = np.asarray([0.0, 0.75, 4.0])
    debt_scale = (0.8 * spot[0]) / (1.05 * float(np.sum(base_debt_units * spot)))
    return make_linear_portfolio_revaluer(
        collateral_units=[1.0, 0.0, 0.0],
        debt_units=base_debt_units * debt_scale,
        liquidation_thresholds=[0.8, 0.0, 0.0],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_convergence_suites_are_deterministic_and_disclose_acceptance_thresholds() -> None:
    history = _history()
    spot = history.iloc[-1].to_numpy()
    revaluer = _near_boundary_revaluer(spot)

    time_step = time_step_convergence(
        spot_prices_usd=spot,
        price_history=history,
        revalue_fn=revaluer,
        step_sizes_days=(1.0, 0.5, 0.25),
        horizon_days=4.0,
        n_paths=600,
        seed=101,
    )
    repeat = time_step_convergence(
        spot_prices_usd=spot,
        price_history=history,
        revalue_fn=revaluer,
        step_sizes_days=(1.0, 0.5, 0.25),
        horizon_days=4.0,
        n_paths=600,
        seed=101,
    )
    assert time_step == repeat
    assert time_step["comparison"]["reference_step_size_days"] == 0.25
    assert set(time_step["comparison"]["checks"]) == set(time_step["comparison"]["thresholds"])

    path_count = path_count_convergence(
        spot_prices_usd=spot,
        price_history=history,
        revalue_fn=revaluer,
        path_counts=(200, 500, 1_000),
        horizon_days=4,
        seed=102,
    )
    assert path_count["comparison"]["reference_n_paths"] == 1_000
    assert (
        path_count["results"][-1]["liquidation_probability_ci_width"]
        < (path_count["results"][0]["liquidation_probability_ci_width"])
    )

    windows = calibration_window_convergence(
        spot_prices_usd=spot,
        price_history=history,
        revalue_fn=revaluer,
        calibration_windows_days=(20, 40, 60),
        horizon_days=4,
        n_paths=600,
        seed=103,
    )
    assert [row["observations"] for row in windows["results"]] == [20, 40, 60]
    assert isinstance(windows["comparison"]["passed"], bool)


def test_convergence_rejects_malformed_grids() -> None:
    history = _history()
    spot = history.iloc[-1].to_numpy()
    revaluer = _near_boundary_revaluer(spot)
    with pytest.raises(ValueError, match="strictly decreasing"):
        time_step_convergence(
            spot_prices_usd=spot,
            price_history=history,
            revalue_fn=revaluer,
            step_sizes_days=(1.0, 0.5, 0.75),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        path_count_convergence(
            spot_prices_usd=spot,
            price_history=history,
            revalue_fn=revaluer,
            path_counts=(1_000, 500),
        )
    with pytest.raises(ValueError, match="too short"):
        calibration_window_convergence(
            spot_prices_usd=spot,
            price_history=history,
            revalue_fn=revaluer,
            calibration_windows_days=(30, 90),
        )


def test_rolling_var_backtest_is_deterministic_and_has_no_future_lookahead() -> None:
    history = _history()
    revaluer = _near_boundary_revaluer(history.iloc[-1].to_numpy())
    common = {
        "revalue_fn": revaluer,
        "calibration_window_days": 30,
        "confidence_level": 0.95,
        "n_paths_per_forecast": 300,
        "seed": 404,
    }
    result = rolling_var_backtest(price_history=history, **common)
    repeat = rolling_var_backtest(price_history=history, **common)
    assert result == repeat
    assert result["observations"] == len(history) - 31
    assert len(result["forecast_observations"]) == result["observations"]
    assert result["breaches"] == sum(row["breach"] for row in result["forecast_observations"])

    changed_future = history.copy()
    changed_future.iloc[33:, 0] *= 1.7
    changed = rolling_var_backtest(price_history=changed_future, **common)
    assert changed["forecast_observations"][0] == result["forecast_observations"][0]


def test_coverage_tests_have_known_likelihood_ratio_behavior() -> None:
    exact = kupiec_unconditional_coverage_test(
        breaches=5,
        observations=100,
        expected_breach_probability=0.05,
    )
    assert exact.statistic == pytest.approx(0.0)
    assert exact.p_value == pytest.approx(1.0)
    assert exact.passed_at_5pct

    rejected = kupiec_unconditional_coverage_test(
        breaches=0,
        observations=100,
        expected_breach_probability=0.05,
    )
    assert rejected.p_value < 0.05
    assert not rejected.passed_at_5pct

    clustered = np.asarray([False] * 20 + [True] * 5 + [False] * 20 + [True] * 5)
    independence = christoffersen_independence_test(clustered)
    assert independence is not None
    assert independence.p_value < 0.05


def test_heavy_tail_and_stablecoin_stress_are_seeded_and_role_sensitive() -> None:
    history = _history()
    spot = history.iloc[-1].to_numpy()
    calibration = calibrate_log_returns(history)
    revaluer = _near_boundary_revaluer(spot)

    student = covariance_matched_student_t_risk(
        spot_prices_usd=spot,
        calibration=calibration,
        revalue_fn=revaluer,
        n_paths=1_000,
        horizon_steps=5,
        seed=505,
    )
    repeated = covariance_matched_student_t_risk(
        spot_prices_usd=spot,
        calibration=calibration,
        revalue_fn=revaluer,
        n_paths=1_000,
        horizon_steps=5,
        seed=505,
    )
    assert student == repeated
    assert np.isfinite(student.expected_shortfall_usd)

    stress = run_stress_tests(
        spot_prices_usd=spot,
        calibration=calibration,
        revalue_fn=revaluer,
        asset_symbols=history.columns,
        stablecoin_indices=(2,),
        stablecoin_roles={2: ("debt",)},
        n_paths=1_000,
        horizon_days=5,
        seed=506,
    )
    down, up = stress["stablecoin_depeg_scenarios"]
    assert down["stablecoin_price_shock_pct"] == -10.0
    assert up["stablecoin_price_shock_pct"] == 10.0
    assert down["initial_health_factor"] > up["initial_health_factor"]
    assert stress["stablecoin_exposures"][0]["roles"] == ["debt"]


def test_checked_in_empirical_validation_artifact_integrity() -> None:
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    inputs = payload["inputs"]
    snapshot = PROJECT_ROOT / inputs["snapshot_path"]
    history = PROJECT_ROOT / inputs["history_path"]
    assert _sha256(snapshot) == inputs["snapshot_sha256"]
    assert _sha256(history) == inputs["history_sha256"]
    generator = PROJECT_ROOT / payload["provenance"]["generator_path"]
    assert generator == PROJECT_ROOT / "scripts" / "run_empirical_validation.py"
    assert _sha256(generator) == payload["provenance"]["generator_sha256"]
    counterfactual = payload["validation_portfolios"]["near_boundary_counterfactual"]
    assert counterfactual["is_observed_account"] is False
    assert counterfactual["target_health_factor"] == pytest.approx(1.05)
    assert counterfactual["realized_initial_health_factor"] == pytest.approx(1.05)
    backtest = payload["results"]["rolling_out_of_sample_var_backtest"]
    assert len(backtest["forecast_observations"]) == backtest["observations"]
    assert all(isinstance(value, bool) for value in payload["summary"].values())
