#!/usr/bin/env python3
"""Generate deterministic empirical validation evidence from checked-in inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from aave_risk_monitor.app.services import (
    active_positions,
    load_price_history,
    position_key,
    prepare_price_history,
)
from aave_risk_monitor.data import load_snapshot
from aave_risk_monitor.domain import calculate_portfolio_totals
from aave_risk_monitor.simulation import calibrate_log_returns, make_linear_portfolio_revaluer
from aave_risk_monitor.validation import (
    calibration_window_convergence,
    path_count_convergence,
    rolling_var_backtest,
    run_stress_tests,
    time_step_convergence,
)
from aave_risk_monitor.validation._common import coerce_validation_valuation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    PROJECT_ROOT / "snapshots" / "1-ed0c6079229e2d407672a117c22b62064f4a4312-block-25573974.json"
)
DEFAULT_HISTORY = PROJECT_ROOT / "data" / "history" / "ethereum-v3-whale-block-25573974.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "empirical_validation.json"
STABLECOIN_SYMBOLS = frozenset(
    {"USDC", "USDT", "DAI", "USDE", "USDS", "GHO", "LUSD", "FRAX", "PYUSD"}
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _runtime_versions() -> dict[str, Any]:
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": {
            "numpy": metadata.version("numpy"),
            "pandas": metadata.version("pandas"),
            "sklearn": metadata.version("scikit-learn"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--convergence-paths", type=_positive_int, default=20_000)
    parser.add_argument("--stress-paths", type=_positive_int, default=20_000)
    parser.add_argument("--backtest-paths", type=_positive_int, default=4_000)
    parser.add_argument("--target-health-factor", type=float, default=1.05)
    parser.add_argument("--base-seed", type=_non_negative_int, default=20_260_721)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not np.isfinite(args.target_health_factor) or args.target_health_factor <= 1.0:
        raise ValueError("target-health-factor must be finite and greater than 1")

    snapshot = load_snapshot(args.snapshot)
    raw_history = load_price_history(args.history, snapshot=snapshot)
    validated_history = prepare_price_history(snapshot, raw_history, calibration_days=365)
    positions = active_positions(snapshot)
    asset_keys = [position_key(position) for position in positions]
    price_history = validated_history.loc[:, asset_keys].astype(np.float64)
    spot = np.asarray([float(position.price_usd) for position in positions])
    collateral_units = np.asarray([float(position.collateral_balance) for position in positions])
    debt_units = np.asarray([float(position.debt_balance) for position in positions])
    thresholds = np.asarray(
        [
            float(position.effective_liquidation_threshold)
            if position.contributes_collateral
            else 0.0
            for position in positions
        ]
    )
    actual_revaluer = make_linear_portfolio_revaluer(collateral_units, debt_units, thresholds)

    adjusted_collateral_usd = float(np.sum(collateral_units * spot * thresholds))
    debt_value_usd = float(np.sum(debt_units * spot))
    if debt_value_usd <= 0.0:
        raise ValueError("the fixed snapshot has no debt to scale")
    original_health_factor = adjusted_collateral_usd / debt_value_usd
    debt_scale = original_health_factor / args.target_health_factor
    counterfactual_debt_units = debt_units * debt_scale
    counterfactual_revaluer = make_linear_portfolio_revaluer(
        collateral_units,
        counterfactual_debt_units,
        thresholds,
    )
    counterfactual_health_factor = float(
        coerce_validation_valuation(counterfactual_revaluer(spot[None, :])).health_factor[0]
    )

    time_step = time_step_convergence(
        spot_prices_usd=spot,
        price_history=price_history,
        revalue_fn=counterfactual_revaluer,
        n_paths=args.convergence_paths,
        seed=args.base_seed,
    )
    path_counts = path_count_convergence(
        spot_prices_usd=spot,
        price_history=price_history,
        revalue_fn=counterfactual_revaluer,
        path_counts=(2_000, 8_000, max(32_000, args.convergence_paths)),
        seed=args.base_seed + 1,
    )
    calibration_windows = calibration_window_convergence(
        spot_prices_usd=spot,
        price_history=price_history,
        revalue_fn=counterfactual_revaluer,
        n_paths=args.convergence_paths,
        seed=args.base_seed + 2,
    )
    backtest = rolling_var_backtest(
        price_history=price_history,
        revalue_fn=actual_revaluer,
        calibration_window_days=180,
        n_paths_per_forecast=args.backtest_paths,
        seed=args.base_seed + 3,
    )

    stablecoin_indices = tuple(
        index
        for index, position in enumerate(positions)
        if position.symbol.upper() in STABLECOIN_SYMBOLS
    )
    stablecoin_roles: dict[int, tuple[str, ...]] = {}
    for index in stablecoin_indices:
        roles: list[str] = []
        if collateral_units[index] > 0.0 and thresholds[index] > 0.0:
            roles.append("collateral")
        if counterfactual_debt_units[index] > 0.0:
            roles.append("debt")
        stablecoin_roles[index] = tuple(roles) or ("inactive",)
    calibration = calibrate_log_returns(price_history)
    stress = run_stress_tests(
        spot_prices_usd=spot,
        calibration=calibration,
        revalue_fn=counterfactual_revaluer,
        asset_symbols=[position.symbol for position in positions],
        stablecoin_indices=stablecoin_indices,
        stablecoin_roles=stablecoin_roles,
        n_paths=args.stress_paths,
        seed=args.base_seed + 4,
    )

    totals = calculate_portfolio_totals(snapshot.positions)
    generator = Path(__file__).resolve()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "evidence_timestamp": datetime.fromtimestamp(snapshot.block.timestamp, tz=UTC).isoformat(),
        "inputs": {
            "snapshot_path": _relative_path(args.snapshot),
            "snapshot_sha256": _sha256(args.snapshot),
            "history_path": _relative_path(args.history),
            "history_sha256": _sha256(args.history),
            "history_rows": len(price_history),
            "history_start_date": price_history.index[0].date().isoformat(),
            "history_end_date": price_history.index[-1].date().isoformat(),
            "asset_addresses": asset_keys,
            "asset_symbols": [position.symbol for position in positions],
            "chain_id": snapshot.chain_id,
            "block_number": snapshot.block.number,
            "block_hash": snapshot.block.hash,
            "user": snapshot.user_address,
        },
        "provenance": {
            "generator_path": _relative_path(generator),
            "generator_sha256": _sha256(generator),
            "runtime": _runtime_versions(),
            "determinism": {
                "base_seed": args.base_seed,
                "convergence_paths": args.convergence_paths,
                "stress_paths": args.stress_paths,
                "backtest_paths_per_forecast": args.backtest_paths,
            },
        },
        "validation_portfolios": {
            "actual_fixed_snapshot": {
                "purpose": "rolling one-day VaR backtest",
                "health_factor": (
                    None if totals.health_factor is None else float(totals.health_factor)
                ),
                "collateral_value_usd": float(totals.collateral_value_usd),
                "debt_value_usd": float(totals.debt_value_usd),
                "position_rule": "current token quantities held fixed across history",
            },
            "near_boundary_counterfactual": {
                "purpose": "convergence and stress sensitivity",
                "construction": (
                    "multiply every debt token quantity by one common factor; preserve "
                    "collateral, debt composition, thresholds, spot, and history"
                ),
                "source_health_factor": original_health_factor,
                "target_health_factor": args.target_health_factor,
                "realized_initial_health_factor": counterfactual_health_factor,
                "debt_quantity_scale": debt_scale,
                "is_observed_account": False,
            },
        },
        "results": {
            "time_step_convergence": time_step,
            "path_count_convergence": path_counts,
            "calibration_window_convergence": calibration_windows,
            "rolling_out_of_sample_var_backtest": backtest,
            "stablecoin_and_heavy_tail_stress": stress,
        },
        "summary": {
            "time_step_convergence_passed": time_step["comparison"]["passed"],
            "path_count_convergence_passed": path_counts["comparison"]["passed"],
            "calibration_window_convergence_passed": (calibration_windows["comparison"]["passed"]),
            "rolling_var_backtest_passed": backtest["passed"],
            "all_statistical_acceptance_checks_passed": all(
                (
                    time_step["comparison"]["passed"],
                    path_counts["comparison"]["passed"],
                    calibration_windows["comparison"]["passed"],
                    backtest["passed"],
                )
            ),
        },
        "limitations": [
            (
                "The VaR backtest applies the current snapshot's fixed token quantities "
                "to historical prices; it is not a reconstruction of the user's historical "
                "balances or health factor."
            ),
            (
                "The backtest validates one-day terminal mark-to-market VaR coverage, not "
                "historical Aave liquidation events or post-liquidation execution."
            ),
            (
                "The HF=1.05 portfolio is a disclosed counterfactual sensitivity case and "
                "must not be presented as an observed borrower account."
            ),
            (
                "Student-t stress matches the calibrated innovation covariance but is not "
                "a fitted claim that returns follow a t distribution with five degrees of "
                "freedom."
            ),
            (
                "Stablecoin shocks are role-dependent: a downward shock to debt reduces "
                "liabilities, while an upward price break is adverse for stablecoin debt."
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote empirical validation evidence to {args.output}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
