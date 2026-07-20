#!/usr/bin/env python3
"""Run a fixed-snapshot simulation and write machine-readable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from aave_risk_monitor.app.services import load_price_history, simulate_snapshot
from aave_risk_monitor.data import load_snapshot
from aave_risk_monitor.domain import calculate_portfolio_totals
from aave_risk_monitor.simulation import MetricEstimate, SimulationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE_ROOT = PROJECT_ROOT / "src" / "aave_risk_monitor"
SOURCE_HASH_ALGORITHM = "sha256-source-tree-v1:path-nul-size-nul-content-nul"


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


def _source_tree_sha256(root: Path) -> tuple[str, tuple[str, ...]]:
    """Hash Python source contents and relative paths independently of checkout path."""

    files = tuple(
        sorted(
            (path for path in root.rglob("*.py") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not files:
        raise ValueError(f"no Python source files found under {root}")

    digest = hashlib.sha256()
    relative_paths: list[str] = []
    for path in files:
        relative_path = path.relative_to(root).as_posix()
        content = path.read_bytes()
        relative_paths.append(relative_path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest(), tuple(relative_paths)


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


def _estimate_payload(estimate: MetricEstimate | None) -> dict[str, Any] | None:
    if estimate is None:
        return None
    interval = estimate.confidence_interval
    return {
        "value": estimate.value,
        "confidence_interval": (
            None
            if interval is None
            else {
                "lower": interval.lower,
                "upper": interval.upper,
                "confidence_level": interval.confidence_level,
                "method": interval.method,
            }
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=_positive_int, default=20_000)
    parser.add_argument("--horizon-days", type=_positive_int, default=30)
    parser.add_argument("--seed", type=_non_negative_int, default=20_260_720)
    parser.add_argument("--bootstrap-resamples", type=_non_negative_int, default=500)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot = load_snapshot(args.snapshot)
    history = load_price_history(args.history, snapshot=snapshot)
    config = SimulationConfig(
        n_paths=args.paths,
        horizon_steps=args.horizon_days,
        step_size_days=1.0,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    calibration, result = simulate_snapshot(snapshot, history, config=config)
    totals = calculate_portfolio_totals(snapshot.positions)
    source_sha256, source_files = _source_tree_sha256(MODEL_SOURCE_ROOT)

    horizons = tuple(
        dict.fromkeys(day for day in (1, 7, config.horizon_steps) if day <= config.horizon_steps)
    )
    ttl = result.summary.time_to_liquidation
    payload: dict[str, Any] = {
        "schema_version": 2,
        "inputs": {
            "snapshot_path": str(args.snapshot),
            "snapshot_sha256": _sha256(args.snapshot),
            "history_path": str(args.history),
            "history_sha256": _sha256(args.history),
            "history_rows": len(history),
            "history_start_date": history.index[0].date().isoformat(),
            "history_end_date": history.index[-1].date().isoformat(),
            "chain_id": snapshot.chain_id,
            "block_number": snapshot.block.number,
            "block_hash": snapshot.block.hash,
            "block_timestamp": snapshot.block.timestamp,
            "pool_revision": snapshot.contracts.pool_revision,
            "oracle": snapshot.contracts.oracle,
            "user": snapshot.user_address,
            "user_emode_category": snapshot.user_emode_category,
            "emode_data_complete": snapshot.emode_data_complete,
        },
        "provenance": {
            "generator": {
                "path": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "source_tree": {
                "root": MODEL_SOURCE_ROOT.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": source_sha256,
                "algorithm": SOURCE_HASH_ALGORITHM,
                "file_count": len(source_files),
                "files": list(source_files),
            },
            "runtime": _runtime_versions(),
        },
        "portfolio": {
            "calculated_health_factor": (
                None if totals.health_factor is None else str(totals.health_factor)
            ),
            "onchain_health_factor": (
                None
                if snapshot.account.health_factor is None
                else str(snapshot.account.health_factor)
            ),
            "collateral_value_usd": str(totals.collateral_value_usd),
            "debt_value_usd": str(totals.debt_value_usd),
        },
        "simulation_config": {
            "n_paths": config.n_paths,
            "horizon_steps": config.horizon_steps,
            "step_size_days": config.step_size_days,
            "seed": config.seed,
            "chunk_size": config.chunk_size,
            "stored_paths": config.stored_paths,
            "liquidation_rule": "health_factor_strictly_below_1",
            "liquidation_boundary_guard_ulps": (config.liquidation_boundary_guard_ulps),
            "drift_mode": config.drift_mode,
            "tail_confidence_levels": list(config.tail_confidence_levels),
            "bootstrap_resamples": config.bootstrap_resamples,
            "interval_confidence_level": config.interval_confidence_level,
        },
        "calibration": {
            "observations": calibration.observations,
            "periods_per_year": calibration.periods_per_year,
            "estimator": calibration.estimator,
            "shrinkage": calibration.shrinkage,
            "asset_addresses": list(calibration.asset_names),
            "daily_mean_log_returns": calibration.mean_log_returns.tolist(),
            "daily_covariance": calibration.covariance.tolist(),
            "correlation": calibration.correlation.tolist(),
            "annualized_volatility": calibration.annualized_volatility.tolist(),
        },
        "risk_results": {
            "liquidation_probability": _estimate_payload(result.summary.liquidation_probability),
            "cumulative_liquidation_probability": {
                str(day): {
                    "value": float(result.cumulative_liquidation_probability[day]),
                    "interval_95_lower": float(
                        result.cumulative_liquidation_probability_lower[day]
                    ),
                    "interval_95_upper": float(
                        result.cumulative_liquidation_probability_upper[day]
                    ),
                }
                for day in horizons
            },
            "time_to_liquidation": {
                "observed_liquidations": ttl.observed_liquidations,
                "right_censored_paths": ttl.censored_paths,
                "conditional_mean_days": _estimate_payload(ttl.conditional_mean_days),
                "conditional_median_days": _estimate_payload(ttl.conditional_median_days),
                "conditional_p05_days": ttl.conditional_p05_days,
                "conditional_p95_days": ttl.conditional_p95_days,
            },
            "terminal_mark_to_market_tail_risk": [
                {
                    "confidence_level": estimate.confidence_level,
                    "value_at_risk_usd": _estimate_payload(estimate.value_at_risk_usd),
                    "expected_shortfall_usd": _estimate_payload(estimate.expected_shortfall_usd),
                }
                for estimate in result.summary.tail_risk
            ],
        },
        "scope": {
            "positions": "fixed token quantities over the horizon",
            "liquidation_probability": "daily discrete first passage",
            "liquidation_numerical_boundary": (
                "strict HF < 1 after an 8-ULP guard against binary floating-point "
                "portfolio-summation noise"
            ),
            "tail_risk": "pre-liquidation terminal mark-to-market; no post-trigger transition",
            "intervals": "Monte Carlo sampling uncertainty only",
            "historical_oracle_address": {
                "address": snapshot.contracts.oracle,
                "csv_rows_match_snapshot_oracle": True,
                "provider_oracle_resolved_at_each_historical_block": False,
                "assumption": (
                    "historical observations query the fixed snapshot oracle contract; "
                    "PoolAddressesProvider.getPriceOracle is not re-resolved at each "
                    "historical block"
                ),
            },
            "emode_metadata": {
                "complete": snapshot.emode_data_complete,
                "user_category": snapshot.user_emode_category,
                "execution_policy": "risk simulation fails closed when metadata is incomplete",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote fixed analysis evidence to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
