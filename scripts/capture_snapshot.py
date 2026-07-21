#!/usr/bin/env python3
"""Capture a versioned, same-block Aave V3 account snapshot."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3

from aave_risk_monitor.config import Settings
from aave_risk_monitor.data import AaveV3DataSource, save_snapshot
from aave_risk_monitor.domain import reconcile_health_factor


def _block_identifier(value: str) -> str | int:
    if value == "latest":
        return value
    try:
        block = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "block must be 'latest' or a non-negative integer"
        ) from exc
    if block < 0:
        raise argparse.ArgumentTypeError("block must be non-negative")
    return block


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Aave borrower address")
    parser.add_argument(
        "--block",
        type=_block_identifier,
        default="latest",
        help="Ethereum block number or 'latest' (default: latest)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON or directory (default: SNAPSHOT_DIRECTORY)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing deterministic snapshot file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv(Path.cwd() / ".env", override=False)
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        parser.error(str(exc))

    web3 = Web3(Web3.HTTPProvider(settings.rpc_url, request_kwargs={"timeout": 30}))
    if not web3.is_connected():
        parser.error("Ethereum RPC connection failed")

    snapshot = AaveV3DataSource(
        web3,
        settings.market,
        reconciliation_tolerance_bps=settings.hf_reconciliation_tolerance_bps,
    ).fetch_portfolio(args.user, args.block)
    destination = save_snapshot(
        snapshot,
        args.output or settings.snapshot_directory,
        overwrite=args.overwrite,
    )
    reconciliation = reconcile_health_factor(
        snapshot,
        tolerance_bps=settings.hf_reconciliation_tolerance_bps,
    )
    print(
        json.dumps(
            {
                "snapshot": str(destination),
                "chain_id": snapshot.chain_id,
                "block_number": snapshot.block.number,
                "block_hash": snapshot.block.hash,
                "pool_revision": snapshot.contracts.pool_revision,
                "positions": len(snapshot.positions),
                "health_factor_reconciled": reconciliation.within_tolerance,
                "health_factor_difference_bps": (
                    None
                    if reconciliation.difference_bps is None
                    else str(reconciliation.difference_bps)
                ),
                "warnings": list(snapshot.warnings),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
