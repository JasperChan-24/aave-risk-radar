#!/usr/bin/env python3
"""Capture and verify one fixed Aave V3 revision-11 liquidation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from web3 import Web3

from aave_risk_monitor.data import load_snapshot
from aave_risk_monitor.domain import calculate_portfolio_totals, reconcile_health_factor
from aave_risk_monitor.validation.liquidation_replay import (
    LIQUIDATION_CALL_TOPIC,
    decode_liquidation_call_log,
    replay_liquidation_event,
)

ROOT = Path(__file__).resolve().parents[1]
TX_HASH = "0xd138a0455f087ad399820fb42f4fd35ca8f8986c223609afabf1cba1ffdab62f"
NEAR_SNAPSHOT = (
    ROOT / "snapshots" / "1-eb75251694d4d1a71af99b1f96c819dc5e9ed3ea-block-25572181.json"
)
HIGH_SNAPSHOT = (
    ROOT / "snapshots" / "1-ed0c6079229e2d407672a117c22b62064f4a4312-block-25573974.json"
)
EVIDENCE = ROOT / "data" / "evidence" / f"aave_v3_ethereum_liquidation_{TX_HASH}.json"
REPORT = ROOT / "reports" / "liquidation_replay.json"
FORK_REPORT = ROOT / "reports" / "mainnet_fork_liquidation.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _hex(value: Any) -> str:
    payload = value if isinstance(value, str) else value.hex()
    return "0x" + str(payload).removeprefix("0x").lower()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tx-hash", default=TX_HASH)
    parser.add_argument("--near-snapshot", type=Path, default=NEAR_SNAPSHOT)
    parser.add_argument("--high-snapshot", type=Path, default=HIGH_SNAPSHOT)
    parser.add_argument("--evidence-output", type=Path, default=EVIDENCE)
    parser.add_argument("--report-output", type=Path, default=REPORT)
    parser.add_argument("--fork-report", type=Path, default=FORK_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)
    rpc_url = (
        os.getenv("ETHEREUM_ARCHIVE_RPC_URL")
        or os.getenv("ETHEREUM_RPC_URL")
        or os.getenv("ALCHEMY_RPC_URL")
    )
    if not rpc_url:
        raise SystemExit("an archive-capable Ethereum RPC URL is required")
    web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 60}))
    if not web3.is_connected() or web3.eth.chain_id != 1:
        raise SystemExit("RPC must be connected to Ethereum mainnet (chain id 1)")

    snapshot = load_snapshot(args.near_snapshot)
    control = load_snapshot(args.high_snapshot)
    receipt = web3.eth.get_transaction_receipt(args.tx_hash)
    transaction = web3.eth.get_transaction(args.tx_hash)
    raw_transaction_response = web3.provider.make_request(
        "eth_getRawTransactionByHash", [args.tx_hash]
    )
    raw_signed_transaction = raw_transaction_response.get("result")
    if not raw_signed_transaction:
        raise SystemExit("RPC did not return the historical raw signed transaction")
    if _hex(Web3.keccak(bytes.fromhex(raw_signed_transaction.removeprefix("0x")))) != _hex(
        transaction["hash"]
    ):
        raise SystemExit("raw signed transaction hash does not match historical tx")
    block = web3.eth.get_block(receipt.blockNumber)
    pre_block = web3.eth.get_block(receipt.blockNumber - 1)
    if receipt.status != 1:
        raise SystemExit("fixed liquidation transaction did not succeed")
    if snapshot.block.number != receipt.blockNumber - 1:
        raise SystemExit("snapshot is not pinned to the block before the liquidation")
    if snapshot.block.hash.lower() != _hex(pre_block["hash"]):
        raise SystemExit("snapshot pre-state block hash does not match RPC")

    events = [
        decode_liquidation_call_log(log)
        for log in receipt.logs
        if _hex(log["topics"][0]) == LIQUIDATION_CALL_TOPIC
        and log["address"].lower() == snapshot.contracts.pool.lower()
        and ("0x" + log["topics"][3].hex()[-40:]).lower() == snapshot.user_address.lower()
    ]
    if len(events) != 1:
        raise SystemExit(f"expected one matching LiquidationCall, found {len(events)}")
    event = events[0]
    historical_replay = replay_liquidation_event(snapshot, event)
    near_reconciliation = reconcile_health_factor(snapshot)
    control_reconciliation = reconcile_health_factor(control)
    control_totals = calculate_portfolio_totals(control.positions)

    raw: dict[str, Any] = {
        "schema_version": 1,
        "chain_id": 1,
        "market": snapshot.market,
        "provenance": {
            "source": "Ethereum JSON-RPC receipt, transaction, and fixed-block calls",
            "explorer_url": f"https://etherscan.io/tx/{args.tx_hash}",
            "pinning_rule": "account snapshot at liquidation block minus one",
        },
        "transaction": {
            "hash": _hex(transaction["hash"]),
            "block_number": receipt.blockNumber,
            "block_hash": _hex(receipt.blockHash),
            "block_timestamp": int(block["timestamp"]),
            "transaction_index": receipt.transactionIndex,
            "status": receipt.status,
            "from": transaction["from"],
            "to": transaction["to"],
            "nonce": transaction["nonce"],
            "value_raw": transaction["value"],
            "gas_limit": transaction["gas"],
            "gas_used": receipt.gasUsed,
            "effective_gas_price_raw": receipt.effectiveGasPrice,
            "max_fee_per_gas_raw": transaction.get("maxFeePerGas"),
            "max_priority_fee_per_gas_raw": transaction.get("maxPriorityFeePerGas"),
            "type": transaction["type"],
            "input": _hex(transaction["input"]),
            "raw_signed_transaction": _hex(raw_signed_transaction),
        },
        "pre_state": {
            "snapshot_path": _relative(args.near_snapshot),
            "snapshot_sha256": _sha256(args.near_snapshot),
            "block_number": snapshot.block.number,
            "block_hash": snapshot.block.hash,
            "pool_address": snapshot.contracts.pool,
            "pool_implementation": snapshot.contracts.pool_implementation,
            "pool_revision": snapshot.contracts.pool_revision,
            "user": snapshot.user_address,
            "health_factor_wad": snapshot.account.health_factor_raw,
        },
        "event": event,
    }
    _write_json(args.evidence_output, raw)

    fork_result: dict[str, Any] | None = None
    if args.fork_report.exists():
        candidate = json.loads(args.fork_report.read_text(encoding="utf-8"))
        if (
            candidate.get("status") == "executed"
            and candidate.get("passed") is True
            and candidate.get("event_payload_exact_match") is True
            and candidate.get("historical_transaction_hash") == raw["transaction"]["hash"]
        ):
            fork_result = candidate
    fork_summary = (
        {
            "status": "passed",
            "artifact_path": _relative(args.fork_report),
            "artifact_sha256": _sha256(args.fork_report),
            "passed": True,
            "event_payload_exact_match": fork_result["event_payload_exact_match"],
            "historical_gas_used": fork_result["gas_comparison"]["historical_gas_used"],
            "fork_gas_used": fork_result["gas_comparison"]["fork_gas_used"],
            "gas_used_delta": fork_result["gas_comparison"]["gas_used_delta"],
        }
        if fork_result is not None
        else {
            "status": "not_executed_in_checked_in_report",
            "runner": "scripts/run_fork_validation.py",
            "required_runtime": "anvil plus an archive-capable Ethereum RPC",
        }
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "evidence": {
            "path": _relative(args.evidence_output),
            "sha256": _sha256(args.evidence_output),
            "transaction_hash": raw["transaction"]["hash"],
            "block_number": receipt.blockNumber,
            "block_hash": raw["transaction"]["block_hash"],
        },
        "accounts": {
            "observed_near_liquidation": {
                "is_observed_account": True,
                "snapshot_path": _relative(args.near_snapshot),
                "snapshot_sha256": _sha256(args.near_snapshot),
                "user": snapshot.user_address,
                "block_number": snapshot.block.number,
                "health_factor_wad": snapshot.account.health_factor_raw,
                "health_factor": str(snapshot.account.health_factor),
                "reconciled": near_reconciliation.within_tolerance,
            },
            "observed_high_health_control": {
                "is_observed_account": True,
                "snapshot_path": _relative(args.high_snapshot),
                "snapshot_sha256": _sha256(args.high_snapshot),
                "user": control.user_address,
                "block_number": control.block.number,
                "health_factor_wad": control.account.health_factor_raw,
                "health_factor": str(control.account.health_factor),
                "collateral_value_usd": str(control_totals.collateral_value_usd),
                "debt_value_usd": str(control_totals.debt_value_usd),
                "reconciled": control_reconciliation.within_tolerance,
            },
        },
        "historical_replay": historical_replay,
        "mainnet_fork": fork_summary,
        "limitations": [
            "The arithmetic replay is conditioned on the debt amount emitted by the event.",
            "Block-minus-one state cannot represent earlier transactions in the liquidation block.",
            "One exact event validates this reserve pair and branch, not every Aave V3.7 branch.",
            "Fork gas can differ because same-block transactions before the target are not replayed.",
        ],
        "summary": {
            "fixed_chain_evidence_verified": True,
            "near_account_reconciled": near_reconciliation.within_tolerance,
            "high_health_control_reconciled": control_reconciliation.within_tolerance,
            "historical_replay_passed": historical_replay["comparison"]["passed"],
            "mainnet_fork_executed": fork_result is not None,
        },
    }
    _write_json(args.report_output, report)
    print(json.dumps(report["summary"], indent=2))
    return 0 if historical_replay["comparison"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
