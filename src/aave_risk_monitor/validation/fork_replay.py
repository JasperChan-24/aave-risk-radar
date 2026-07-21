"""Fail-closed Anvil mainnet-fork replay for fixed liquidation evidence."""

from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from web3 import Web3

from .liquidation_replay import LIQUIDATION_CALL_TOPIC, decode_liquidation_call_log


class ForkReplayError(RuntimeError):
    """Raised when the fork cannot start or does not reproduce the fixed event."""


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def run_anvil_fork_replay(
    *,
    evidence_path: Path,
    replay_report_path: Path,
    rpc_url: str,
    anvil_command: list[str],
    startup_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Replay the original transaction on block-minus-one state and compare logs."""

    if not anvil_command:
        raise ForkReplayError("anvil command is empty")
    if not rpc_url.strip():
        raise ForkReplayError("archive RPC URL is empty")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    historical_report = json.loads(replay_report_path.read_text(encoding="utf-8"))
    transaction = evidence["transaction"]
    expected = evidence["event"]
    pre_state = evidence["pre_state"]
    port = _free_port()
    command = [
        *anvil_command,
        "--fork-url",
        rpc_url,
        "--fork-block-number",
        str(pre_state["block_number"]),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--chain-id",
        "1",
        "--quiet",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    web3 = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{port}", request_kwargs={"timeout": 30}))
    try:
        deadline = time.monotonic() + startup_timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = "" if process.stdout is None else process.stdout.read()
                raise ForkReplayError(
                    "anvil exited before RPC became ready: "
                    + output.replace(rpc_url, "<redacted>")[-2_000:]
                )
            if web3.is_connected():
                break
            time.sleep(0.2)
        else:
            raise ForkReplayError("anvil RPC did not become ready before timeout")
        if web3.eth.chain_id != 1:
            raise ForkReplayError(f"fork chain id is {web3.eth.chain_id}, expected 1")
        if web3.eth.block_number != int(pre_state["block_number"]):
            raise ForkReplayError("fork did not start at the pinned pre-state block")

        raw_signed_transaction = transaction.get("raw_signed_transaction")
        if not raw_signed_transaction:
            raise ForkReplayError("fixed evidence does not contain the raw signed transaction")
        replay_hash = web3.eth.send_raw_transaction(
            bytes.fromhex(raw_signed_transaction.removeprefix("0x"))
        )
        receipt = web3.eth.wait_for_transaction_receipt(replay_hash, timeout=120)
        if receipt["status"] != 1:
            raise ForkReplayError("original liquidation transaction reverted on the fork")
        matches = [
            decode_liquidation_call_log(log)
            for log in receipt["logs"]
            if ("0x" + log["topics"][0].hex().removeprefix("0x")) == LIQUIDATION_CALL_TOPIC
            and ("0x" + log["topics"][3].hex()[-40:]).lower() == expected["user"].lower()
        ]
        if len(matches) != 1:
            raise ForkReplayError(
                f"fork receipt contains {len(matches)} matching LiquidationCall events"
            )
        observed = matches[0]
        model = historical_report["historical_replay"]["model_result"]
        event_payload_exact_match = (
            observed["pool_address"].lower() == expected["pool_address"].lower()
            and observed["topics"] == expected["topics"]
            and observed["data"] == expected["data"]
        )
        checks = {
            "replay_transaction_hash_matches_historical": (
                "0x" + replay_hash.hex().removeprefix("0x") == transaction["hash"]
            ),
            "fork_receipt_status_is_one": receipt["status"] == 1,
            "fork_block_matches_historical_block": (
                receipt["blockNumber"] == transaction["block_number"]
            ),
            "event_topics_and_data_match_exactly": event_payload_exact_match,
            "fork_debt_matches_historical_event": (
                observed["debt_to_cover_raw"] == expected["debt_to_cover_raw"]
            ),
            "fork_collateral_matches_historical_event": (
                observed["liquidated_collateral_amount_raw"]
                == expected["liquidated_collateral_amount_raw"]
            ),
            "fork_debt_matches_model": (
                observed["debt_to_cover_raw"] == model["actual_debt_to_liquidate_raw"]
            ),
            "fork_collateral_matches_model": (
                observed["liquidated_collateral_amount_raw"]
                == model["collateral_to_liquidator_raw"]
            ),
        }
        return {
            "schema_version": 1,
            "status": "executed",
            "fork": {
                "chain_id": web3.eth.chain_id,
                "source_block_number": pre_state["block_number"],
                "source_block_hash": pre_state["block_hash"],
                "replay_transaction_hash": "0x" + replay_hash.hex().removeprefix("0x"),
                "block_number": receipt["blockNumber"],
                "receipt_status": receipt["status"],
                "gas_used": receipt["gasUsed"],
                "log_count": len(receipt["logs"]),
            },
            "historical_transaction_hash": transaction["hash"],
            "historical_event": expected,
            "fork_event": observed,
            "model_result": {
                "actual_debt_to_liquidate_raw": model["actual_debt_to_liquidate_raw"],
                "collateral_to_liquidator_raw": model["collateral_to_liquidator_raw"],
            },
            "checks": checks,
            "event_payload_exact_match": event_payload_exact_match,
            "gas_comparison": {
                "historical_gas_used": transaction["gas_used"],
                "fork_gas_used": receipt["gasUsed"],
                "gas_used_delta": receipt["gasUsed"] - transaction["gas_used"],
                "explanation": (
                    "The block-minus-one fork does not replay the target block's "
                    f"{transaction['transaction_index']} preceding transactions; warm/cold "
                    "state can therefore change gas without changing event semantics."
                ),
            },
            "passed": all(checks.values()),
        }
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


__all__ = ["ForkReplayError", "run_anvil_fork_replay"]
