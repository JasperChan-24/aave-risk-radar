"""Offline replay of a fixed Aave ``LiquidationCall`` event.

The replay is intentionally conditioned on the debt amount emitted by the
historical event.  It validates Aave V3.7's asset-pair arithmetic (close factor,
collateral conversion, liquidation bonus, and protocol fee) against a pinned
same-block account snapshot.  It does not claim to reconstruct the liquidator's
private routing or transaction ordering inside the containing block.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from eth_abi import decode
from web3 import Web3

from aave_risk_monitor.domain import MAX_UINT256, AssetPosition, PortfolioSnapshot
from aave_risk_monitor.models.liquidation import LiquidationInputs, calculate_liquidation

LIQUIDATION_CALL_SIGNATURE = "LiquidationCall(address,address,address,uint256,uint256,address,bool)"
LIQUIDATION_CALL_TOPIC = "0x" + Web3.keccak(text=LIQUIDATION_CALL_SIGNATURE).hex().removeprefix(
    "0x"
)
SUPPORTED_POOL_REVISION = 11


class ReplayValidationError(ValueError):
    """Raised when fixed evidence is incomplete or semantically incompatible."""


def _normalise_hex(value: Any) -> str:
    if isinstance(value, str):
        payload = value
    elif hasattr(value, "hex"):
        payload = str(value.hex())
    else:
        payload = bytes(value).hex()
    return "0x" + payload.removeprefix("0x").lower()


def _topic_address(value: Any) -> str:
    payload = _normalise_hex(value)
    if len(payload) != 66:
        raise ReplayValidationError("indexed address topic must contain exactly 32 bytes")
    return Web3.to_checksum_address("0x" + payload[-40:])


def decode_liquidation_call_log(log: Any) -> dict[str, Any]:
    """Decode one canonical Aave V3 ``LiquidationCall`` receipt log."""

    topics = list(log["topics"])
    if len(topics) != 4:
        raise ReplayValidationError("LiquidationCall must contain four topics")
    if _normalise_hex(topics[0]) != LIQUIDATION_CALL_TOPIC:
        raise ReplayValidationError("receipt log is not a LiquidationCall")
    raw_data = log["data"]
    if isinstance(raw_data, str):
        data = bytes.fromhex(raw_data.removeprefix("0x"))
    else:
        data = bytes(raw_data)
    if len(data) != 128:
        raise ReplayValidationError("LiquidationCall data must contain four ABI words")
    debt_to_cover, collateral_amount, liquidator, receive_a_token = decode(
        ["uint256", "uint256", "address", "bool"], data
    )
    return {
        "pool_address": Web3.to_checksum_address(log["address"]),
        "collateral_asset": _topic_address(topics[1]),
        "debt_asset": _topic_address(topics[2]),
        "user": _topic_address(topics[3]),
        "debt_to_cover_raw": int(debt_to_cover),
        "liquidated_collateral_amount_raw": int(collateral_amount),
        "liquidator": Web3.to_checksum_address(liquidator),
        "receive_a_token": bool(receive_a_token),
        "log_index": int(log["logIndex"]),
        "topics": [_normalise_hex(topic) for topic in topics],
        "data": _normalise_hex(raw_data),
    }


def _position(snapshot: PortfolioSnapshot, address: str) -> AssetPosition:
    matches = [
        position
        for position in snapshot.positions
        if position.asset_address.lower() == address.lower()
    ]
    if len(matches) != 1:
        raise ReplayValidationError(
            f"snapshot must contain exactly one position for reserve {address}"
        )
    return matches[0]


def build_liquidation_inputs(
    snapshot: PortfolioSnapshot,
    event: dict[str, Any],
) -> LiquidationInputs:
    """Map a pinned account snapshot and event into contract-style model inputs."""

    if snapshot.contracts.pool_revision != SUPPORTED_POOL_REVISION:
        raise ReplayValidationError(
            "historical replay requires Pool revision "
            f"{SUPPORTED_POOL_REVISION}, got {snapshot.contracts.pool_revision}"
        )
    if event["pool_address"].lower() != snapshot.contracts.pool.lower():
        raise ReplayValidationError("event Pool does not match snapshot Pool")
    if event["user"].lower() != snapshot.user_address.lower():
        raise ReplayValidationError("event borrower does not match snapshot user")
    if snapshot.account.total_debt_base_raw <= 0:
        raise ReplayValidationError("snapshot account has no debt")
    if snapshot.account.health_factor_raw == MAX_UINT256:
        raise ReplayValidationError("snapshot account has an infinite health factor")
    if snapshot.user_emode_category != 0 and not snapshot.emode_data_complete:
        raise ReplayValidationError("active eMode metadata is incomplete")

    collateral = _position(snapshot, event["collateral_asset"])
    debt = _position(snapshot, event["debt_asset"])
    if not collateral.contributes_collateral:
        raise ReplayValidationError("selected collateral did not contribute at the pre-state")
    if debt.total_debt_balance_raw <= 0:
        raise ReplayValidationError("selected reserve has no borrower debt")
    if collateral.reserve.liquidation_protocol_fee_bps is None:
        raise ReplayValidationError("liquidation protocol fee is missing from snapshot")

    emode_collateral = (
        snapshot.user_emode_category != 0 and collateral.effective_liquidation_bonus_bps is not None
    )
    return LiquidationInputs(
        health_factor_wad=snapshot.account.health_factor_raw,
        total_debt_base_raw=snapshot.account.total_debt_base_raw,
        borrower_collateral_balance_raw=collateral.collateral_balance_raw,
        borrower_debt_balance_raw=debt.total_debt_balance_raw,
        collateral_price_base_raw=collateral.oracle_price_raw,
        debt_price_base_raw=debt.oracle_price_raw,
        collateral_decimals=collateral.decimals,
        debt_decimals=debt.decimals,
        # LiquidationCall emits the actual debt liquidated. Conditioning the
        # arithmetic replay on that observed amount avoids pretending the
        # event discloses a larger user-requested maximum.
        debt_to_cover_raw=int(event["debt_to_cover_raw"]),
        reserve_liquidation_bonus_bps=collateral.reserve.liquidation_bonus_bps,
        liquidation_protocol_fee_bps=collateral.reserve.liquidation_protocol_fee_bps,
        borrower_emode_category=snapshot.user_emode_category,
        collateral_enabled_in_emode=emode_collateral,
        emode_liquidation_bonus_bps=(
            collateral.effective_liquidation_bonus_bps if emode_collateral else None
        ),
    )


def replay_liquidation_event(
    snapshot: PortfolioSnapshot,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Return auditable inputs, model output, and exact event comparisons."""

    inputs = build_liquidation_inputs(snapshot, event)
    result = calculate_liquidation(inputs)
    checks = {
        "pool_revision_11": snapshot.contracts.pool_revision == SUPPORTED_POOL_REVISION,
        "pre_state_health_factor_below_one": snapshot.account.health_factor_raw < 10**18,
        "actual_debt_matches_event": (
            result.actual_debt_to_liquidate_raw == int(event["debt_to_cover_raw"])
        ),
        "liquidator_collateral_matches_event": (
            result.collateral_to_liquidator_raw == int(event["liquidated_collateral_amount_raw"])
        ),
    }
    return {
        "method": (
            "revision-11 integer liquidation arithmetic conditioned on the "
            "event-emitted debt amount"
        ),
        "model_inputs": asdict(inputs),
        "model_result": asdict(result),
        "comparison": {
            "event_debt_to_cover_raw": int(event["debt_to_cover_raw"]),
            "model_actual_debt_to_liquidate_raw": result.actual_debt_to_liquidate_raw,
            "debt_delta_raw": (
                result.actual_debt_to_liquidate_raw - int(event["debt_to_cover_raw"])
            ),
            "event_liquidated_collateral_amount_raw": int(
                event["liquidated_collateral_amount_raw"]
            ),
            "model_collateral_to_liquidator_raw": result.collateral_to_liquidator_raw,
            "collateral_delta_raw": (
                result.collateral_to_liquidator_raw - int(event["liquidated_collateral_amount_raw"])
            ),
            "model_gross_collateral_seized_raw": result.gross_collateral_seized_raw,
            "model_protocol_fee_collateral_raw": (result.liquidation_protocol_fee_collateral_raw),
            "checks": checks,
            "passed": all(checks.values()),
        },
    }


__all__ = [
    "LIQUIDATION_CALL_SIGNATURE",
    "LIQUIDATION_CALL_TOPIC",
    "ReplayValidationError",
    "SUPPORTED_POOL_REVISION",
    "build_liquidation_inputs",
    "decode_liquidation_call_log",
    "replay_liquidation_event",
]
