from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from aave_risk_monitor.data import load_snapshot
from aave_risk_monitor.validation.liquidation_replay import (
    ReplayValidationError,
    decode_liquidation_call_log,
    replay_liquidation_event,
)

ROOT = Path(__file__).resolve().parents[1]
TX_HASH = "0xd138a0455f087ad399820fb42f4fd35ca8f8986c223609afabf1cba1ffdab62f"
EVIDENCE_PATH = ROOT / "data" / "evidence" / f"aave_v3_ethereum_liquidation_{TX_HASH}.json"
REPORT_PATH = ROOT / "reports" / "liquidation_replay.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_revision_11_liquidation_replays_exactly_to_the_wei() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    snapshot_path = ROOT / evidence["pre_state"]["snapshot_path"]
    assert _sha256(snapshot_path) == evidence["pre_state"]["snapshot_sha256"]
    snapshot = load_snapshot(snapshot_path)

    replay = replay_liquidation_event(snapshot, evidence["event"])

    assert snapshot.contracts.pool_revision == 11
    assert snapshot.account.health_factor_raw == 999_565_096_476_407_496
    assert replay["comparison"]["passed"]
    assert replay["comparison"]["debt_delta_raw"] == 0
    assert replay["comparison"]["collateral_delta_raw"] == 0
    assert replay["model_result"]["actual_debt_to_liquidate_raw"] == 15_001_159
    assert replay["model_result"]["collateral_to_liquidator_raw"] == 8_495_915_601_721_874
    assert replay["model_result"]["liquidation_protocol_fee_collateral_raw"] == 40_650_313_883_837


def test_raw_log_words_decode_to_checked_in_event() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    event = evidence["event"]
    decoded = decode_liquidation_call_log(
        {
            "address": event["pool_address"],
            "topics": event["topics"],
            "data": event["data"],
            "logIndex": event["log_index"],
        }
    )
    assert decoded == event


def test_observed_near_boundary_and_high_health_accounts_are_distinct_controls() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    near = report["accounts"]["observed_near_liquidation"]
    high = report["accounts"]["observed_high_health_control"]
    assert near["is_observed_account"] is True
    assert high["is_observed_account"] is True
    assert 0.99 < float(near["health_factor"]) < 1.0
    assert float(high["health_factor"]) > 4.0
    assert near["reconciled"] is True
    assert high["reconciled"] is True
    assert _sha256(ROOT / report["evidence"]["path"]) == report["evidence"]["sha256"]
    assert report["summary"]["mainnet_fork_executed"] is True
    assert report["mainnet_fork"]["status"] == "passed"
    fork_artifact = ROOT / report["mainnet_fork"]["artifact_path"]
    assert _sha256(fork_artifact) == report["mainnet_fork"]["artifact_sha256"]


def test_revision_mismatch_is_rejected_instead_of_silently_compared() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    snapshot = load_snapshot(ROOT / evidence["pre_state"]["snapshot_path"])
    incompatible = replace(
        snapshot,
        contracts=replace(snapshot.contracts, pool_revision=10),
    )
    with pytest.raises(ReplayValidationError, match="revision 11"):
        replay_liquidation_event(incompatible, evidence["event"])
