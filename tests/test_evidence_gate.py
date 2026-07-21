"""Release evidence must remain content-addressed and fail closed."""

from __future__ import annotations

import importlib.util
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = PROJECT_ROOT / "scripts" / "verify_evidence.py"
SPEC = importlib.util.spec_from_file_location("verify_evidence", VERIFIER_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib platform guard
    raise RuntimeError(f"cannot load evidence verifier from {VERIFIER_PATH}")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
DEFAULT_MANIFEST = VERIFIER.DEFAULT_MANIFEST
EvidenceError = VERIFIER.EvidenceError
verify_manifest = VERIFIER.verify_manifest
verify_liquidation_event_evidence = VERIFIER._verify_liquidation_event_evidence
verify_mainnet_fork_result = VERIFIER._verify_mainnet_fork_result
verify_liquidation_replay = VERIFIER._verify_liquidation_replay


def _manifest() -> dict[str, Any]:
    return json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))


def _write_manifest(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "evidence_manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _artifact_relative_path(role: str) -> Path:
    entry = next(entry for entry in _manifest()["artifacts"] if entry["role"] == role)
    return Path(entry["path"])


def _copy_artifact(tmp_path: Path, role: str) -> Path:
    relative_path = _artifact_relative_path(role)
    destination = tmp_path / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT_ROOT / relative_path, destination)
    return destination


def test_checked_in_release_evidence_passes_gate() -> None:
    result = verify_manifest()

    assert result["status"] == "pass"
    assert result["policy"] == "content-addressed-fail-closed-v1"
    assert result["artifacts_verified"] >= 10
    assert result["trees_verified"] >= 2
    assert result["release_tag"] == "v1.0.0-research"


def test_gate_rejects_stale_artifact_hash(tmp_path: Path) -> None:
    payload = deepcopy(_manifest())
    payload["artifacts"][0]["sha256"] = "0" * 64

    with pytest.raises(EvidenceError, match="stale artifact hash"):
        verify_manifest(_write_manifest(tmp_path, payload), project_root=PROJECT_ROOT)


def test_gate_rejects_missing_screenshot(tmp_path: Path) -> None:
    payload = deepcopy(_manifest())
    screenshot = next(
        entry for entry in payload["artifacts"] if entry["role"] == "application_screenshot"
    )
    screenshot["path"] = "docs/assets/missing-release-screenshot.png"

    with pytest.raises(EvidenceError, match="required artifact is missing"):
        verify_manifest(_write_manifest(tmp_path, payload), project_root=PROJECT_ROOT)


def test_gate_rejects_stale_test_inventory(tmp_path: Path) -> None:
    payload = deepcopy(_manifest())
    test_tree = next(entry for entry in payload["trees"] if entry["id"] == "offline_test_evidence")
    test_tree["sha256"] = "f" * 64

    with pytest.raises(EvidenceError, match="stale tree hash"):
        verify_manifest(_write_manifest(tmp_path, payload), project_root=PROJECT_ROOT)


def test_gate_rejects_claim_without_evidence_role(tmp_path: Path) -> None:
    payload = deepcopy(_manifest())
    payload["claims"][0]["evidence_roles"].append("nonexistent_artifact")

    with pytest.raises(EvidenceError, match="unknown artifact roles"):
        verify_manifest(_write_manifest(tmp_path, payload), project_root=PROJECT_ROOT)


def test_gate_rejects_malformed_release_tag(tmp_path: Path) -> None:
    payload = deepcopy(_manifest())
    payload["release"]["tag"] = "latest"

    with pytest.raises(EvidenceError, match="semantic release tag"):
        verify_manifest(_write_manifest(tmp_path, payload), project_root=PROJECT_ROOT)


def test_event_verifier_rejects_wrong_prestate_block_binding(tmp_path: Path) -> None:
    evidence_path = _copy_artifact(tmp_path, "liquidation_event_evidence")
    snapshot_path = _copy_artifact(tmp_path, "liquidation_prestate_snapshot")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["transaction"]["block_number"] = evidence["pre_state"]["block_number"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(EvidenceError, match="must follow the pinned pre-state block"):
        verify_liquidation_event_evidence(
            tmp_path,
            {
                "liquidation_event_evidence": evidence_path,
                "liquidation_prestate_snapshot": snapshot_path,
            },
        )


def test_replay_verifier_rejects_one_wei_model_drift(tmp_path: Path) -> None:
    evidence_path = _copy_artifact(tmp_path, "liquidation_event_evidence")
    replay_path = _copy_artifact(tmp_path, "liquidation_replay_json")
    fixed_snapshot_path = _copy_artifact(tmp_path, "fixed_snapshot")
    prestate_snapshot_path = _copy_artifact(tmp_path, "liquidation_prestate_snapshot")
    fork_result_path = _copy_artifact(tmp_path, "mainnet_fork_result")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    comparison = replay["historical_replay"]["comparison"]
    comparison["model_collateral_to_liquidator_raw"] += 1
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    event_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    fork_result = json.loads(fork_result_path.read_text(encoding="utf-8"))

    with pytest.raises(
        EvidenceError,
        match="model_collateral_to_liquidator_raw is inconsistent",
    ):
        verify_liquidation_replay(
            tmp_path,
            {
                "liquidation_event_evidence": evidence_path,
                "liquidation_replay_json": replay_path,
                "fixed_snapshot": fixed_snapshot_path,
                "liquidation_prestate_snapshot": prestate_snapshot_path,
                "fork_validation_runner": tmp_path / "scripts" / "run_fork_validation.py",
                "mainnet_fork_result": fork_result_path,
            },
            event_evidence,
            fork_result,
        )


def test_replay_verifier_rejects_inconsistent_fork_status(tmp_path: Path) -> None:
    evidence_path = _copy_artifact(tmp_path, "liquidation_event_evidence")
    replay_path = _copy_artifact(tmp_path, "liquidation_replay_json")
    fixed_snapshot_path = _copy_artifact(tmp_path, "fixed_snapshot")
    prestate_snapshot_path = _copy_artifact(tmp_path, "liquidation_prestate_snapshot")
    fork_result_path = _copy_artifact(tmp_path, "mainnet_fork_result")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["summary"]["mainnet_fork_executed"] = True
    replay["mainnet_fork"]["status"] = "not_executed_in_checked_in_report"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    event_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    fork_result = json.loads(fork_result_path.read_text(encoding="utf-8"))

    with pytest.raises(EvidenceError, match="recorded as executed and passed"):
        verify_liquidation_replay(
            tmp_path,
            {
                "liquidation_event_evidence": evidence_path,
                "liquidation_replay_json": replay_path,
                "fixed_snapshot": fixed_snapshot_path,
                "liquidation_prestate_snapshot": prestate_snapshot_path,
                "fork_validation_runner": tmp_path / "scripts" / "run_fork_validation.py",
                "mainnet_fork_result": fork_result_path,
            },
            event_evidence,
            fork_result,
        )


def test_fork_verifier_rejects_one_wei_event_drift(tmp_path: Path) -> None:
    evidence_path = _copy_artifact(tmp_path, "liquidation_event_evidence")
    fork_result_path = _copy_artifact(tmp_path, "mainnet_fork_result")
    fork_result = json.loads(fork_result_path.read_text(encoding="utf-8"))
    fork_result["fork_event"]["debt_to_cover_raw"] += 1
    fork_result_path.write_text(json.dumps(fork_result), encoding="utf-8")
    event_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    with pytest.raises(EvidenceError, match="debt_to_cover_raw differs from history"):
        verify_mainnet_fork_result(
            {"mainnet_fork_result": fork_result_path},
            event_evidence,
        )
