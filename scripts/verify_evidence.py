#!/usr/bin/env python3
"""Fail-closed verification for checked-in release evidence.

The evidence manifest is an intentional release attestation. Every artifact and
source tree named by the manifest is content-addressed. A model, test, report,
workflow, or screenshot change therefore requires an explicit manifest refresh
before a release can pass this gate.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import re
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "evidence_manifest.json"
TREE_HASH_ALGORITHM = "sha256-tree-v1:path-nul-size-nul-content-nul"
SHA256_LENGTH = 64
RELEASE_TAG_PATTERN = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9][a-z0-9.-]*)?")

REQUIRED_ARTIFACT_ROLES = frozenset(
    {
        "fixed_snapshot",
        "historical_prices",
        "fixed_analysis",
        "validation_report",
        "research_report",
        "application_screenshot",
        "analysis_generator",
        "evidence_verifier",
        "evidence_workflow",
        "primary_ci_workflow",
        "dependency_lock",
        "project_configuration",
        "project_readme",
        "empirical_validation",
        "empirical_validation_generator",
        "liquidation_event_evidence",
        "liquidation_prestate_snapshot",
        "liquidation_replay_json",
        "liquidation_replay_report",
        "liquidation_capture_generator",
        "fork_validation_runner",
        "mainnet_fork_result",
    }
)
REQUIRED_TREE_IDS = frozenset({"model_source", "offline_test_evidence"})
REQUIRED_CLAIM_IDS = frozenset(
    {
        "fixed_input_integrity",
        "model_validation",
        "research_scope",
        "user_interface",
        "continuous_verification",
        "empirical_convergence_and_stress",
        "liquidation_replay_and_fork",
    }
)


class EvidenceError(RuntimeError):
    """Raised when release evidence is absent, malformed, or stale."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be an object")
    return value


def _expect_non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{label} must be a non-empty string")
    return value


def _expect_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise EvidenceError(f"{label} must be a non-empty array")
    return value


def _expect_sha256(value: object, label: str) -> str:
    digest = _expect_non_empty_string(value, label)
    if len(digest) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise EvidenceError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def checked_in_path(project_root: Path, relative_path: object, label: str) -> Path:
    raw_path = _expect_non_empty_string(relative_path, label)
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise EvidenceError(f"{label} must be repository-relative: {raw_path}")
    path = (project_root / candidate).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise EvidenceError(f"{label} escapes the repository: {raw_path}") from exc
    return path


def _matches_any(relative_path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(relative_path, pattern) for pattern in patterns)


def hash_tree(
    root: Path,
    *,
    include: Sequence[str],
    exclude: Sequence[str] = (),
) -> tuple[str, tuple[str, ...]]:
    if not root.is_dir():
        raise EvidenceError(f"tree root is missing or not a directory: {root}")
    files = tuple(
        sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file()
                and _matches_any(path.relative_to(root).as_posix(), include)
                and not _matches_any(path.relative_to(root).as_posix(), exclude)
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not files:
        raise EvidenceError(f"tree has no files matching {list(include)}: {root}")

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


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    return _expect_mapping(payload, label)


def _verify_artifacts(
    project_root: Path,
    entries: list[Any],
) -> dict[str, Path]:
    paths_by_role: dict[str, Path] = {}
    seen_paths: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry = _expect_mapping(raw_entry, f"artifacts[{index}]")
        role = _expect_non_empty_string(entry.get("role"), f"artifacts[{index}].role")
        if role in paths_by_role:
            raise EvidenceError(f"duplicate artifact role: {role}")
        raw_path = _expect_non_empty_string(entry.get("path"), f"artifacts[{index}].path")
        if raw_path in seen_paths:
            raise EvidenceError(f"duplicate artifact path: {raw_path}")
        path = checked_in_path(project_root, raw_path, f"artifacts[{index}].path")
        if not path.is_file() or path.is_symlink():
            raise EvidenceError(
                f"required artifact is missing or is not a regular file: {raw_path}"
            )

        expected_bytes = entry.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes <= 0:
            raise EvidenceError(f"artifacts[{index}].bytes must be a positive integer")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise EvidenceError(
                f"stale artifact size for {raw_path}: expected {expected_bytes}, got {actual_bytes}"
            )

        expected_sha256 = _expect_sha256(entry.get("sha256"), f"artifacts[{index}].sha256")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise EvidenceError(
                f"stale artifact hash for {raw_path}: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        paths_by_role[role] = path
        seen_paths.add(raw_path)

    missing_roles = REQUIRED_ARTIFACT_ROLES - paths_by_role.keys()
    if missing_roles:
        raise EvidenceError(f"manifest is missing required artifact roles: {sorted(missing_roles)}")
    return paths_by_role


def _verify_trees(
    project_root: Path,
    entries: list[Any],
) -> dict[str, tuple[str, tuple[str, ...]]]:
    trees: dict[str, tuple[str, tuple[str, ...]]] = {}
    for index, raw_entry in enumerate(entries):
        entry = _expect_mapping(raw_entry, f"trees[{index}]")
        tree_id = _expect_non_empty_string(entry.get("id"), f"trees[{index}].id")
        if tree_id in trees:
            raise EvidenceError(f"duplicate tree id: {tree_id}")
        if entry.get("algorithm") != TREE_HASH_ALGORITHM:
            raise EvidenceError(f"trees[{index}].algorithm must be {TREE_HASH_ALGORITHM!r}")
        root = checked_in_path(project_root, entry.get("root"), f"trees[{index}].root")
        include = [
            _expect_non_empty_string(value, f"trees[{index}].include")
            for value in _expect_list(entry.get("include"), f"trees[{index}].include")
        ]
        raw_exclude = entry.get("exclude", [])
        if not isinstance(raw_exclude, list):
            raise EvidenceError(f"trees[{index}].exclude must be an array")
        exclude = [
            _expect_non_empty_string(value, f"trees[{index}].exclude") for value in raw_exclude
        ]
        actual_sha256, actual_files = hash_tree(root, include=include, exclude=exclude)
        expected_sha256 = _expect_sha256(entry.get("sha256"), f"trees[{index}].sha256")
        if actual_sha256 != expected_sha256:
            raise EvidenceError(
                f"stale tree hash for {tree_id}: expected {expected_sha256}, got {actual_sha256}"
            )
        expected_files = entry.get("files")
        if expected_files != list(actual_files):
            raise EvidenceError(f"stale file inventory for tree {tree_id}")
        expected_count = entry.get("file_count")
        if expected_count != len(actual_files):
            raise EvidenceError(
                f"stale file count for tree {tree_id}: expected {expected_count}, "
                f"got {len(actual_files)}"
            )
        trees[tree_id] = (actual_sha256, actual_files)

    missing_ids = REQUIRED_TREE_IDS - trees.keys()
    if missing_ids:
        raise EvidenceError(f"manifest is missing required source trees: {sorted(missing_ids)}")
    return trees


def _verify_png(path: Path) -> None:
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise EvidenceError(f"application screenshot is not a valid PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width < 1000 or height < 600:
        raise EvidenceError(
            f"application screenshot is too small for release evidence: {width}x{height}"
        )


def _verify_history(path: Path) -> None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise EvidenceError(f"historical evidence is not valid CSV: {path}") from exc
    if len(rows) < 365:
        raise EvidenceError(f"historical evidence must contain at least 365 observations: {path}")
    required_columns = {
        "date",
        "block_number",
        "block_hash",
        "block_timestamp_utc",
        "oracle_address",
    }
    fieldnames = set(rows[0]) if rows else set()
    missing_columns = required_columns - fieldnames
    if missing_columns:
        raise EvidenceError(
            f"historical evidence is missing provenance columns: {sorted(missing_columns)}"
        )


def _verify_report(path: Path, required_markers: Sequence[str], label: str) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"{label} is not valid UTF-8 Markdown: {path}") from exc
    if len(content) < 1000:
        raise EvidenceError(f"{label} is too short to be release evidence: {path}")
    missing_markers = [marker for marker in required_markers if marker not in content]
    if missing_markers:
        raise EvidenceError(f"{label} is missing required sections: {missing_markers}")


def _verify_readme(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceError(f"project README is not valid UTF-8 Markdown: {path}") from exc
    required_release_links = (
        "actions/workflows/ci.yml/badge.svg",
        "actions/workflows/evidence-gate.yml/badge.svg",
        "docs/assets/aave-risk-monitor.png",
        "reports/fixed_snapshot_analysis.json",
        "reports/validation_report.md",
        "reports/research_report.md",
    )
    missing_links = [link for link in required_release_links if link not in content]
    if missing_links:
        raise EvidenceError(f"project README is missing release evidence links: {missing_links}")


def _verify_fixed_analysis(
    project_root: Path,
    paths_by_role: Mapping[str, Path],
    trees: Mapping[str, tuple[str, tuple[str, ...]]],
) -> None:
    snapshot = _read_json(paths_by_role["fixed_snapshot"], "fixed snapshot")
    analysis = _read_json(paths_by_role["fixed_analysis"], "fixed analysis")
    if analysis.get("schema_version") != 2:
        raise EvidenceError("fixed analysis schema_version must be 2")

    inputs = _expect_mapping(analysis.get("inputs"), "fixed analysis inputs")
    input_bindings = (
        ("snapshot_path", "snapshot_sha256", "fixed_snapshot"),
        ("history_path", "history_sha256", "historical_prices"),
    )
    for path_key, hash_key, role in input_bindings:
        analysis_path = checked_in_path(project_root, inputs.get(path_key), f"inputs.{path_key}")
        if analysis_path != paths_by_role[role]:
            raise EvidenceError(f"fixed analysis {path_key} does not match manifest role {role}")
        if inputs.get(hash_key) != sha256_file(analysis_path):
            raise EvidenceError(f"fixed analysis has stale {hash_key}")

    block = _expect_mapping(snapshot.get("block"), "snapshot.block")
    if inputs.get("chain_id") != snapshot.get("chain_id"):
        raise EvidenceError("fixed analysis chain_id does not match snapshot")
    if inputs.get("block_number") != block.get("number"):
        raise EvidenceError("fixed analysis block number does not match snapshot")
    if inputs.get("block_hash") != block.get("hash"):
        raise EvidenceError("fixed analysis block hash does not match snapshot")

    provenance = _expect_mapping(analysis.get("provenance"), "fixed analysis provenance")
    generator = _expect_mapping(provenance.get("generator"), "fixed analysis generator")
    generator_path = checked_in_path(project_root, generator.get("path"), "generator.path")
    if generator_path != paths_by_role["analysis_generator"]:
        raise EvidenceError("fixed analysis generator does not match manifest")
    if generator.get("sha256") != sha256_file(generator_path):
        raise EvidenceError("fixed analysis generator hash is stale")

    source = _expect_mapping(provenance.get("source_tree"), "fixed analysis source tree")
    model_hash, model_files = trees["model_source"]
    if source.get("sha256") != model_hash:
        raise EvidenceError("fixed analysis model source hash is stale")
    if source.get("files") != list(model_files):
        raise EvidenceError("fixed analysis model source inventory is stale")
    if source.get("file_count") != len(model_files):
        raise EvidenceError("fixed analysis model source count is stale")

    simulation_config = _expect_mapping(
        analysis.get("simulation_config"), "fixed analysis simulation_config"
    )
    if not isinstance(simulation_config.get("seed"), int):
        raise EvidenceError("fixed analysis must record an integer simulation seed")
    if not isinstance(simulation_config.get("n_paths"), int) or simulation_config["n_paths"] < 1000:
        raise EvidenceError("fixed analysis must record at least 1,000 simulation paths")
    for section in ("calibration", "portfolio", "risk_results", "scope"):
        _expect_mapping(analysis.get(section), f"fixed analysis {section}")


def _verify_empirical_validation(
    project_root: Path,
    paths_by_role: Mapping[str, Path],
) -> None:
    payload = _read_json(paths_by_role["empirical_validation"], "empirical validation")
    if payload.get("schema_version") != 1:
        raise EvidenceError("empirical validation schema_version must be 1")

    inputs = _expect_mapping(payload.get("inputs"), "empirical validation inputs")
    input_bindings = (
        ("snapshot_path", "snapshot_sha256", "fixed_snapshot"),
        ("history_path", "history_sha256", "historical_prices"),
    )
    for path_key, hash_key, role in input_bindings:
        input_path = checked_in_path(project_root, inputs.get(path_key), f"inputs.{path_key}")
        if input_path != paths_by_role[role]:
            raise EvidenceError(f"empirical validation {path_key} does not match {role}")
        if inputs.get(hash_key) != sha256_file(input_path):
            raise EvidenceError(f"empirical validation has stale {hash_key}")

    provenance = _expect_mapping(payload.get("provenance"), "empirical validation provenance")
    generator_path = checked_in_path(
        project_root,
        provenance.get("generator_path"),
        "empirical validation generator_path",
    )
    if generator_path != paths_by_role["empirical_validation_generator"]:
        raise EvidenceError("empirical validation generator does not match manifest")
    if provenance.get("generator_sha256") != sha256_file(generator_path):
        raise EvidenceError("empirical validation generator hash is stale")

    results = _expect_mapping(payload.get("results"), "empirical validation results")
    required_results = {
        "time_step_convergence",
        "path_count_convergence",
        "calibration_window_convergence",
        "rolling_out_of_sample_var_backtest",
        "stablecoin_and_heavy_tail_stress",
    }
    missing_results = required_results - results.keys()
    if missing_results:
        raise EvidenceError(
            f"empirical validation is missing result groups: {sorted(missing_results)}"
        )

    portfolios = _expect_mapping(
        payload.get("validation_portfolios"), "empirical validation portfolios"
    )
    high_hf = _expect_mapping(
        portfolios.get("actual_fixed_snapshot"), "actual fixed-snapshot control"
    )
    near_hf = _expect_mapping(
        portfolios.get("near_boundary_counterfactual"), "near-boundary counterfactual"
    )
    high_hf_value = high_hf.get("health_factor")
    target_hf = near_hf.get("target_health_factor")
    realized_hf = near_hf.get("realized_initial_health_factor")
    if not isinstance(high_hf_value, (int, float)) or high_hf_value <= 1.5:
        raise EvidenceError("empirical validation must retain a disclosed high-HF control")
    if not isinstance(target_hf, (int, float)) or not 1.0 < target_hf <= 1.1:
        raise EvidenceError("near-boundary validation target must be in (1.0, 1.1]")
    if not isinstance(realized_hf, (int, float)) or abs(realized_hf - target_hf) > 1e-9:
        raise EvidenceError("near-boundary realized HF does not match its disclosed target")
    if near_hf.get("is_observed_account") is not False:
        raise EvidenceError("counterfactual near-boundary portfolio must be disclosed as synthetic")

    summary = _expect_mapping(payload.get("summary"), "empirical validation summary")
    required_acceptance_checks = (
        "time_step_convergence_passed",
        "path_count_convergence_passed",
        "calibration_window_convergence_passed",
        "rolling_var_backtest_passed",
        "all_statistical_acceptance_checks_passed",
    )
    non_boolean_checks = [
        name for name in required_acceptance_checks if not isinstance(summary.get(name), bool)
    ]
    if non_boolean_checks:
        raise EvidenceError(
            f"empirical validation has non-boolean acceptance results: {non_boolean_checks}"
        )
    component_checks = required_acceptance_checks[:-1]
    if summary["all_statistical_acceptance_checks_passed"] is not all(
        summary[name] for name in component_checks
    ):
        raise EvidenceError("empirical validation aggregate acceptance result is inconsistent")
    limitations = payload.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise EvidenceError("empirical validation must record limitations")


def _verify_liquidation_event_evidence(
    project_root: Path,
    paths_by_role: Mapping[str, Path],
) -> Mapping[str, Any]:
    evidence = _read_json(paths_by_role["liquidation_event_evidence"], "liquidation evidence")
    if evidence.get("schema_version") != 1:
        raise EvidenceError("liquidation evidence schema_version must be 1")
    if evidence.get("chain_id") != 1:
        raise EvidenceError("liquidation evidence must be pinned to Ethereum mainnet")

    transaction = _expect_mapping(evidence.get("transaction"), "liquidation transaction")
    pre_state = _expect_mapping(evidence.get("pre_state"), "liquidation pre_state")
    event = _expect_mapping(evidence.get("event"), "LiquidationCall event")
    snapshot_path = checked_in_path(
        project_root, pre_state.get("snapshot_path"), "liquidation pre_state snapshot_path"
    )
    if snapshot_path != paths_by_role["liquidation_prestate_snapshot"]:
        raise EvidenceError("liquidation pre-state snapshot does not match manifest")
    if pre_state.get("snapshot_sha256") != sha256_file(snapshot_path):
        raise EvidenceError("liquidation pre-state snapshot hash is stale")

    snapshot = _read_json(snapshot_path, "liquidation pre-state snapshot")
    snapshot_block = _expect_mapping(snapshot.get("block"), "liquidation snapshot block")
    snapshot_contracts = _expect_mapping(
        snapshot.get("contracts"), "liquidation snapshot contracts"
    )
    snapshot_account = _expect_mapping(snapshot.get("account"), "liquidation snapshot account")
    equality_checks = (
        ("chain id", evidence.get("chain_id"), snapshot.get("chain_id")),
        ("pre-state block number", pre_state.get("block_number"), snapshot_block.get("number")),
        ("pre-state block hash", pre_state.get("block_hash"), snapshot_block.get("hash")),
        ("Pool address", pre_state.get("pool_address"), snapshot_contracts.get("pool")),
        ("Pool revision", pre_state.get("pool_revision"), snapshot_contracts.get("pool_revision")),
        ("borrower", pre_state.get("user"), snapshot.get("user_address")),
        (
            "health factor",
            pre_state.get("health_factor_wad"),
            snapshot_account.get("health_factor_raw"),
        ),
    )
    for label, observed, expected in equality_checks:
        if isinstance(observed, str) and isinstance(expected, str):
            matches = observed.lower() == expected.lower()
        else:
            matches = observed == expected
        if not matches:
            raise EvidenceError(f"liquidation evidence {label} does not match snapshot")

    if pre_state.get("pool_revision") != 11:
        raise EvidenceError("liquidation evidence must use validated Pool revision 11 semantics")
    health_factor_wad = pre_state.get("health_factor_wad")
    if (
        not isinstance(health_factor_wad, int)
        or isinstance(health_factor_wad, bool)
        or not 950_000_000_000_000_000 < health_factor_wad < 10**18
    ):
        raise EvidenceError("liquidation pre-state must be an observed near-boundary HF<1 account")
    pre_state_block = pre_state.get("block_number")
    if not isinstance(pre_state_block, int) or isinstance(pre_state_block, bool):
        raise EvidenceError("liquidation pre-state block number must be an integer")
    transaction_block = transaction.get("block_number")
    if not isinstance(transaction_block, int) or transaction_block != pre_state_block + 1:
        raise EvidenceError("liquidation transaction must follow the pinned pre-state block")
    transaction_hash = transaction.get("hash")
    if (
        not isinstance(transaction_hash, str)
        or re.fullmatch(r"0x[0-9a-fA-F]{64}", transaction_hash) is None
    ):
        raise EvidenceError("liquidation transaction hash is malformed")
    if transaction.get("status") not in {1, "success"}:
        raise EvidenceError("liquidation transaction was not successful")

    for event_field in ("debt_to_cover_raw", "liquidated_collateral_amount_raw"):
        if not isinstance(event.get(event_field), int) or event[event_field] <= 0:
            raise EvidenceError(f"LiquidationCall {event_field} must be a positive integer")
    for event_field, pre_field in (("pool_address", "pool_address"), ("user", "user")):
        event_value = _expect_non_empty_string(
            event.get(event_field), f"LiquidationCall {event_field}"
        )
        pre_value = _expect_non_empty_string(pre_state.get(pre_field), f"pre_state {pre_field}")
        if event_value.lower() != pre_value.lower():
            raise EvidenceError(f"LiquidationCall {event_field} does not match pre-state")
    return evidence


def _verify_mainnet_fork_result(
    paths_by_role: Mapping[str, Path],
    event_evidence: Mapping[str, Any],
) -> Mapping[str, Any]:
    result = _read_json(paths_by_role["mainnet_fork_result"], "mainnet fork result")
    if result.get("schema_version") != 1:
        raise EvidenceError("mainnet fork result schema_version must be 1")
    if result.get("status") != "executed" or result.get("passed") is not True:
        raise EvidenceError("mainnet fork result must be executed and passed")

    transaction = _expect_mapping(event_evidence.get("transaction"), "liquidation transaction")
    pre_state = _expect_mapping(event_evidence.get("pre_state"), "liquidation pre-state")
    event = _expect_mapping(event_evidence.get("event"), "LiquidationCall event")
    if result.get("historical_transaction_hash") != transaction.get("hash"):
        raise EvidenceError("mainnet fork result transaction hash does not match evidence")

    fork = _expect_mapping(result.get("fork"), "mainnet fork execution")
    fork_checks = (
        ("chain_id", event_evidence.get("chain_id")),
        ("source_block_number", pre_state.get("block_number")),
        ("source_block_hash", pre_state.get("block_hash")),
        ("block_number", transaction.get("block_number")),
        ("receipt_status", 1),
    )
    for field, expected in fork_checks:
        observed = fork.get(field)
        if isinstance(observed, str) and isinstance(expected, str):
            matches = observed.lower() == expected.lower()
        else:
            matches = observed == expected
        if not matches:
            raise EvidenceError(f"mainnet fork {field} does not match pinned evidence")
    replay_transaction_hash = fork.get("replay_transaction_hash")
    historical_transaction_hash = transaction.get("hash")
    if not isinstance(replay_transaction_hash, str) or not isinstance(
        historical_transaction_hash, str
    ):
        raise EvidenceError("mainnet fork transaction hashes must be strings")
    if (
        replay_transaction_hash.removeprefix("0x").lower()
        != historical_transaction_hash.removeprefix("0x").lower()
    ):
        raise EvidenceError("fork replay transaction hash differs from the historical transaction")

    historical_event = _expect_mapping(result.get("historical_event"), "fork historical event")
    if historical_event != event:
        raise EvidenceError("mainnet fork historical event does not match raw evidence")
    fork_event = _expect_mapping(result.get("fork_event"), "fork replay event")
    for field in (
        "pool_address",
        "collateral_asset",
        "debt_asset",
        "user",
        "debt_to_cover_raw",
        "liquidated_collateral_amount_raw",
        "liquidator",
        "receive_a_token",
        "topics",
        "data",
    ):
        observed = fork_event.get(field)
        expected = event.get(field)
        if isinstance(observed, str) and isinstance(expected, str):
            matches = observed.lower() == expected.lower()
        else:
            matches = observed == expected
        if not matches:
            raise EvidenceError(f"mainnet-fork LiquidationCall {field} differs from history")

    model = _expect_mapping(result.get("model_result"), "mainnet fork model result")
    if model.get("actual_debt_to_liquidate_raw") != event.get("debt_to_cover_raw"):
        raise EvidenceError("mainnet fork debt amount differs from the model or event")
    if model.get("collateral_to_liquidator_raw") != event.get("liquidated_collateral_amount_raw"):
        raise EvidenceError("mainnet fork collateral amount differs from the model or event")
    checks = _expect_mapping(result.get("checks"), "mainnet fork checks")
    required_checks = {
        "replay_transaction_hash_matches_historical",
        "fork_receipt_status_is_one",
        "fork_block_matches_historical_block",
        "event_topics_and_data_match_exactly",
        "fork_debt_matches_historical_event",
        "fork_collateral_matches_historical_event",
        "fork_debt_matches_model",
        "fork_collateral_matches_model",
    }
    if set(checks) != required_checks or any(value is not True for value in checks.values()):
        raise EvidenceError("mainnet fork comparison checks must all pass")
    if result.get("event_payload_exact_match") is not True:
        raise EvidenceError("mainnet fork event topics and data must match history exactly")
    gas = _expect_mapping(result.get("gas_comparison"), "mainnet fork gas comparison")
    historical_gas = transaction.get("gas_used")
    fork_gas = fork.get("gas_used")
    if gas.get("historical_gas_used") != historical_gas:
        raise EvidenceError("mainnet fork historical gas does not match transaction evidence")
    if gas.get("fork_gas_used") != fork_gas:
        raise EvidenceError("mainnet fork gas does not match the replay receipt")
    if not isinstance(historical_gas, int) or not isinstance(fork_gas, int):
        raise EvidenceError("mainnet fork gas values must be integers")
    if gas.get("gas_used_delta") != fork_gas - historical_gas:
        raise EvidenceError("mainnet fork gas delta is inconsistent")
    return result


def _verify_liquidation_replay(
    project_root: Path,
    paths_by_role: Mapping[str, Path],
    event_evidence: Mapping[str, Any],
    fork_result: Mapping[str, Any],
) -> None:
    replay = _read_json(paths_by_role["liquidation_replay_json"], "liquidation replay")
    if replay.get("schema_version") != 1:
        raise EvidenceError("liquidation replay schema_version must be 1")

    evidence_reference = _expect_mapping(replay.get("evidence"), "replay evidence reference")
    evidence_path = checked_in_path(
        project_root, evidence_reference.get("path"), "replay evidence path"
    )
    if evidence_path != paths_by_role["liquidation_event_evidence"]:
        raise EvidenceError("liquidation replay evidence path does not match manifest")
    if evidence_reference.get("sha256") != sha256_file(evidence_path):
        raise EvidenceError("liquidation replay evidence hash is stale")

    transaction = _expect_mapping(event_evidence.get("transaction"), "liquidation transaction")
    event = _expect_mapping(event_evidence.get("event"), "LiquidationCall event")
    pre_state = _expect_mapping(event_evidence.get("pre_state"), "liquidation pre-state")
    evidence_checks = (
        ("transaction_hash", transaction.get("hash")),
        ("block_number", transaction.get("block_number")),
        ("block_hash", transaction.get("block_hash")),
    )
    for field, expected in evidence_checks:
        observed = evidence_reference.get(field)
        if isinstance(observed, str) and isinstance(expected, str):
            matches = observed.lower() == expected.lower()
        else:
            matches = observed == expected
        if not matches:
            raise EvidenceError(f"liquidation replay {field} does not match raw evidence")

    accounts = _expect_mapping(replay.get("accounts"), "replay accounts")
    high_account = _expect_mapping(
        accounts.get("observed_high_health_control"), "observed high-HF control"
    )
    near_account = _expect_mapping(
        accounts.get("observed_near_liquidation"), "observed near-liquidation account"
    )
    account_checks = (
        (high_account, "fixed_snapshot", True),
        (near_account, "liquidation_prestate_snapshot", False),
    )
    for account, role, is_high in account_checks:
        snapshot_path = checked_in_path(
            project_root, account.get("snapshot_path"), f"{role} account snapshot_path"
        )
        if snapshot_path != paths_by_role[role]:
            raise EvidenceError(f"replay account snapshot does not match manifest role {role}")
        if account.get("snapshot_sha256") != sha256_file(snapshot_path):
            raise EvidenceError(f"replay account snapshot hash is stale for role {role}")
        if account.get("is_observed_account") is not True or account.get("reconciled") is not True:
            raise EvidenceError(f"replay account {role} must be observed and reconciled")
        health_factor_wad = account.get("health_factor_wad")
        if not isinstance(health_factor_wad, int):
            raise EvidenceError(f"replay account {role} health_factor_wad must be an integer")
        if is_high and health_factor_wad <= 15 * 10**17:
            raise EvidenceError("observed high-HF control must have HF above 1.5")
        if not is_high and not 950_000_000_000_000_000 < health_factor_wad < 10**18:
            raise EvidenceError("observed near-liquidation account must have HF in (0.95, 1)")

    if near_account.get("health_factor_wad") != pre_state.get("health_factor_wad"):
        raise EvidenceError("near-liquidation account HF does not match raw evidence")
    if near_account.get("block_number") != pre_state.get("block_number"):
        raise EvidenceError("near-liquidation account block does not match raw evidence")

    historical = _expect_mapping(replay.get("historical_replay"), "historical replay")
    comparison = _expect_mapping(historical.get("comparison"), "historical replay comparison")
    amount_checks = (
        ("event_debt_to_cover_raw", event.get("debt_to_cover_raw")),
        (
            "event_liquidated_collateral_amount_raw",
            event.get("liquidated_collateral_amount_raw"),
        ),
        ("model_actual_debt_to_liquidate_raw", event.get("debt_to_cover_raw")),
        (
            "model_collateral_to_liquidator_raw",
            event.get("liquidated_collateral_amount_raw"),
        ),
        ("debt_delta_raw", 0),
        ("collateral_delta_raw", 0),
    )
    for field, expected in amount_checks:
        if comparison.get(field) != expected:
            raise EvidenceError(f"historical liquidation replay {field} is inconsistent")
    checks = _expect_mapping(comparison.get("checks"), "historical replay checks")
    required_checks = {
        "actual_debt_matches_event",
        "liquidator_collateral_matches_event",
        "pool_revision_11",
        "pre_state_health_factor_below_one",
    }
    if set(checks) != required_checks or any(value is not True for value in checks.values()):
        raise EvidenceError("historical liquidation replay checks must all pass")
    if comparison.get("passed") is not True:
        raise EvidenceError("historical liquidation replay comparison did not pass")

    fork = _expect_mapping(replay.get("mainnet_fork"), "mainnet fork status")
    summary = _expect_mapping(replay.get("summary"), "liquidation replay summary")
    for field in (
        "fixed_chain_evidence_verified",
        "high_health_control_reconciled",
        "historical_replay_passed",
        "near_account_reconciled",
    ):
        if summary.get(field) is not True:
            raise EvidenceError(f"liquidation replay summary check did not pass: {field}")
    if summary.get("mainnet_fork_executed") is not True or fork.get("status") != "passed":
        raise EvidenceError("checked-in mainnet fork must be recorded as executed and passed")
    fork_artifact_path = checked_in_path(
        project_root, fork.get("artifact_path"), "mainnet fork artifact_path"
    )
    if fork_artifact_path != paths_by_role["mainnet_fork_result"]:
        raise EvidenceError("mainnet fork artifact path does not match manifest")
    if fork.get("artifact_sha256") != sha256_file(fork_artifact_path):
        raise EvidenceError("mainnet fork artifact hash is stale")
    if fork.get("passed") is not True or fork_result.get("passed") is not True:
        raise EvidenceError("bound mainnet fork result did not pass")
    if fork.get("event_payload_exact_match") is not fork_result.get("event_payload_exact_match"):
        raise EvidenceError("replay report fork event-payload status is inconsistent")
    fork_execution = _expect_mapping(fork_result.get("fork"), "bound mainnet fork execution")
    gas_comparison = _expect_mapping(
        fork_result.get("gas_comparison"), "bound mainnet fork gas comparison"
    )
    report_gas_checks = (
        ("historical_gas_used", gas_comparison.get("historical_gas_used")),
        ("fork_gas_used", fork_execution.get("gas_used")),
        ("gas_used_delta", gas_comparison.get("gas_used_delta")),
    )
    for field, expected in report_gas_checks:
        if fork.get(field) != expected:
            raise EvidenceError(f"replay report mainnet fork {field} is inconsistent")
    limitations = replay.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise EvidenceError("liquidation replay must record limitations")


def _verify_claims(entries: list[Any], artifact_roles: set[str]) -> None:
    claims: dict[str, set[str]] = {}
    for index, raw_entry in enumerate(entries):
        entry = _expect_mapping(raw_entry, f"claims[{index}]")
        claim_id = _expect_non_empty_string(entry.get("id"), f"claims[{index}].id")
        if claim_id in claims:
            raise EvidenceError(f"duplicate evidence claim id: {claim_id}")
        raw_roles = _expect_list(entry.get("evidence_roles"), f"claims[{index}].evidence_roles")
        roles = {
            _expect_non_empty_string(value, f"claims[{index}].evidence_roles")
            for value in raw_roles
        }
        unknown_roles = roles - artifact_roles
        if unknown_roles:
            raise EvidenceError(
                f"claim {claim_id} references unknown artifact roles: {sorted(unknown_roles)}"
            )
        claims[claim_id] = roles
    missing_claims = REQUIRED_CLAIM_IDS - claims.keys()
    if missing_claims:
        raise EvidenceError(f"manifest is missing required claims: {sorted(missing_claims)}")


def _verify_release(manifest: Mapping[str, Any], project_root: Path, *, require_tag: bool) -> str:
    release = _expect_mapping(manifest.get("release"), "release")
    tag = _expect_non_empty_string(release.get("tag"), "release.tag")
    if RELEASE_TAG_PATTERN.fullmatch(tag) is None:
        raise EvidenceError(f"release.tag is not a supported semantic release tag: {tag}")
    if release.get("channel") not in {"prerelease", "stable"}:
        raise EvidenceError("release.channel must be 'prerelease' or 'stable'")
    evidence_date = _expect_non_empty_string(release.get("evidence_date"), "release.evidence_date")
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", evidence_date) is None:
        raise EvidenceError("release.evidence_date must use YYYY-MM-DD")
    if not require_tag:
        return tag

    try:
        tag_commit = subprocess.run(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        head_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceError(f"release tag is not available for verification: {tag}") from exc
    if tag_commit != head_commit:
        raise EvidenceError(
            f"release tag {tag} does not resolve to HEAD: tag={tag_commit}, HEAD={head_commit}"
        )
    return tag


def verify_manifest(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    project_root: Path = PROJECT_ROOT,
    require_tag: bool = False,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    manifest = _read_json(manifest_path, "evidence manifest")
    if manifest.get("schema_version") != 1:
        raise EvidenceError("evidence manifest schema_version must be 1")
    if manifest.get("policy") != "content-addressed-fail-closed-v1":
        raise EvidenceError("unsupported evidence manifest policy")
    release_tag = _verify_release(manifest, project_root, require_tag=require_tag)

    artifacts = _expect_list(manifest.get("artifacts"), "artifacts")
    trees = _expect_list(manifest.get("trees"), "trees")
    claims = _expect_list(manifest.get("claims"), "claims")
    paths_by_role = _verify_artifacts(project_root, artifacts)
    verified_trees = _verify_trees(project_root, trees)
    _verify_claims(claims, set(paths_by_role))

    _verify_png(paths_by_role["application_screenshot"])
    _verify_history(paths_by_role["historical_prices"])
    _verify_report(
        paths_by_role["validation_report"],
        (
            "## Reproducible artifacts",
            "## Implemented automated checks",
            "## Interpretation boundaries",
            "empirical_validation.json",
            "liquidation_replay.json",
            "tests/test_empirical_validation.py",
            "tests/test_liquidation_replay.py",
        ),
        "validation report",
    )
    _verify_report(
        paths_by_role["research_report"],
        (
            "## Abstract",
            "## 4. Validation status",
            "## 5. Limitations",
            "empirical_validation.json",
            "liquidation_replay",
        ),
        "research report",
    )
    _verify_report(
        paths_by_role["liquidation_replay_report"],
        ("## Result", "## Observed account controls", "## Reproduce", "## Scope limits"),
        "liquidation replay report",
    )
    _verify_readme(paths_by_role["project_readme"])
    _verify_fixed_analysis(project_root, paths_by_role, verified_trees)
    _verify_empirical_validation(project_root, paths_by_role)
    event_evidence = _verify_liquidation_event_evidence(project_root, paths_by_role)
    fork_result = _verify_mainnet_fork_result(paths_by_role, event_evidence)
    _verify_liquidation_replay(project_root, paths_by_role, event_evidence, fork_result)

    return {
        "status": "pass",
        "policy": manifest["policy"],
        "artifacts_verified": len(paths_by_role),
        "trees_verified": len(verified_trees),
        "claims_verified": len(claims),
        "release_tag": release_tag,
        "manifest": manifest_path.resolve().relative_to(project_root).as_posix(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--require-tag",
        action="store_true",
        help="require the manifest release tag to exist and resolve to HEAD",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify_manifest(
            args.manifest,
            project_root=args.root,
            require_tag=args.require_tag,
        )
    except EvidenceError as exc:
        print(f"evidence gate failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
