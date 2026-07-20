"""Integrity checks for the checked-in fixed-snapshot analysis evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE_ROOT = PROJECT_ROOT / "src" / "aave_risk_monitor"
SOURCE_HASH_ALGORITHM = "sha256-source-tree-v1:path-nul-size-nul-content-nul"
ARTIFACT_PATH = PROJECT_ROOT / "reports" / "fixed_snapshot_analysis.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(root: Path) -> tuple[str, tuple[str, ...]]:
    files = tuple(
        sorted(
            (path for path in root.rglob("*.py") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
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


def _checked_in_path(relative_path: str) -> Path:
    path = (PROJECT_ROOT / relative_path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise AssertionError(f"artifact path escapes the repository: {relative_path}")
    return path


def test_checked_in_fixed_analysis_artifact_integrity() -> None:
    payload: dict[str, Any] = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    inputs = payload["inputs"]
    snapshot_path = _checked_in_path(inputs["snapshot_path"])
    history_path = _checked_in_path(inputs["history_path"])
    assert _sha256(snapshot_path) == inputs["snapshot_sha256"]
    assert _sha256(history_path) == inputs["history_sha256"]

    generator = payload["provenance"]["generator"]
    generator_path = _checked_in_path(generator["path"])
    assert generator_path == PROJECT_ROOT / "scripts" / "run_fixed_analysis.py"
    assert _sha256(generator_path) == generator["sha256"]

    source_sha256, source_files = _source_tree_sha256(MODEL_SOURCE_ROOT)
    source = payload["provenance"]["source_tree"]
    assert source == {
        "root": MODEL_SOURCE_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": source_sha256,
        "algorithm": SOURCE_HASH_ALGORITHM,
        "file_count": len(source_files),
        "files": list(source_files),
    }

    runtime = payload["provenance"]["runtime"]
    assert set(runtime["python"]) == {"implementation", "version"}
    assert all(isinstance(value, str) and value for value in runtime["python"].values())
    assert set(runtime["packages"]) == {"numpy", "pandas", "sklearn"}
    assert all(isinstance(value, str) and value for value in runtime["packages"].values())

    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    scope = payload["scope"]
    assert "8-ULP guard" in scope["liquidation_numerical_boundary"]
    assert (
        scope["historical_oracle_address"]["provider_oracle_resolved_at_each_historical_block"]
        is False
    )
    assert scope["historical_oracle_address"]["csv_rows_match_snapshot_oracle"] is True
    assert scope["historical_oracle_address"]["address"].lower() == inputs["oracle"].lower()
    assert scope["emode_metadata"]["complete"] is snapshot_payload.get("emode_data_complete", False)
    assert scope["emode_metadata"]["complete"] is inputs["emode_data_complete"]
    assert scope["emode_metadata"]["user_category"] == inputs["user_emode_category"]
