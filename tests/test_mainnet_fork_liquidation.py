from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_fork_validation.py"


def test_fork_runner_fails_closed_when_anvil_is_unavailable(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--anvil-command",
            str(tmp_path / "missing-anvil"),
            "--output",
            str(tmp_path / "must-not-exist.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "not completed" in result.stdout
    assert not (tmp_path / "must-not-exist.json").exists()


@pytest.mark.integration
def test_original_transaction_matches_model_on_mainnet_fork(tmp_path: Path) -> None:
    if os.getenv("RUN_MAINNET_FORK") != "1":
        pytest.skip("set RUN_MAINNET_FORK=1 with ANVIL_COMMAND and archive RPC")
    command = shlex.split(os.getenv("ANVIL_COMMAND", "anvil"))
    if not command:
        pytest.fail("ANVIL_COMMAND is empty")
    output = tmp_path / "fork.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--anvil-command",
            shlex.join(command),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "executed"
    assert payload["passed"] is True
    assert all(payload["checks"].values())
