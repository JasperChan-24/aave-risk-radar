#!/usr/bin/env python3
"""Replay fixed liquidation evidence on an Anvil Ethereum mainnet fork."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import tempfile
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from aave_risk_monitor.validation.fork_replay import ForkReplayError, run_anvil_fork_replay

ROOT = Path(__file__).resolve().parents[1]
TX_HASH = "0xd138a0455f087ad399820fb42f4fd35ca8f8986c223609afabf1cba1ffdab62f"
EVIDENCE = ROOT / "data" / "evidence" / f"aave_v3_ethereum_liquidation_{TX_HASH}.json"
REPLAY_REPORT = ROOT / "reports" / "liquidation_replay.json"
OUTPUT = ROOT / "reports" / "mainnet_fork_liquidation.json"


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--replay-report", type=Path, default=REPLAY_REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--anvil-command",
        default=os.getenv("ANVIL_COMMAND", "anvil"),
        help="shell-style command prefix, e.g. 'npx --yes @foundry-rs/anvil@1.7.1'",
    )
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
        print("fork validation not executed: archive RPC URL is missing")
        return 2
    try:
        result = run_anvil_fork_replay(
            evidence_path=args.evidence,
            replay_report_path=args.replay_report,
            rpc_url=rpc_url,
            anvil_command=shlex.split(args.anvil_command),
        )
    except (ForkReplayError, FileNotFoundError) as exc:
        print(f"fork validation not completed: {exc}")
        return 2
    _write(args.output, result)
    print(json.dumps({"output": str(args.output), "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
