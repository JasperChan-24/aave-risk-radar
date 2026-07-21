#!/usr/bin/env python3
"""Build a daily, common-block Aave-oracle price cache from archive blocks."""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections.abc import Sequence
from datetime import UTC
from pathlib import Path
from statistics import median
from typing import Any

from dotenv import load_dotenv
from web3 import Web3

from aave_risk_monitor.data import AaveOracleArchivePriceProvider, load_snapshot


def _positive_block(value: str) -> int:
    try:
        block = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("block numbers must be integers") from exc
    if block < 0:
        raise argparse.ArgumentTypeError("block numbers must be non-negative")
    return block


def _read_blocks(path: Path) -> tuple[int, ...]:
    values: list[int] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            values.append(_positive_block(line))
        except argparse.ArgumentTypeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not values:
        raise ValueError(f"no block numbers found in {path}")
    return tuple(values)


def _block_at_or_before_timestamp(
    web3: Any,
    *,
    target_timestamp: int,
    end_block: int,
    end_timestamp: int,
) -> int:
    """Locate a PoS-era block with a timestamp at or immediately before target."""

    candidate = max(0, min(end_block, end_block - round((end_timestamp - target_timestamp) / 12)))
    for _ in range(12):
        block = web3.eth.get_block(candidate)
        timestamp = int(block["timestamp"])
        delta = target_timestamp - timestamp
        if abs(delta) <= 24:
            break
        correction = round(delta / 12)
        if correction == 0:
            correction = 1 if delta > 0 else -1
        candidate = max(0, min(end_block, candidate + correction))

    block = web3.eth.get_block(candidate)
    timestamp = int(block["timestamp"])
    while timestamp > target_timestamp:
        step = max(1, (timestamp - target_timestamp + 11) // 12)
        candidate = max(0, candidate - step)
        block = web3.eth.get_block(candidate)
        timestamp = int(block["timestamp"])
    while candidate < end_block:
        next_block = web3.eth.get_block(candidate + 1)
        if int(next_block["timestamp"]) > target_timestamp:
            break
        candidate += 1
    return candidate


def _get_blocks_batched(web3: Any, block_numbers: Sequence[int]) -> list[Any]:
    resolved: list[Any] = []
    for offset in range(0, len(block_numbers), 10):
        chunk = block_numbers[offset : offset + 10]
        for attempt in range(5):
            batch = web3.batch_requests()
            for block_number in chunk:
                batch.add(web3.eth.get_block(block_number))
            try:
                resolved.extend(batch.execute())
                break
            except Exception as exc:
                batch.cancel()
                is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
                if not is_rate_limit or attempt == 4:
                    raise
                time.sleep(0.5 * (attempt + 1))
        time.sleep(0.15)
    return resolved


def _daily_blocks(web3: Any, *, end_block: int, days: int) -> tuple[int, ...]:
    if days < 3:
        raise ValueError("days must be at least 3")
    end = web3.eth.get_block(end_block)
    end_timestamp = int(end["timestamp"])
    targets = [end_timestamp - 86_400 * offset for offset in range(days - 1, -1, -1)]
    if hasattr(web3, "batch_requests"):
        candidates = [
            max(0, end_block - round((end_timestamp - target) / 12)) for target in targets
        ]
        seconds_per_block = 12.0
        for _ in range(10):
            requested = [
                block_number
                for candidate in candidates
                for block_number in (
                    candidate,
                    min(end_block, candidate + 1),
                )
            ]
            resolved = _get_blocks_batched(web3, requested)

            ratios = [
                (end_timestamp - int(resolved[index * 2]["timestamp"])) / (end_block - candidate)
                for index, candidate in enumerate(candidates)
                if 0 < candidate < end_block
            ]
            if ratios:
                seconds_per_block = median(ratios)
            settled = True
            for index, target in enumerate(targets):
                current_timestamp = int(resolved[index * 2]["timestamp"])
                next_timestamp = int(resolved[index * 2 + 1]["timestamp"])
                candidate = candidates[index]
                if current_timestamp <= target and (
                    candidate == end_block or next_timestamp > target
                ):
                    continue
                settled = False
                delta = target - current_timestamp
                correction = round(delta / seconds_per_block)
                if correction == 0:
                    correction = 1 if delta > 0 else -1
                candidates[index] = max(0, min(end_block, candidate + correction))
            if settled:
                break
        else:
            raise ValueError("daily block search did not converge")
        blocks = tuple(candidates)
    else:
        blocks = tuple(
            _block_at_or_before_timestamp(
                web3,
                target_timestamp=target,
                end_block=end_block,
                end_timestamp=end_timestamp,
            )
            for target in targets
        )
    if len(set(blocks)) != len(blocks):
        raise ValueError("daily block search returned duplicate blocks")
    return blocks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Fixed snapshot defining oracle, base unit, and active assets",
    )
    blocks = parser.add_mutually_exclusive_group(required=True)
    blocks.add_argument(
        "--blocks",
        nargs="+",
        type=_positive_block,
        help="Explicit daily Ethereum block numbers",
    )
    blocks.add_argument(
        "--blocks-file",
        type=Path,
        help="Text file with one daily Ethereum block number per line",
    )
    blocks.add_argument(
        "--days",
        type=int,
        help="Derive this many trailing daily blocks from --end-block/snapshot",
    )
    parser.add_argument(
        "--end-block",
        type=_positive_block,
        help="End block for --days (default: the fixed snapshot block)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv(Path.cwd() / ".env", override=False)
    rpc_url = os.getenv("ETHEREUM_ARCHIVE_RPC_URL") or os.getenv("ALCHEMY_RPC_URL")
    if not rpc_url:
        parser.error("ETHEREUM_ARCHIVE_RPC_URL (or ALCHEMY_RPC_URL) is required")

    try:
        snapshot = load_snapshot(args.snapshot)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    active = tuple(
        position.asset_address
        for position in snapshot.positions
        if position.collateral_balance_raw > 0 or position.total_debt_balance_raw > 0
    )
    if not active:
        parser.error("snapshot has no active collateral or debt assets")

    web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    if not web3.is_connected():
        parser.error("Ethereum archive RPC connection failed")
    try:
        if args.blocks is not None:
            block_numbers = tuple(args.blocks)
        elif args.blocks_file is not None:
            block_numbers = _read_blocks(args.blocks_file)
        else:
            end_block = args.end_block or snapshot.block.number
            if end_block > snapshot.block.number:
                raise ValueError("history end block cannot be after the fixed snapshot block")
            block_numbers = _daily_blocks(web3, end_block=end_block, days=args.days)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    history = AaveOracleArchivePriceProvider(
        web3=web3,
        oracle_address=snapshot.contracts.oracle,
        base_currency_unit=snapshot.base_currency_unit,
        block_numbers=block_numbers,
    ).load(active)

    rows: list[dict[str, object]] = []
    previous_date = None
    for timestamp, block_number, block_hash, prices in zip(
        history.timestamps,
        history.block_numbers,
        history.block_hashes,
        history.prices_usd,
        strict=True,
    ):
        assert block_number is not None
        date = timestamp.astimezone(UTC).date()
        if previous_date is not None and (date - previous_date).days != 1:
            parser.error(
                "archive blocks must contain exactly one observation per consecutive UTC date"
            )
        previous_date = date
        assert block_hash is not None
        row: dict[str, object] = {
            "date": date.isoformat(),
            "block_number": block_number,
            "block_hash": block_hash,
            "block_timestamp_utc": timestamp.isoformat(),
            "oracle_address": snapshot.contracts.oracle,
        }
        row.update(
            {address.lower(): str(price) for address, price in zip(active, prices, strict=True)}
        )
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} daily observations for {len(active)} assets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
