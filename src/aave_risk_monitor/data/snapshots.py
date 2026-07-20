"""Versioned JSON persistence for fixed, reproducible account snapshots."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from aave_risk_monitor.domain import PortfolioSnapshot


class SnapshotFormatError(ValueError):
    """Raised when a snapshot file is not valid or uses an unsupported schema."""


def snapshot_filename(snapshot: PortfolioSnapshot) -> str:
    address = snapshot.user_address.lower().removeprefix("0x")
    return f"{snapshot.chain_id}-{address}-block-{snapshot.block.number}.json"


def save_snapshot(
    snapshot: PortfolioSnapshot,
    path: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    """Write a snapshot atomically and return the final path.

    If ``path`` is a directory, a deterministic filename derived from chain,
    user and block is used.
    """

    destination = Path(path)
    if (destination.exists() and destination.is_dir()) or destination.suffix == "":
        destination = destination / snapshot_filename(snapshot)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)

    payload = json.dumps(
        snapshot.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_snapshot(path: str | Path) -> PortfolioSnapshot:
    source = Path(path)
    try:
        payload: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotFormatError(f"cannot read snapshot {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SnapshotFormatError(f"snapshot root must be an object: {source}")
    try:
        return PortfolioSnapshot.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotFormatError(f"invalid snapshot {source}: {exc}") from exc


# Explicit aliases make the I/O direction discoverable for CLI callers.
read_snapshot = load_snapshot
write_snapshot = save_snapshot


__all__ = [
    "SnapshotFormatError",
    "load_snapshot",
    "read_snapshot",
    "save_snapshot",
    "snapshot_filename",
    "write_snapshot",
]
