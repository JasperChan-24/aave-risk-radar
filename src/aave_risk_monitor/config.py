"""Configuration for the Ethereum Aave V3 market and runtime services."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

ETHEREUM_CHAIN_ID = 1
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
ETHEREUM_V3_POOL_ADDRESSES_PROVIDER = "0x2f39D218133AFaB8F2B819B1066c7E434Ad94E9e"


@dataclass(frozen=True, slots=True)
class AaveV3MarketConfig:
    """The stable entrypoint for a market; other contract addresses are discovered."""

    name: str = "Aave V3 Ethereum Core"
    chain_id: int = ETHEREUM_CHAIN_ID
    addresses_provider: str = ETHEREUM_V3_POOL_ADDRESSES_PROVIDER
    expected_base_currency_address: str = ZERO_ADDRESS

    def __post_init__(self) -> None:
        if self.chain_id <= 0:
            raise ValueError("chain_id must be positive")
        if not self.addresses_provider:
            raise ValueError("addresses_provider is required")


DEFAULT_ETHEREUM_MARKET = AaveV3MarketConfig()


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings loaded by the app or CLI, never by the model layer."""

    rpc_url: str
    archive_rpc_url: str | None = None
    snapshot_directory: Path = Path("snapshots")
    hf_reconciliation_tolerance_bps: int = 5
    market: AaveV3MarketConfig = field(default_factory=AaveV3MarketConfig)

    def __post_init__(self) -> None:
        if not self.rpc_url.strip():
            raise ValueError("rpc_url is required")
        if self.hf_reconciliation_tolerance_bps < 0:
            raise ValueError("hf_reconciliation_tolerance_bps must be non-negative")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Load settings without mutating process environment or reading ``.env`` files."""

        values = os.environ if environ is None else environ
        rpc_url = (
            values.get("ETHEREUM_RPC_URL")
            or values.get("ALCHEMY_RPC_URL")
            or values.get("RPC_URL")
            or ""
        )
        provider = (
            values.get("AAVE_V3_POOL_ADDRESSES_PROVIDER")
            or values.get("AAVE_POOL_ADDRESSES_PROVIDER")
            or ETHEREUM_V3_POOL_ADDRESSES_PROVIDER
        )
        market = AaveV3MarketConfig(addresses_provider=provider)
        return cls(
            rpc_url=rpc_url,
            archive_rpc_url=values.get("ETHEREUM_ARCHIVE_RPC_URL") or None,
            snapshot_directory=Path(values.get("SNAPSHOT_DIRECTORY", "snapshots")),
            hf_reconciliation_tolerance_bps=int(values.get("HF_RECONCILIATION_TOLERANCE_BPS", "5")),
            market=market,
        )


__all__ = [
    "AaveV3MarketConfig",
    "DEFAULT_ETHEREUM_MARKET",
    "ETHEREUM_CHAIN_ID",
    "ETHEREUM_V3_POOL_ADDRESSES_PROVIDER",
    "Settings",
    "ZERO_ADDRESS",
]
