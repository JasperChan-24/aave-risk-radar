"""Historical asset-price inputs with explicit UTC timestamps."""

from __future__ import annotations

import csv
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .abis import AAVE_ORACLE_ABI


class HistoryDataError(ValueError):
    """Raised when historical prices are incomplete, ambiguous or invalid."""


def _as_utc(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise HistoryDataError(f"invalid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class PriceHistory:
    """Aligned price matrix; rows are UTC observations and columns are assets."""

    asset_addresses: tuple[str, ...]
    timestamps: tuple[datetime, ...]
    prices_usd: tuple[tuple[Decimal, ...], ...]
    block_numbers: tuple[int | None, ...] = ()
    block_hashes: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_addresses", tuple(self.asset_addresses))
        object.__setattr__(self, "timestamps", tuple(_as_utc(item) for item in self.timestamps))
        object.__setattr__(
            self,
            "prices_usd",
            tuple(tuple(Decimal(value) for value in row) for row in self.prices_usd),
        )
        blocks = self.block_numbers or tuple(None for _ in self.timestamps)
        hashes = self.block_hashes or tuple(None for _ in self.timestamps)
        object.__setattr__(self, "block_numbers", tuple(blocks))
        object.__setattr__(self, "block_hashes", tuple(hashes))
        if not self.asset_addresses:
            raise HistoryDataError("price history must contain at least one asset")
        if len(set(item.lower() for item in self.asset_addresses)) != len(self.asset_addresses):
            raise HistoryDataError("price history contains duplicate asset columns")
        if not self.timestamps:
            raise HistoryDataError("price history must contain at least one observation")
        if (
            len(self.timestamps) != len(self.prices_usd)
            or len(self.timestamps) != len(blocks)
            or len(self.timestamps) != len(hashes)
        ):
            raise HistoryDataError(
                "timestamps, prices, block numbers and block hashes must have equal lengths"
            )
        if tuple(sorted(self.timestamps)) != self.timestamps:
            raise HistoryDataError("price history timestamps must be sorted")
        if len(set(self.timestamps)) != len(self.timestamps):
            raise HistoryDataError("price history contains duplicate timestamps")
        for row in self.prices_usd:
            if len(row) != len(self.asset_addresses):
                raise HistoryDataError("price row width does not match asset columns")
            if any(not price.is_finite() or price <= 0 for price in row):
                raise HistoryDataError("historical prices must be finite and positive")
        if any(block is not None and block < 0 for block in blocks):
            raise HistoryDataError("block numbers must be non-negative")

    def to_frame(self) -> Any:
        """Return a pandas frame only when callers need the simulation interface."""

        import pandas as pd

        frame = pd.DataFrame(
            [[float(value) for value in row] for row in self.prices_usd],
            index=pd.DatetimeIndex(self.timestamps, name="date"),
            columns=[address.lower() for address in self.asset_addresses],
        )
        return frame


@runtime_checkable
class HistoricalPriceProvider(Protocol):
    """Source of aligned prices for an ordered set of assets."""

    def load(
        self,
        asset_addresses: Sequence[str],
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> PriceHistory:
        """Load positive prices at UTC timestamps, preserving asset order."""


@dataclass(frozen=True, slots=True)
class CsvHistoricalPriceProvider:
    """Load a checked-in price cache (one timestamp column, one column per asset)."""

    path: str | Path
    timestamp_column: str = "date"
    asset_columns: Mapping[str, str] | None = None

    def load(
        self,
        asset_addresses: Sequence[str],
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> PriceHistory:
        assets = tuple(str(asset) for asset in asset_addresses)
        if not assets:
            raise HistoryDataError("asset_addresses must not be empty")
        start_utc = _as_utc(start) if start is not None else None
        end_utc = _as_utc(end) if end is not None else None
        if start_utc is not None and end_utc is not None and start_utc > end_utc:
            raise HistoryDataError("start must not be after end")

        mapping = {key.lower(): value for key, value in (self.asset_columns or {}).items()}
        columns = [mapping.get(asset.lower(), asset.lower()) for asset in assets]
        source = Path(self.path)
        try:
            handle = source.open("r", encoding="utf-8", newline="")
        except OSError as exc:
            raise HistoryDataError(f"cannot open historical price CSV {source}: {exc}") from exc

        observations: list[tuple[datetime, tuple[Decimal, ...]]] = []
        with handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise HistoryDataError(f"historical price CSV is empty: {source}")
            lower_to_actual = {name.lower(): name for name in reader.fieldnames}
            timestamp_column = lower_to_actual.get(self.timestamp_column.lower())
            if timestamp_column is None:
                raise HistoryDataError(f"CSV is missing timestamp column {self.timestamp_column!r}")
            actual_columns: list[str] = []
            for column in columns:
                actual = lower_to_actual.get(column.lower())
                if actual is None:
                    raise HistoryDataError(f"CSV is missing asset column {column!r}")
                actual_columns.append(actual)

            for row_number, row in enumerate(reader, start=2):
                timestamp = _as_utc(row[timestamp_column])
                if start_utc is not None and timestamp < start_utc:
                    continue
                if end_utc is not None and timestamp > end_utc:
                    continue
                try:
                    prices = tuple(Decimal(row[column]) for column in actual_columns)
                except (InvalidOperation, KeyError) as exc:
                    raise HistoryDataError(
                        f"invalid price in {source} row {row_number}: {exc}"
                    ) from exc
                observations.append((timestamp, prices))

        if not observations:
            raise HistoryDataError("no CSV observations remain after date filtering")
        observations.sort(key=lambda item: item[0])
        return PriceHistory(
            asset_addresses=assets,
            timestamps=tuple(item[0] for item in observations),
            prices_usd=tuple(item[1] for item in observations),
        )

    get_history = load


@dataclass(frozen=True, slots=True)
class AaveOracleArchivePriceProvider:
    """Sample Aave's oracle at an explicit list of historical Ethereum blocks.

    The caller supplies block numbers deliberately; timestamp-to-block search is
    outside this adapter.  The RPC endpoint must retain archive state.
    """

    web3: Any
    oracle_address: str
    base_currency_unit: int
    block_numbers: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_numbers", tuple(int(item) for item in self.block_numbers))
        if self.base_currency_unit <= 0:
            raise HistoryDataError("base_currency_unit must be positive")
        if not self.block_numbers or any(item < 0 for item in self.block_numbers):
            raise HistoryDataError("block_numbers must contain non-negative blocks")
        if len(set(self.block_numbers)) != len(self.block_numbers):
            raise HistoryDataError("block_numbers must not contain duplicates")

    def load(
        self,
        asset_addresses: Sequence[str],
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> PriceHistory:
        assets = tuple(self.web3.to_checksum_address(asset) for asset in asset_addresses)
        if not assets:
            raise HistoryDataError("asset_addresses must not be empty")
        oracle = self.web3.eth.contract(
            address=self.web3.to_checksum_address(self.oracle_address),
            abi=AAVE_ORACLE_ABI,
        )
        start_utc = _as_utc(start) if start is not None else None
        end_utc = _as_utc(end) if end is not None else None
        observations: list[tuple[datetime, tuple[Decimal, ...], int, str]] = []
        ordered_blocks = tuple(sorted(self.block_numbers))
        try:
            if hasattr(self.web3, "batch_requests"):
                batch_size = 5
                raw_observations: list[tuple[int, Any, Any]] = []
                for offset in range(0, len(ordered_blocks), batch_size):
                    chunk = ordered_blocks[offset : offset + batch_size]
                    for attempt in range(5):
                        batch = self.web3.batch_requests()
                        for block_number in chunk:
                            batch.add(self.web3.eth.get_block(block_number))
                            batch.add(
                                oracle.functions.getAssetsPrices(list(assets)).call(
                                    block_identifier=block_number
                                )
                            )
                        try:
                            responses = batch.execute()
                            break
                        except Exception as exc:
                            batch.cancel()
                            is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
                            if not is_rate_limit or attempt == 4:
                                raise
                            time.sleep(0.5 * (attempt + 1))
                    for index, block_number in enumerate(chunk):
                        raw_observations.append(
                            (block_number, responses[index * 2], responses[index * 2 + 1])
                        )
                    time.sleep(0.15)
            else:
                raw_observations = [
                    (
                        block_number,
                        self.web3.eth.get_block(block_number),
                        oracle.functions.getAssetsPrices(list(assets)).call(
                            block_identifier=block_number
                        ),
                    )
                    for block_number in ordered_blocks
                ]
        except Exception as exc:
            raise HistoryDataError(f"archive oracle batch read failed: {exc}") from exc

        for block_number, block, raw_prices in raw_observations:
            timestamp = datetime.fromtimestamp(int(block["timestamp"]), tz=UTC)
            if start_utc is not None and timestamp < start_utc:
                continue
            if end_utc is not None and timestamp > end_utc:
                continue
            if len(raw_prices) != len(assets):
                raise HistoryDataError(
                    f"oracle returned {len(raw_prices)} prices for {len(assets)} assets"
                )
            prices = tuple(
                Decimal(int(value)) / Decimal(self.base_currency_unit) for value in raw_prices
            )
            block_hash = str(block["hash"].hex())
            if not block_hash.startswith("0x"):
                block_hash = f"0x{block_hash}"
            observations.append((timestamp, prices, block_number, block_hash))
        if not observations:
            raise HistoryDataError("no archive observations remain after date filtering")
        observations.sort(key=lambda item: item[0])
        return PriceHistory(
            asset_addresses=assets,
            timestamps=tuple(item[0] for item in observations),
            prices_usd=tuple(item[1] for item in observations),
            block_numbers=tuple(item[2] for item in observations),
            block_hashes=tuple(item[3] for item in observations),
        )

    get_history = load


__all__ = [
    "AaveOracleArchivePriceProvider",
    "CsvHistoricalPriceProvider",
    "HistoricalPriceProvider",
    "HistoryDataError",
    "PriceHistory",
]
