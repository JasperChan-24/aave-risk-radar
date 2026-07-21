"""Focused tests for reproducible historical-price ingestion."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from aave_risk_monitor.data.history import (
    AaveOracleArchivePriceProvider,
    CsvHistoricalPriceProvider,
    HistoryDataError,
    PriceHistory,
)

ASSET_A = "0xAa00000000000000000000000000000000000001"
ASSET_B = "0xBb00000000000000000000000000000000000002"
ORACLE = "0xCc00000000000000000000000000000000000003"
DAY_1 = datetime(2025, 1, 1, tzinfo=UTC)
DAY_2 = datetime(2025, 1, 2, tzinfo=UTC)


def test_price_history_normalises_values_defaults_and_builds_frame() -> None:
    history = PriceHistory(
        asset_addresses=(ASSET_A, ASSET_B),
        timestamps=(datetime(2025, 1, 1), "2025-01-02T00:00:00Z"),
        prices_usd=(("2000.25", 1), (Decimal("2100.5"), Decimal("1.001"))),
    )

    assert history.timestamps == (DAY_1, DAY_2)
    assert history.block_numbers == (None, None)
    assert history.block_hashes == (None, None)
    assert history.prices_usd[0] == (Decimal("2000.25"), Decimal("1"))

    frame = history.to_frame()
    assert isinstance(frame, pd.DataFrame)
    assert frame.index.name == "date"
    assert list(frame.columns) == [ASSET_A.lower(), ASSET_B.lower()]
    assert frame.loc[DAY_2, ASSET_A.lower()] == pytest.approx(2100.5)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"asset_addresses": ()}, "at least one asset"),
        ({"asset_addresses": (ASSET_A, ASSET_A.lower())}, "duplicate asset"),
        ({"timestamps": (), "prices_usd": ()}, "at least one observation"),
        (
            {"timestamps": (DAY_1, DAY_2), "prices_usd": ((Decimal("1"), Decimal("2")),)},
            "equal lengths",
        ),
        ({"timestamps": (DAY_2, DAY_1)}, "timestamps must be sorted"),
        ({"timestamps": (DAY_1, DAY_1)}, "duplicate timestamps"),
        ({"prices_usd": ((Decimal("1"),), (Decimal("2"), Decimal("3")))}, "row width"),
        (
            {"prices_usd": ((Decimal("0"), Decimal("2")), (Decimal("2"), Decimal("3")))},
            "finite and positive",
        ),
        (
            {
                "prices_usd": (
                    (Decimal("NaN"), Decimal("2")),
                    (Decimal("2"), Decimal("3")),
                )
            },
            "finite and positive",
        ),
        ({"block_numbers": (1, -1)}, "non-negative"),
        ({"block_hashes": ("0x01",)}, "equal lengths"),
    ],
)
def test_price_history_rejects_ambiguous_or_invalid_inputs(
    overrides: dict[str, Any], message: str
) -> None:
    arguments: dict[str, Any] = {
        "asset_addresses": (ASSET_A, ASSET_B),
        "timestamps": (DAY_1, DAY_2),
        "prices_usd": ((Decimal("1"), Decimal("2")), (Decimal("2"), Decimal("3"))),
        "block_numbers": (1, 2),
        "block_hashes": ("0x01", "0x02"),
    }
    arguments.update(overrides)

    with pytest.raises(HistoryDataError, match=message):
        PriceHistory(**arguments)


def test_price_history_rejects_invalid_timestamp() -> None:
    with pytest.raises(HistoryDataError, match="invalid ISO-8601 timestamp"):
        PriceHistory(
            asset_addresses=(ASSET_A,),
            timestamps=("not-a-date",),
            prices_usd=((Decimal("1"),),),
        )


def test_csv_provider_maps_columns_sorts_and_filters_inclusive_dates(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    source.write_text(
        "DATE,WETH,USDC\n"
        "2025-01-03T00:00:00Z,2200,1.002\n"
        "2025-01-01T00:00:00Z,2000,1.000\n"
        "2025-01-02T00:00:00+00:00,2100,1.001\n",
        encoding="utf-8",
    )
    provider = CsvHistoricalPriceProvider(
        source,
        timestamp_column="date",
        asset_columns={ASSET_A: "weth", ASSET_B: "usdc"},
    )

    history = provider.get_history(
        (ASSET_B, ASSET_A),
        start=DAY_2,
        end="2025-01-03T00:00:00Z",
    )

    assert history.timestamps == (DAY_2, datetime(2025, 1, 3, tzinfo=UTC))
    assert history.prices_usd == (
        (Decimal("1.001"), Decimal("2100")),
        (Decimal("1.002"), Decimal("2200")),
    )


def test_csv_provider_validates_request_and_source_shape(tmp_path: Path) -> None:
    source = tmp_path / "prices.csv"
    source.write_text("date,WETH\n2025-01-01,2000\n", encoding="utf-8")
    provider = CsvHistoricalPriceProvider(source, asset_columns={ASSET_A: "WETH"})

    with pytest.raises(HistoryDataError, match="must not be empty"):
        provider.load(())
    with pytest.raises(HistoryDataError, match="start must not be after end"):
        provider.load((ASSET_A,), start=DAY_2, end=DAY_1)
    with pytest.raises(HistoryDataError, match="missing asset column"):
        provider.load((ASSET_B,))
    with pytest.raises(HistoryDataError, match="no CSV observations"):
        provider.load((ASSET_A,), start=DAY_2)

    with pytest.raises(HistoryDataError, match="cannot open historical price CSV"):
        CsvHistoricalPriceProvider(tmp_path / "missing.csv").load((ASSET_A,))


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "CSV is empty"),
        ("timestamp,WETH\n2025-01-01,2000\n", "missing timestamp column"),
        ("date,WETH\nnot-a-date,2000\n", "invalid ISO-8601 timestamp"),
        ("date,WETH\n2025-01-01,not-a-price\n", "invalid price"),
        ("date,WETH\n2025-01-01,0\n", "finite and positive"),
    ],
)
def test_csv_provider_rejects_malformed_rows(tmp_path: Path, contents: str, message: str) -> None:
    source = tmp_path / "prices.csv"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(HistoryDataError, match=message):
        CsvHistoricalPriceProvider(source, asset_columns={ASSET_A: "WETH"}).load((ASSET_A,))


class _ArchiveCall:
    def __init__(
        self, prices_by_block: dict[int, list[int]], requested_assets: list[tuple[str, ...]]
    ):
        self.prices_by_block = prices_by_block
        self.requested_assets = requested_assets
        self.assets: tuple[str, ...] = ()

    def prepare(self, assets: list[str]) -> _ArchiveCall:
        self.assets = tuple(assets)
        self.requested_assets.append(self.assets)
        return self

    def call(self, *, block_identifier: int) -> list[int]:
        return self.prices_by_block[block_identifier]


class _ArchiveFunctions:
    def __init__(self, prices_by_block: dict[int, list[int]]):
        self.call = _ArchiveCall(prices_by_block, [])

    @property
    def requested_assets(self) -> list[tuple[str, ...]]:
        return self.call.requested_assets

    def getAssetsPrices(self, assets: list[str]) -> _ArchiveCall:  # noqa: N802
        return self.call.prepare(assets)


class _ArchiveOracle:
    def __init__(self, prices_by_block: dict[int, list[int]]):
        self.functions = _ArchiveFunctions(prices_by_block)


class _ArchiveEth:
    def __init__(
        self,
        blocks: dict[int, dict[str, Any]],
        prices_by_block: dict[int, list[int]],
    ) -> None:
        self.blocks = blocks
        self.oracle = _ArchiveOracle(prices_by_block)
        self.contract_address: str | None = None

    def contract(self, *, address: str, abi: Any) -> _ArchiveOracle:
        del abi
        self.contract_address = address
        return self.oracle

    def get_block(self, block_number: int) -> dict[str, Any]:
        return self.blocks[block_number]


class _SequentialWeb3:
    def __init__(
        self,
        blocks: dict[int, dict[str, Any]],
        prices_by_block: dict[int, list[int]],
    ) -> None:
        self.eth = _ArchiveEth(blocks, prices_by_block)

    @staticmethod
    def to_checksum_address(address: str) -> str:
        return address.lower()


def _blocks() -> dict[int, dict[str, Any]]:
    return {
        10: {"timestamp": int(DAY_1.timestamp()), "hash": bytes.fromhex("10" * 32)},
        20: {"timestamp": int(DAY_2.timestamp()), "hash": bytes.fromhex("20" * 32)},
    }


def test_archive_provider_sequential_path_preserves_provenance_and_asset_order() -> None:
    web3 = _SequentialWeb3(
        _blocks(),
        {10: [200_000_000_000, 100_000_000], 20: [210_000_000_000, 100_100_000]},
    )
    provider = AaveOracleArchivePriceProvider(
        web3=web3,
        oracle_address=ORACLE,
        base_currency_unit=10**8,
        block_numbers=(20, 10),
    )

    history = provider.get_history((ASSET_A, ASSET_B))

    assert history.timestamps == (DAY_1, DAY_2)
    assert history.block_numbers == (10, 20)
    assert history.block_hashes == (f"0x{'10' * 32}", f"0x{'20' * 32}")
    assert history.prices_usd[0] == (Decimal("2000"), Decimal("1"))
    assert history.prices_usd[1] == (Decimal("2100"), Decimal("1.001"))
    assert web3.eth.contract_address == ORACLE.lower()
    assert web3.eth.oracle.functions.requested_assets[-1] == (
        ASSET_A.lower(),
        ASSET_B.lower(),
    )


def test_archive_provider_filters_dates_and_rejects_empty_result() -> None:
    provider = AaveOracleArchivePriceProvider(
        web3=_SequentialWeb3(_blocks(), {10: [100_000_000], 20: [101_000_000]}),
        oracle_address=ORACLE,
        base_currency_unit=10**8,
        block_numbers=(10, 20),
    )

    history = provider.load((ASSET_A,), start=DAY_2, end=DAY_2)
    assert history.block_numbers == (20,)

    with pytest.raises(HistoryDataError, match="no archive observations"):
        provider.load((ASSET_A,), start=DAY_2 + timedelta(days=1))


def test_archive_provider_wraps_rpc_errors_and_checks_oracle_width() -> None:
    failing = AaveOracleArchivePriceProvider(
        web3=_SequentialWeb3({}, {}),
        oracle_address=ORACLE,
        base_currency_unit=10**8,
        block_numbers=(10,),
    )
    with pytest.raises(HistoryDataError, match="archive oracle batch read failed"):
        failing.load((ASSET_A,))

    wrong_width = AaveOracleArchivePriceProvider(
        web3=_SequentialWeb3(_blocks(), {10: [100_000_000], 20: [101_000_000]}),
        oracle_address=ORACLE,
        base_currency_unit=10**8,
        block_numbers=(10,),
    )
    with pytest.raises(HistoryDataError, match="returned 1 prices for 2 assets"):
        wrong_width.load((ASSET_A, ASSET_B))


class _FakeBatch:
    def __init__(self, web3: _BatchWeb3) -> None:
        self.web3 = web3
        self.requests: list[Any] = []

    def add(self, request: Any) -> None:
        self.requests.append(request)

    def execute(self) -> list[Any]:
        if self.web3.rate_limit_failures:
            self.web3.rate_limit_failures -= 1
            raise RuntimeError("429 rate limited")
        return self.requests

    def cancel(self) -> None:
        self.web3.cancel_count += 1


class _BatchWeb3(_SequentialWeb3):
    def __init__(
        self,
        blocks: dict[int, dict[str, Any]],
        prices_by_block: dict[int, list[int]],
        *,
        rate_limit_failures: int = 0,
    ) -> None:
        super().__init__(blocks, prices_by_block)
        self.rate_limit_failures = rate_limit_failures
        self.cancel_count = 0

    def batch_requests(self) -> _FakeBatch:
        return _FakeBatch(self)


def test_archive_provider_batch_path_retries_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aave_risk_monitor.data.history.time.sleep", lambda _seconds: None)
    web3 = _BatchWeb3(
        _blocks(),
        {10: [200_000_000_000], 20: [210_000_000_000]},
        rate_limit_failures=1,
    )
    provider = AaveOracleArchivePriceProvider(
        web3=web3,
        oracle_address=ORACLE,
        base_currency_unit=10**8,
        block_numbers=(10, 20),
    )

    history = provider.load((ASSET_A,))

    assert history.block_numbers == (10, 20)
    assert web3.cancel_count == 1


@pytest.mark.parametrize(
    ("base_currency_unit", "block_numbers", "message"),
    [
        (0, (1,), "base_currency_unit must be positive"),
        (10**8, (), "non-negative blocks"),
        (10**8, (-1,), "non-negative blocks"),
        (10**8, (1, 1), "must not contain duplicates"),
    ],
)
def test_archive_provider_validates_configuration(
    base_currency_unit: int, block_numbers: tuple[int, ...], message: str
) -> None:
    with pytest.raises(HistoryDataError, match=message):
        AaveOracleArchivePriceProvider(
            web3=_SequentialWeb3({}, {}),
            oracle_address=ORACLE,
            base_currency_unit=base_currency_unit,
            block_numbers=block_numbers,
        )


def test_archive_provider_rejects_empty_asset_request() -> None:
    provider = AaveOracleArchivePriceProvider(
        web3=_SequentialWeb3(_blocks(), {}),
        oracle_address=ORACLE,
        base_currency_unit=10**8,
        block_numbers=(10,),
    )

    with pytest.raises(HistoryDataError, match="asset_addresses must not be empty"):
        provider.load(())
