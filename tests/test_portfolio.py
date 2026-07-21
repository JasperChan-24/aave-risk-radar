from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from aave_risk_monitor.config import (
    ETHEREUM_V3_POOL_ADDRESSES_PROVIDER,
    Settings,
)
from aave_risk_monitor.data.history import (
    CsvHistoricalPriceProvider,
    HistoricalPriceProvider,
)
from aave_risk_monitor.data.onchain import AaveV3DataSource, AaveV3DataSourceError
from aave_risk_monitor.data.snapshots import load_snapshot, save_snapshot
from aave_risk_monitor.domain import (
    AaveContracts,
    AccountSummary,
    AssetPosition,
    BlockMetadata,
    PortfolioSnapshot,
    ReserveRiskParameters,
    calculate_health_factor,
    reconcile_health_factor,
)

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"


def _risk(*, threshold_bps: int, collateral_enabled: bool = True) -> ReserveRiskParameters:
    return ReserveRiskParameters(
        ltv_bps=max(0, threshold_bps - 500),
        liquidation_threshold_bps=threshold_bps,
        liquidation_bonus_bps=10_500,
        reserve_factor_bps=1_000,
        usage_as_collateral_enabled=collateral_enabled,
        borrowing_enabled=True,
        stable_borrow_rate_enabled=False,
        is_active=True,
        is_frozen=False,
        liquidation_protocol_fee_bps=1_000,
        is_paused=False,
        flash_loan_enabled=True,
        liquidation_grace_period_until=0,
    )


def _position(
    *,
    symbol: str,
    address: str,
    decimals: int,
    collateral_raw: int,
    debt_raw: int,
    price_raw: int,
    threshold_bps: int,
    user_uses_collateral: bool = True,
    collateral_enabled: bool = True,
) -> AssetPosition:
    return AssetPosition(
        asset_address=address,
        symbol=symbol,
        decimals=decimals,
        collateral_balance_raw=collateral_raw,
        stable_debt_balance_raw=0,
        variable_debt_balance_raw=debt_raw,
        oracle_price_raw=price_raw,
        oracle_base_currency_unit=10**8,
        user_uses_as_collateral=user_uses_collateral,
        reserve=_risk(
            threshold_bps=threshold_bps,
            collateral_enabled=collateral_enabled,
        ),
    )


def test_asset_level_health_factor_weights_each_collateral_threshold() -> None:
    positions = (
        _position(
            symbol="WETH",
            address=WETH,
            decimals=18,
            collateral_raw=10**18,
            debt_raw=0,
            price_raw=2_000 * 10**8,
            threshold_bps=8_000,
        ),
        _position(
            symbol="WBTC",
            address="0x1111111111111111111111111111111111111111",
            decimals=8,
            collateral_raw=10_000_000,
            debt_raw=0,
            price_raw=60_000 * 10**8,
            threshold_bps=7_000,
        ),
        _position(
            symbol="USDC",
            address=USDC,
            decimals=6,
            collateral_raw=0,
            debt_raw=4_000 * 10**6,
            price_raw=10**8,
            threshold_bps=8_000,
            user_uses_collateral=False,
        ),
    )

    # (2,000 * 80% + 6,000 * 70%) / 4,000 = 1.45
    assert calculate_health_factor(positions) == Decimal("1.45")


def test_disabled_collateral_is_excluded_and_zero_debt_has_no_finite_hf() -> None:
    disabled = _position(
        symbol="WETH",
        address=WETH,
        decimals=18,
        collateral_raw=10**18,
        debt_raw=1_000 * 10**18,
        price_raw=2_000 * 10**8,
        threshold_bps=8_000,
        collateral_enabled=False,
    )
    assert calculate_health_factor((disabled,)) == Decimal(0)
    assert (
        calculate_health_factor(
            (
                _position(
                    symbol="WETH",
                    address=WETH,
                    decimals=18,
                    collateral_raw=10**18,
                    debt_raw=0,
                    price_raw=2_000 * 10**8,
                    threshold_bps=8_000,
                ),
            )
        )
        is None
    )


def test_fixed_snapshot_reconciles_and_round_trips_losslessly(tmp_path: Path) -> None:
    snapshot = load_snapshot(FIXTURES / "portfolio_snapshot.json")

    reconciliation = reconcile_health_factor(snapshot)
    assert reconciliation.calculated_health_factor == Decimal("1.6")
    assert reconciliation.onchain_health_factor == Decimal("1.6")
    assert reconciliation.within_tolerance
    assert snapshot.positions[0].reserve.liquidation_protocol_fee_bps == 1_000
    assert snapshot.flashloan_premium_total_bps == 5
    assert snapshot.emode_data_complete

    legacy_payload = snapshot.to_dict()
    legacy_payload.pop("emode_data_complete")
    legacy_payload["user_emode_category"] = 1
    assert not PortfolioSnapshot.from_dict(legacy_payload).emode_data_complete

    invalid_payload = snapshot.to_dict()
    invalid_payload["emode_data_complete"] = "false"
    with pytest.raises(TypeError, match="emode_data_complete must be bool"):
        PortfolioSnapshot.from_dict(invalid_payload)

    destination = save_snapshot(snapshot, tmp_path, overwrite=False)
    assert load_snapshot(destination) == snapshot


def test_checked_in_mainnet_snapshot_is_fixed_and_reconciles() -> None:
    snapshot = load_snapshot(
        ROOT / "snapshots" / "1-ed0c6079229e2d407672a117c22b62064f4a4312-block-25573974.json"
    )

    assert snapshot.chain_id == 1
    assert snapshot.block.number == 25_573_974
    assert snapshot.block.hash == (
        "0x626780b3e20464c19f2f746710aa2e9b6256f20b326bace0fb2bed42d5b17c9d"
    )
    assert snapshot.contracts.pool_revision == 11
    assert snapshot.flashloan_premium_total_bps == 5
    assert [position.symbol for position in snapshot.positions] == ["WETH", "WBTC", "USDT"]
    assert all(position.reserve.is_paused is False for position in snapshot.positions)
    assert all(position.reserve.flash_loan_enabled is True for position in snapshot.positions)
    assert all(position.available_liquidity_raw is not None for position in snapshot.positions)
    assert all(position.a_token_total_supply_raw is not None for position in snapshot.positions)
    assert all(
        position.virtual_underlying_balance_raw is not None for position in snapshot.positions
    )
    assert snapshot.emode_data_complete
    assert reconcile_health_factor(snapshot).within_tolerance


def test_reconciliation_flags_material_mismatch() -> None:
    snapshot = load_snapshot(FIXTURES / "portfolio_snapshot.json")
    mismatched = PortfolioSnapshot(
        chain_id=snapshot.chain_id,
        market=snapshot.market,
        user_address=snapshot.user_address,
        block=snapshot.block,
        contracts=snapshot.contracts,
        base_currency_address=snapshot.base_currency_address,
        base_currency_unit=snapshot.base_currency_unit,
        positions=snapshot.positions,
        account=AccountSummary(
            total_collateral_base_raw=snapshot.account.total_collateral_base_raw,
            total_debt_base_raw=snapshot.account.total_debt_base_raw,
            available_borrows_base_raw=snapshot.account.available_borrows_base_raw,
            current_liquidation_threshold_bps=8_000,
            ltv_bps=7_500,
            health_factor_raw=15 * 10**17,
            base_currency_unit=10**8,
        ),
    )

    result = reconcile_health_factor(mismatched, tolerance_bps=5)
    assert not result.within_tolerance
    assert result.difference_bps is not None
    assert result.difference_bps > 5


def test_reconciliation_rejects_matching_hf_with_wrong_asset_level_totals() -> None:
    snapshot = load_snapshot(FIXTURES / "portfolio_snapshot.json")
    scaled_positions = tuple(
        replace(
            position,
            collateral_balance_raw=position.collateral_balance_raw // 2,
            stable_debt_balance_raw=position.stable_debt_balance_raw // 2,
            variable_debt_balance_raw=position.variable_debt_balance_raw // 2,
        )
        for position in snapshot.positions
    )

    result = reconcile_health_factor(
        replace(snapshot, positions=scaled_positions),
        tolerance_bps=5,
    )

    assert result.difference_bps == 0
    assert result.health_factor_within_tolerance
    assert result.collateral_difference_bps == 5_000
    assert result.debt_difference_bps == 5_000
    assert not result.collateral_within_tolerance
    assert not result.debt_within_tolerance
    assert not result.within_tolerance


def test_csv_history_normalises_dates_to_utc_and_preserves_asset_order() -> None:
    provider = CsvHistoricalPriceProvider(
        FIXTURES / "prices.csv",
        asset_columns={WETH: "WETH", USDC: "USDC"},
    )
    assert isinstance(provider, HistoricalPriceProvider)

    history = provider.load((USDC, WETH))

    assert history.asset_addresses == (USDC, WETH)
    assert all(timestamp.utcoffset().total_seconds() == 0 for timestamp in history.timestamps)
    assert history.prices_usd[0] == (Decimal("1.0001"), Decimal("2000"))


def test_settings_prefers_documented_v3_provider_environment_name() -> None:
    settings = Settings.from_env(
        {
            "ETHEREUM_RPC_URL": "https://example.invalid",
            "AAVE_V3_POOL_ADDRESSES_PROVIDER": "0x1111111111111111111111111111111111111111",
            "AAVE_POOL_ADDRESSES_PROVIDER": "0x2222222222222222222222222222222222222222",
        }
    )
    assert settings.market.addresses_provider.endswith("1111")
    defaults = Settings.from_env({"ETHEREUM_RPC_URL": "https://example.invalid"})
    assert defaults.market.addresses_provider == ETHEREUM_V3_POOL_ADDRESSES_PROVIDER


class _FakeCall:
    def __init__(
        self,
        web3: _FakeWeb3,
        address: str,
        name: str,
        args: tuple[Any, ...],
        result: Any,
    ) -> None:
        self.web3 = web3
        self.address = address
        self.name = name
        self.args = args
        self.result = result

    def call(self, *, block_identifier: int) -> Any:
        self.web3.calls.append((self.address, self.name, self.args, block_identifier))
        if isinstance(self.result, Exception):
            raise self.result
        if callable(self.result):
            return self.result(*self.args)
        return self.result


class _FakeFunctions:
    def __init__(self, web3: _FakeWeb3, address: str, results: dict[str, Any]) -> None:
        self.web3 = web3
        self.address = address
        self.results = results

    def __getattr__(self, name: str) -> Callable[..., _FakeCall]:
        if name not in self.results:
            raise AttributeError(name)

        def build(*args: Any) -> _FakeCall:
            return _FakeCall(self.web3, self.address, name, args, self.results[name])

        return build


class _FakeContract:
    def __init__(self, web3: _FakeWeb3, address: str, results: dict[str, Any]) -> None:
        self.functions = _FakeFunctions(web3, address, results)


class _FakeEth:
    chain_id = 1

    def __init__(self, web3: _FakeWeb3, contracts: dict[str, dict[str, Any]]) -> None:
        self.web3 = web3
        self.contracts = {address.lower(): value for address, value in contracts.items()}

    def get_block(self, identifier: Any) -> dict[str, Any]:
        assert identifier in ("latest", 123)
        return {"number": 123, "hash": bytes.fromhex("aa" * 32), "timestamp": 1_700_000_000}

    def contract(self, *, address: str, abi: Any) -> _FakeContract:
        del abi
        return _FakeContract(self.web3, address, self.contracts[address.lower()])

    def get_storage_at(
        self,
        address: str,
        position: int,
        *,
        block_identifier: int,
    ) -> bytes:
        del address, position
        self.web3.storage_block = block_identifier
        return bytes(12) + bytes.fromhex("44" * 20)


class _FakeWeb3:
    def __init__(self, contracts: dict[str, dict[str, Any]]) -> None:
        self.calls: list[tuple[str, str, tuple[Any, ...], int]] = []
        self.storage_block: int | None = None
        self.eth = _FakeEth(self, contracts)

    @staticmethod
    def to_checksum_address(address: str) -> str:
        return address


def _minimal_onchain_web3(
    *,
    get_user_emode: Any,
    ltvzero_bitmap: Any = 0,
) -> tuple[_FakeWeb3, str]:
    provider = ETHEREUM_V3_POOL_ADDRESSES_PROVIDER
    pool = "0x1111111111111111111111111111111111111111"
    oracle = "0x2222222222222222222222222222222222222222"
    data_provider = "0x3333333333333333333333333333333333333333"
    source = "0x5555555555555555555555555555555555555555"
    user = "0x7777777777777777777777777777777777777777"
    a_token = "0x8888888888888888888888888888888888888888"
    emode_active = get_user_emode == 1
    health_factor = 186 * 10**16 if emode_active else 16 * 10**17
    contracts: dict[str, dict[str, Any]] = {
        provider: {
            "getMarketId": "Aave V3 Ethereum Core",
            "getPool": pool,
            "getPriceOracle": oracle,
            "getPoolDataProvider": data_provider,
        },
        pool: {
            "POOL_REVISION": 11,
            "FLASHLOAN_PREMIUM_TOTAL": 5,
            "getLiquidationGracePeriod": 0,
            "getUserAccountData": (
                2_000 * 10**8,
                1_000 * 10**8,
                600 * 10**8,
                9_300 if emode_active else 8_000,
                9_000 if emode_active else 7_500,
                health_factor,
            ),
            "getUserEMode": get_user_emode,
            "getEModeCategoryCollateralConfig": (9_000, 9_300, 10_100),
            "getEModeCategoryCollateralBitmap": 1,
            "getEModeCategoryLtvzeroBitmap": ltvzero_bitmap,
            "getIsEModeCategoryIsolated": False,
            "getVirtualUnderlyingBalance": 50_000 * 10**18,
            "getReservesCount": 1,
            "getReserveAddressById": WETH,
        },
        oracle: {
            "BASE_CURRENCY": "0x0000000000000000000000000000000000000000",
            "BASE_CURRENCY_UNIT": 10**8,
            "getAssetPrice": 2_000 * 10**8,
            "getSourceOfAsset": source,
        },
        data_provider: {
            "getAllReservesTokens": (("WETH", WETH),),
            "getUserReserveData": (
                10**18,
                0,
                5 * 10**17,
                0,
                0,
                0,
                0,
                0,
                True,
            ),
            "getReserveConfigurationData": (
                18,
                7_500,
                8_000,
                10_500,
                1_000,
                True,
                True,
                False,
                True,
                False,
            ),
            "getLiquidationProtocolFee": 1_000,
            "getPaused": False,
            "getFlashLoanEnabled": True,
            "getReserveTokensAddresses": (
                a_token,
                "0x0000000000000000000000000000000000000000",
                "0x9999999999999999999999999999999999999999",
            ),
        },
        WETH: {"balanceOf": 50_000 * 10**18},
        a_token: {"totalSupply": 60_000 * 10**18},
        source: {"latestRoundData": (101, 2_000 * 10**8, 0, 1_699_999_900, 101)},
    }
    return _FakeWeb3(contracts), user


def test_onchain_reader_discovers_contracts_and_pins_every_call_to_one_block() -> None:
    provider = ETHEREUM_V3_POOL_ADDRESSES_PROVIDER
    pool = "0x1111111111111111111111111111111111111111"
    oracle = "0x2222222222222222222222222222222222222222"
    data_provider = "0x3333333333333333333333333333333333333333"
    weth_source = "0x5555555555555555555555555555555555555555"
    usdc_source = "0x6666666666666666666666666666666666666666"
    user = "0x7777777777777777777777777777777777777777"

    def user_reserve(asset: str, _user: str) -> tuple[Any, ...]:
        assert _user == user
        if asset == WETH:
            return (10**18, 0, 0, 0, 0, 0, 0, 0, True)
        return (0, 0, 1_000 * 10**6, 0, 0, 0, 0, 0, False)

    def reserve_config(asset: str) -> tuple[Any, ...]:
        decimals = 18 if asset == WETH else 6
        return (decimals, 7_500, 8_000, 10_500, 1_000, True, True, False, True, False)

    contracts: dict[str, dict[str, Any]] = {
        provider: {
            "getMarketId": "Aave V3 Ethereum Core",
            "getPool": pool,
            "getPriceOracle": oracle,
            "getPoolDataProvider": data_provider,
        },
        pool: {
            "POOL_REVISION": 11,
            "FLASHLOAN_PREMIUM_TOTAL": 5,
            "getLiquidationGracePeriod": 0,
            "getUserAccountData": (
                2_000 * 10**8,
                1_000 * 10**8,
                500 * 10**8,
                9_300,
                9_000,
                186 * 10**16,
            ),
            "getUserEMode": 1,
            "getEModeCategoryCollateralConfig": (9_000, 9_300, 10_100),
            "getEModeCategoryCollateralBitmap": 1,
            "getEModeCategoryLtvzeroBitmap": 0,
            "getIsEModeCategoryIsolated": False,
            "getVirtualUnderlyingBalance": lambda asset: (
                50_000 * 10**18 if asset == WETH else 25_000_000 * 10**6
            ),
            "getReservesCount": 2,
            "getReserveAddressById": lambda reserve_id: (WETH, USDC)[reserve_id],
        },
        oracle: {
            "BASE_CURRENCY": "0x0000000000000000000000000000000000000000",
            "BASE_CURRENCY_UNIT": 10**8,
            "getAssetPrice": lambda asset: 2_000 * 10**8 if asset == WETH else 10**8,
            "getSourceOfAsset": lambda asset: weth_source if asset == WETH else usdc_source,
        },
        data_provider: {
            "getAllReservesTokens": (("WETH", WETH), ("USDC", USDC)),
            "getUserReserveData": user_reserve,
            "getReserveConfigurationData": reserve_config,
            "getLiquidationProtocolFee": 1_000,
            "getPaused": False,
            "getFlashLoanEnabled": True,
            "getReserveTokensAddresses": lambda asset: (
                "0x7777777777777777777777777777777777777777"
                if asset == WETH
                else "0x8888888888888888888888888888888888888888",
                "0x0000000000000000000000000000000000000000",
                "0x9999999999999999999999999999999999999999",
            ),
        },
        WETH: {"balanceOf": 50_000 * 10**18},
        USDC: {"balanceOf": 25_000_000 * 10**6},
        user: {"totalSupply": 60_000 * 10**18},
        "0x8888888888888888888888888888888888888888": {"totalSupply": 30_000_000 * 10**6},
        weth_source: {"latestRoundData": (101, 2_000 * 10**8, 0, 1_699_999_900, 101)},
        usdc_source: {"latestRoundData": (202, 10**8, 0, 1_699_999_950, 202)},
    }
    web3 = _FakeWeb3(contracts)

    snapshot = AaveV3DataSource(web3).fetch_portfolio(user)

    assert snapshot.block.number == 123
    assert snapshot.contracts.pool == pool
    assert snapshot.contracts.oracle == oracle
    assert snapshot.contracts.pool_data_provider == data_provider
    assert snapshot.contracts.pool_implementation == "0x" + "44" * 20
    assert snapshot.contracts.pool_revision == 11
    assert snapshot.flashloan_premium_total_bps == 5
    assert snapshot.emode_data_complete
    assert len(snapshot.positions) == 2
    assert snapshot.positions[0].effective_liquidation_threshold_bps == 9_300
    assert snapshot.positions[0].effective_liquidation_bonus_bps == 10_100
    assert snapshot.positions[0].applied_liquidation_bonus_bps == 10_100
    assert snapshot.positions[0].reserve.is_paused is False
    assert snapshot.positions[0].reserve.flash_loan_enabled is True
    assert snapshot.positions[0].reserve.liquidation_grace_period_until == 0
    assert snapshot.positions[0].available_liquidity_raw == 50_000 * 10**18
    assert snapshot.positions[0].a_token_total_supply_raw == 60_000 * 10**18
    assert snapshot.positions[0].virtual_underlying_balance_raw == 50_000 * 10**18
    assert snapshot.positions[1].effective_liquidation_bonus_bps is None
    assert reconcile_health_factor(snapshot).within_tolerance
    assert web3.storage_block == 123
    assert web3.calls
    assert {call[3] for call in web3.calls} == {123}


@pytest.mark.parametrize(
    ("get_user_emode", "ltvzero_bitmap", "warning_fragment"),
    [
        (RuntimeError("RPC unavailable"), 0, "user eMode category unavailable"),
        (1, RuntimeError("method unavailable"), "eMode ltvzero bitmap unavailable"),
    ],
)
def test_onchain_reader_marks_incomplete_emode_metadata_fail_closed(
    get_user_emode: Any,
    ltvzero_bitmap: Any,
    warning_fragment: str,
) -> None:
    web3, user = _minimal_onchain_web3(
        get_user_emode=get_user_emode,
        ltvzero_bitmap=ltvzero_bitmap,
    )

    snapshot = AaveV3DataSource(web3).fetch_portfolio(user)

    assert not snapshot.emode_data_complete
    assert any(warning_fragment in warning for warning in snapshot.warnings)
    assert any("risk simulation" in warning for warning in snapshot.warnings)


class _ReorgFakeEth(_FakeEth):
    def __init__(self, web3: _FakeWeb3, contracts: dict[str, dict[str, Any]]) -> None:
        super().__init__(web3, contracts)
        self.block_reads = 0

    def get_block(self, identifier: Any) -> dict[str, Any]:
        block = super().get_block(identifier)
        self.block_reads += 1
        if self.block_reads > 1:
            block = {**block, "hash": bytes.fromhex("bb" * 32)}
        return block


def test_onchain_reader_rejects_a_block_hash_change_during_capture() -> None:
    web3, user = _minimal_onchain_web3(get_user_emode=0)
    web3.eth = _ReorgFakeEth(web3, web3.eth.contracts)

    with pytest.raises(AaveV3DataSourceError, match="hash changed"):
        AaveV3DataSource(web3).fetch_portfolio(user)


def test_public_domain_constructors_remain_straightforward() -> None:
    snapshot = PortfolioSnapshot(
        chain_id=1,
        market="test",
        user_address="0x1",
        block=BlockMetadata(1, "0x01", 1),
        contracts=AaveContracts("0x1", "0x2", "0x3", "0x4"),
        base_currency_address="0x0",
        base_currency_unit=10**8,
        positions=(),
        account=AccountSummary(0, 0, 0, 0, 0, 2**256 - 1, 10**8),
    )
    assert snapshot.positions == ()
    assert not snapshot.emode_data_complete
    assert reconcile_health_factor(snapshot).within_tolerance
