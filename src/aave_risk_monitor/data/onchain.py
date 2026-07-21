"""Same-block Ethereum Aave V3 portfolio reader."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from aave_risk_monitor.config import DEFAULT_ETHEREUM_MARKET, AaveV3MarketConfig
from aave_risk_monitor.domain import (
    AaveContracts,
    AccountSummary,
    AssetPosition,
    BlockMetadata,
    PortfolioSnapshot,
    ReserveRiskParameters,
    reconcile_health_factor,
)

from .abis import (
    AAVE_ORACLE_ABI,
    CHAINLINK_AGGREGATOR_ABI,
    EIP1967_IMPLEMENTATION_SLOT,
    ERC20_ABI,
    POOL_ABI,
    POOL_ADDRESSES_PROVIDER_ABI,
    POOL_DATA_PROVIDER_ABI,
)


class AaveV3DataSourceError(RuntimeError):
    """Raised when a consistent portfolio snapshot cannot be produced."""


class UnsupportedBaseCurrencyError(AaveV3DataSourceError):
    """Raised when a market is not denominated in the configured base currency."""


@dataclass(frozen=True, slots=True)
class _EModeContext:
    category_id: int
    ltv_bps: int
    liquidation_threshold_bps: int
    liquidation_bonus_bps: int
    collateral_bitmap: int | None = None
    ltvzero_bitmap: int = 0
    isolated: bool = False
    reserve_ids: dict[str, int] | None = None
    legacy: bool = False
    legacy_price_source: str | None = None
    complete: bool = True


def _unwrap_tuple(value: Any) -> tuple[Any, ...]:
    """Normalise web3's representation of a single Solidity struct output."""

    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            return tuple(value[0])
        return tuple(value)
    raise TypeError(f"expected tuple-like contract result, got {type(value).__name__}")


def _hex(value: Any) -> str:
    if isinstance(value, str):
        return value if value.startswith("0x") else f"0x{value}"
    if hasattr(value, "hex"):
        result = value.hex()
        return result if str(result).startswith("0x") else f"0x{result}"
    return f"0x{bytes(value).hex()}"


class AaveV3DataSource:
    """Read an Aave V3 account with every ``eth_call`` pinned to one block."""

    def __init__(
        self,
        web3: Any,
        market: AaveV3MarketConfig = DEFAULT_ETHEREUM_MARKET,
        *,
        reconciliation_tolerance_bps: Decimal | int | str = Decimal("5"),
    ) -> None:
        self.web3 = web3
        self.market = market
        self.reconciliation_tolerance_bps = Decimal(reconciliation_tolerance_bps)
        if self.reconciliation_tolerance_bps < 0:
            raise ValueError("reconciliation_tolerance_bps must be non-negative")

    @staticmethod
    def _call(contract: Any, function_name: str, block_number: int, *args: Any) -> Any:
        function = getattr(contract.functions, function_name)(*args)
        return function.call(block_identifier=block_number)

    def _checksum(self, address: str) -> str:
        return str(self.web3.to_checksum_address(address))

    def _contract(self, address: str, abi: Sequence[dict[str, Any]]) -> Any:
        return self.web3.eth.contract(address=self._checksum(address), abi=abi)

    def fetch_portfolio(
        self,
        user_address: str,
        block_identifier: Any = "latest",
    ) -> PortfolioSnapshot:
        """Fetch a lossless per-reserve snapshot at ``block_identifier``.

        The requested block is resolved first.  Contract discovery, balances,
        reserve configuration, oracle values and aggregate account data all use
        that resolved numeric block as their explicit ``block_identifier``.
        """

        try:
            return self._fetch_portfolio(user_address, block_identifier)
        except AaveV3DataSourceError:
            raise
        except Exception as exc:
            raise AaveV3DataSourceError(f"failed to fetch Aave V3 portfolio: {exc}") from exc

    # Common alias for adapters that use generic repository terminology.
    fetch_snapshot = fetch_portfolio

    def _fetch_portfolio(
        self,
        user_address: str,
        block_identifier: Any,
    ) -> PortfolioSnapshot:
        block = self.web3.eth.get_block(block_identifier)
        block_number = int(block["number"])
        block_metadata = BlockMetadata(
            number=block_number,
            hash=_hex(block["hash"]),
            timestamp=int(block["timestamp"]),
        )
        chain_id = int(self.web3.eth.chain_id)
        if chain_id != self.market.chain_id:
            raise AaveV3DataSourceError(
                f"connected chain id {chain_id} does not match configured {self.market.chain_id}"
            )

        user = self._checksum(user_address)
        provider_address = self._checksum(self.market.addresses_provider)
        provider = self._contract(provider_address, POOL_ADDRESSES_PROVIDER_ABI)

        pool_address = self._checksum(self._call(provider, "getPool", block_number))
        oracle_address = self._checksum(self._call(provider, "getPriceOracle", block_number))
        data_provider_address = self._checksum(
            self._call(provider, "getPoolDataProvider", block_number)
        )
        if any(
            int(address, 16) == 0
            for address in (pool_address, oracle_address, data_provider_address)
        ):
            raise AaveV3DataSourceError("addresses provider returned a zero contract address")

        pool = self._contract(pool_address, POOL_ABI)
        oracle = self._contract(oracle_address, AAVE_ORACLE_ABI)
        data_provider = self._contract(data_provider_address, POOL_DATA_PROVIDER_ABI)
        warnings: list[str] = []

        try:
            market_name = str(self._call(provider, "getMarketId", block_number))
        except Exception as exc:
            market_name = self.market.name
            warnings.append(f"market id unavailable at block {block_number}: {exc}")

        base_currency_address = self._checksum(self._call(oracle, "BASE_CURRENCY", block_number))
        expected_base = self._checksum(self.market.expected_base_currency_address)
        if base_currency_address.lower() != expected_base.lower():
            raise UnsupportedBaseCurrencyError(
                f"market base currency {base_currency_address} is not configured USD base {expected_base}"
            )
        base_currency_unit = int(self._call(oracle, "BASE_CURRENCY_UNIT", block_number))
        if base_currency_unit <= 0:
            raise AaveV3DataSourceError("oracle BASE_CURRENCY_UNIT must be positive")

        pool_implementation = self._pool_implementation(pool_address, block_number, warnings)
        pool_revision = self._optional_int_call(
            pool, "POOL_REVISION", block_number, warnings, "pool revision"
        )
        flashloan_premium = self._optional_int_call(
            pool,
            "FLASHLOAN_PREMIUM_TOTAL",
            block_number,
            warnings,
            "flashloan premium",
        )

        account_raw = _unwrap_tuple(self._call(pool, "getUserAccountData", block_number, user))
        if len(account_raw) != 6:
            raise AaveV3DataSourceError(
                f"getUserAccountData returned {len(account_raw)} fields instead of 6"
            )
        account = AccountSummary(
            total_collateral_base_raw=int(account_raw[0]),
            total_debt_base_raw=int(account_raw[1]),
            available_borrows_base_raw=int(account_raw[2]),
            current_liquidation_threshold_bps=int(account_raw[3]),
            ltv_bps=int(account_raw[4]),
            health_factor_raw=int(account_raw[5]),
            base_currency_unit=base_currency_unit,
        )

        try:
            user_emode_category = int(self._call(pool, "getUserEMode", block_number, user))
            if not 0 <= user_emode_category <= 255:
                raise ValueError("user eMode category does not fit uint8")
            emode_data_complete = True
        except Exception as exc:
            user_emode_category = 0
            emode_data_complete = False
            warnings.append(f"user eMode category unavailable: {exc}")

        reserve_tokens = self._call(data_provider, "getAllReservesTokens", block_number)
        tokens = self._normalise_reserve_tokens(reserve_tokens)
        emode = self._load_emode_context(
            pool,
            data_provider,
            tokens,
            user_emode_category,
            block_number,
            warnings,
        )
        if user_emode_category != 0:
            emode_data_complete = bool(emode is not None and emode.complete)

        positions: list[AssetPosition] = []
        for symbol, asset in tokens:
            user_data = _unwrap_tuple(
                self._call(data_provider, "getUserReserveData", block_number, asset, user)
            )
            if len(user_data) != 9:
                raise AaveV3DataSourceError(
                    f"getUserReserveData({symbol}) returned {len(user_data)} fields instead of 9"
                )
            collateral_raw = int(user_data[0])
            stable_debt_raw = int(user_data[1])
            variable_debt_raw = int(user_data[2])
            if collateral_raw == stable_debt_raw == variable_debt_raw == 0:
                continue

            config = _unwrap_tuple(
                self._call(data_provider, "getReserveConfigurationData", block_number, asset)
            )
            if len(config) != 10:
                raise AaveV3DataSourceError(
                    f"getReserveConfigurationData({symbol}) returned {len(config)} fields instead of 10"
                )
            try:
                protocol_fee_bps = int(
                    self._call(
                        data_provider,
                        "getLiquidationProtocolFee",
                        block_number,
                        asset,
                    )
                )
            except Exception as exc:
                protocol_fee_bps = None
                warnings.append(f"{symbol} liquidation protocol fee unavailable: {exc}")

            is_paused = self._optional_bool_call(
                data_provider,
                "getPaused",
                block_number,
                warnings,
                f"{symbol} paused flag",
                asset,
            )
            flash_loan_enabled = self._optional_bool_call(
                data_provider,
                "getFlashLoanEnabled",
                block_number,
                warnings,
                f"{symbol} flash-loan flag",
                asset,
            )
            liquidation_grace_period_until = self._optional_int_call(
                pool,
                "getLiquidationGracePeriod",
                block_number,
                warnings,
                f"{symbol} liquidation grace period",
                asset,
            )
            (
                a_token_address,
                available_liquidity_raw,
                a_token_total_supply_raw,
                virtual_underlying_balance_raw,
            ) = self._reserve_liquidity(
                pool,
                data_provider,
                asset,
                symbol,
                block_number,
                warnings,
            )

            reserve = ReserveRiskParameters(
                ltv_bps=int(config[1]),
                liquidation_threshold_bps=int(config[2]),
                liquidation_bonus_bps=int(config[3]),
                reserve_factor_bps=int(config[4]),
                usage_as_collateral_enabled=bool(config[5]),
                borrowing_enabled=bool(config[6]),
                stable_borrow_rate_enabled=bool(config[7]),
                is_active=bool(config[8]),
                is_frozen=bool(config[9]),
                liquidation_protocol_fee_bps=protocol_fee_bps,
                is_paused=is_paused,
                flash_loan_enabled=flash_loan_enabled,
                liquidation_grace_period_until=liquidation_grace_period_until,
            )
            oracle_price_raw = int(self._call(oracle, "getAssetPrice", block_number, asset))
            if oracle_price_raw <= 0:
                raise AaveV3DataSourceError(
                    f"Aave oracle returned a non-positive price for {symbol} ({asset})"
                )

            (
                reserve_id,
                effective_ltv,
                effective_lt,
                effective_bonus,
                position_emode_complete,
            ) = self._effective_emode_parameters(
                emode,
                data_provider,
                asset,
                block_number,
                warnings,
            )
            emode_data_complete = emode_data_complete and position_emode_complete
            source, source_kind, round_id, updated_at = self._oracle_source_metadata(
                oracle,
                asset,
                symbol,
                block_number,
                warnings,
            )
            positions.append(
                AssetPosition(
                    asset_address=asset,
                    symbol=symbol,
                    decimals=int(config[0]),
                    collateral_balance_raw=collateral_raw,
                    stable_debt_balance_raw=stable_debt_raw,
                    variable_debt_balance_raw=variable_debt_raw,
                    oracle_price_raw=oracle_price_raw,
                    oracle_base_currency_unit=base_currency_unit,
                    user_uses_as_collateral=bool(user_data[8]),
                    reserve=reserve,
                    price_source_address=source,
                    price_source_kind=source_kind,
                    oracle_round_id=round_id,
                    oracle_updated_at=updated_at,
                    reserve_id=reserve_id,
                    effective_ltv_bps=effective_ltv,
                    effective_liquidation_threshold_bps=effective_lt,
                    effective_liquidation_bonus_bps=effective_bonus,
                    a_token_address=a_token_address,
                    available_liquidity_raw=available_liquidity_raw,
                    a_token_total_supply_raw=a_token_total_supply_raw,
                    virtual_underlying_balance_raw=virtual_underlying_balance_raw,
                )
            )

        if not emode_data_complete:
            warnings.append(
                "eMode metadata is incomplete; risk simulation and liquidation modelling "
                "must remain disabled for this snapshot"
            )

        snapshot = PortfolioSnapshot(
            chain_id=chain_id,
            market=market_name,
            user_address=user,
            block=block_metadata,
            contracts=AaveContracts(
                addresses_provider=provider_address,
                pool=pool_address,
                oracle=oracle_address,
                pool_data_provider=data_provider_address,
                pool_implementation=pool_implementation,
                pool_revision=pool_revision,
            ),
            base_currency_address=base_currency_address,
            base_currency_unit=base_currency_unit,
            positions=tuple(positions),
            account=account,
            user_emode_category=user_emode_category,
            emode_data_complete=emode_data_complete,
            flashloan_premium_total_bps=flashloan_premium,
            warnings=tuple(warnings),
        )
        reconciliation = reconcile_health_factor(
            snapshot, tolerance_bps=self.reconciliation_tolerance_bps
        )
        if not reconciliation.within_tolerance:

            def display_bps(value: Decimal | None) -> str:
                return "undefined" if value is None else f"{value:.4f} bps"

            snapshot = snapshot.with_warning(
                "asset-level account values do not reconcile with Pool.getUserAccountData "
                f"(HF {display_bps(reconciliation.difference_bps)}, collateral "
                f"{display_bps(reconciliation.collateral_difference_bps)}, debt "
                f"{display_bps(reconciliation.debt_difference_bps)}); inspect eMode/oracle "
                "metadata and snapshot block"
            )
        closing_block = self.web3.eth.get_block(block_number)
        closing_hash = _hex(closing_block["hash"])
        if closing_hash.lower() != block_metadata.hash.lower():
            raise AaveV3DataSourceError(
                f"block {block_number} hash changed while the snapshot was being read; "
                "discarding potentially reorged data"
            )
        return snapshot

    @staticmethod
    def _normalise_reserve_tokens(raw_tokens: Iterable[Any]) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for item in raw_tokens:
            values = _unwrap_tuple(item)
            if len(values) != 2:
                raise AaveV3DataSourceError("invalid TokenData returned by pool data provider")
            result.append((str(values[0]), str(values[1])))
        return result

    def _pool_implementation(
        self,
        pool_address: str,
        block_number: int,
        warnings: list[str],
    ) -> str | None:
        try:
            raw = self.web3.eth.get_storage_at(
                pool_address,
                EIP1967_IMPLEMENTATION_SLOT,
                block_identifier=block_number,
            )
            implementation_bytes = bytes(raw)[-20:]
            if len(implementation_bytes) != 20 or not any(implementation_bytes):
                warnings.append("pool EIP-1967 implementation slot is empty")
                return None
            return self._checksum(f"0x{implementation_bytes.hex()}")
        except Exception as exc:
            warnings.append(f"pool implementation unavailable: {exc}")
            return None

    def _optional_int_call(
        self,
        contract: Any,
        function_name: str,
        block_number: int,
        warnings: list[str],
        label: str,
        *args: Any,
    ) -> int | None:
        try:
            return int(self._call(contract, function_name, block_number, *args))
        except Exception as exc:
            warnings.append(f"{label} unavailable: {exc}")
            return None

    def _optional_bool_call(
        self,
        contract: Any,
        function_name: str,
        block_number: int,
        warnings: list[str],
        label: str,
        *args: Any,
    ) -> bool | None:
        try:
            return bool(self._call(contract, function_name, block_number, *args))
        except Exception as exc:
            warnings.append(f"{label} unavailable: {exc}")
            return None

    def _reserve_liquidity(
        self,
        pool: Any,
        data_provider: Any,
        asset: str,
        symbol: str,
        block_number: int,
        warnings: list[str],
    ) -> tuple[str | None, int | None, int | None, int | None]:
        try:
            token_addresses = _unwrap_tuple(
                self._call(
                    data_provider,
                    "getReserveTokensAddresses",
                    block_number,
                    asset,
                )
            )
            if len(token_addresses) != 3:
                raise ValueError(f"expected 3 reserve token addresses, got {len(token_addresses)}")
            a_token = self._checksum(str(token_addresses[0]))
            underlying = self._contract(asset, ERC20_ABI)
            available = int(self._call(underlying, "balanceOf", block_number, a_token))
            a_token_contract = self._contract(a_token, ERC20_ABI)
            total_supply = int(self._call(a_token_contract, "totalSupply", block_number))
            virtual_balance = int(
                self._call(
                    pool,
                    "getVirtualUnderlyingBalance",
                    block_number,
                    asset,
                )
            )
            if min(available, total_supply, virtual_balance) < 0:
                raise ValueError("reserve liquidity metadata is negative")
            return a_token, available, total_supply, virtual_balance
        except Exception as exc:
            warnings.append(f"{symbol} reserve liquidity unavailable: {exc}")
            return None, None, None, None

    def _load_emode_context(
        self,
        pool: Any,
        data_provider: Any,
        tokens: list[tuple[str, str]],
        category_id: int,
        block_number: int,
        warnings: list[str],
    ) -> _EModeContext | None:
        if category_id == 0:
            return None

        try:
            complete = True
            config = _unwrap_tuple(
                self._call(
                    pool,
                    "getEModeCategoryCollateralConfig",
                    block_number,
                    category_id,
                )
            )
            bitmap = int(
                self._call(
                    pool,
                    "getEModeCategoryCollateralBitmap",
                    block_number,
                    category_id,
                )
            )
            try:
                ltvzero_bitmap = int(
                    self._call(
                        pool,
                        "getEModeCategoryLtvzeroBitmap",
                        block_number,
                        category_id,
                    )
                )
            except Exception as exc:
                ltvzero_bitmap = 0
                complete = False
                warnings.append(f"eMode ltvzero bitmap unavailable: {exc}")
            try:
                isolated = bool(
                    self._call(
                        pool,
                        "getIsEModeCategoryIsolated",
                        block_number,
                        category_id,
                    )
                )
            except Exception as exc:
                isolated = False
                complete = False
                warnings.append(f"eMode isolated flag unavailable: {exc}")
            reserve_ids = self._load_reserve_ids(pool, block_number)
            if len(config) != 3:
                raise ValueError(f"expected 3 eMode collateral fields, got {len(config)}")
            return _EModeContext(
                category_id,
                int(config[0]),
                int(config[1]),
                int(config[2]),
                bitmap,
                ltvzero_bitmap,
                isolated,
                reserve_ids,
                complete=complete,
            )
        except Exception as modern_exc:
            try:
                legacy = _unwrap_tuple(
                    self._call(pool, "getEModeCategoryData", block_number, category_id)
                )
                if len(legacy) != 5:
                    raise ValueError(f"expected 5 legacy eMode fields, got {len(legacy)}")
                price_source = self._checksum(str(legacy[3]))
                if int(price_source, 16) != 0:
                    warnings.append(
                        "legacy eMode custom price source is recorded but asset price substitution "
                        "is not reproduced; HF reconciliation may fail"
                    )
                warnings.append(f"using legacy eMode API after modern API failed: {modern_exc}")
                return _EModeContext(
                    category_id,
                    int(legacy[0]),
                    int(legacy[1]),
                    int(legacy[2]),
                    legacy=True,
                    legacy_price_source=price_source,
                    complete=int(price_source, 16) == 0,
                )
            except Exception as legacy_exc:
                warnings.append(
                    f"eMode {category_id} parameters unavailable; modern={modern_exc}; "
                    f"legacy={legacy_exc}"
                )
                return None

    def _load_reserve_ids(self, pool: Any, block_number: int) -> dict[str, int]:
        count = int(self._call(pool, "getReservesCount", block_number))
        result: dict[str, int] = {}
        for reserve_id in range(count):
            asset = self._checksum(
                self._call(pool, "getReserveAddressById", block_number, reserve_id)
            )
            if int(asset, 16) != 0:
                result[asset.lower()] = reserve_id
        return result

    def _effective_emode_parameters(
        self,
        emode: _EModeContext | None,
        data_provider: Any,
        asset: str,
        block_number: int,
        warnings: list[str],
    ) -> tuple[int | None, int | None, int | None, int | None, bool]:
        if emode is None:
            return None, None, None, None, True
        if emode.legacy:
            try:
                reserve_category = int(
                    self._call(
                        data_provider,
                        "getReserveEModeCategory",
                        block_number,
                        asset,
                    )
                )
            except Exception as exc:
                warnings.append(f"legacy eMode category unavailable for {asset}: {exc}")
                return None, None, None, None, False
            if reserve_category == emode.category_id:
                return (
                    None,
                    emode.ltv_bps,
                    emode.liquidation_threshold_bps,
                    emode.liquidation_bonus_bps,
                    True,
                )
            return None, None, None, None, True

        reserve_ids = emode.reserve_ids or {}
        reserve_id = reserve_ids.get(asset.lower())
        if reserve_id is None:
            warnings.append(f"reserve id unavailable for eMode asset {asset}")
            return None, None, None, None, False
        eligible = bool((int(emode.collateral_bitmap or 0) >> reserve_id) & 1)
        if eligible:
            effective_ltv = 0 if bool((emode.ltvzero_bitmap >> reserve_id) & 1) else emode.ltv_bps
            return (
                reserve_id,
                effective_ltv,
                emode.liquidation_threshold_bps,
                emode.liquidation_bonus_bps,
                True,
            )
        if emode.isolated:
            return reserve_id, 0, None, None, True
        return reserve_id, None, None, None, True

    def _oracle_source_metadata(
        self,
        oracle: Any,
        asset: str,
        symbol: str,
        block_number: int,
        warnings: list[str],
    ) -> tuple[str | None, str | None, int | None, int | None]:
        try:
            source = self._checksum(self._call(oracle, "getSourceOfAsset", block_number, asset))
        except Exception as exc:
            warnings.append(f"{symbol} oracle source unavailable: {exc}")
            return None, "aave_oracle", None, None
        if int(source, 16) == 0:
            warnings.append(f"{symbol} has no asset-specific Aave oracle source")
            return None, "aave_oracle_fallback", None, None

        aggregator = self._contract(source, CHAINLINK_AGGREGATOR_ABI)
        try:
            round_data = _unwrap_tuple(self._call(aggregator, "latestRoundData", block_number))
            if len(round_data) != 5:
                raise ValueError(f"expected 5 Chainlink round fields, got {len(round_data)}")
            if int(round_data[1]) <= 0:
                warnings.append(
                    f"{symbol} AggregatorV3-compatible source returned a non-positive answer"
                )
            return (
                source,
                "aggregator_v3_compatible",
                int(round_data[0]),
                int(round_data[3]),
            )
        except Exception as exc:
            warnings.append(
                f"{symbol} oracle source does not expose AggregatorV3 latestRoundData: {exc}"
            )
            return source, "aave_oracle_adapter", None, None


__all__ = [
    "AaveV3DataSource",
    "AaveV3DataSourceError",
    "UnsupportedBaseCurrencyError",
]
