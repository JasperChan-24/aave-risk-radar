"""Domain types and pure portfolio accounting for Aave V3.

Raw token balances, oracle prices and basis-point parameters are retained in the
snapshot.  Human-readable values are exposed as :class:`~decimal.Decimal`
properties so a snapshot can be audited without reconstructing integers from
rounded floats.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal, localcontext
from typing import Any

BPS = Decimal(10_000)
WAD = 10**18
MAX_UINT256 = 2**256 - 1


def _decimal_ratio(numerator: int, denominator: int) -> Decimal:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    with localcontext() as context:
        context.prec = 60
        return Decimal(numerator) / Decimal(denominator)


def _require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_bps(name: str, value: int) -> None:
    if not 0 <= value <= 10_000:
        raise ValueError(f"{name} must be between 0 and 10,000 bps")


@dataclass(frozen=True, slots=True)
class ReserveRiskParameters:
    """Asset-level Aave reserve parameters, in their lossless on-chain units.

    ``liquidation_bonus_bps`` is Aave's multiplier representation: ``10_500``
    means the liquidator receives principal plus a 5% bonus.
    """

    ltv_bps: int
    liquidation_threshold_bps: int
    liquidation_bonus_bps: int
    reserve_factor_bps: int
    usage_as_collateral_enabled: bool
    borrowing_enabled: bool
    stable_borrow_rate_enabled: bool
    is_active: bool
    is_frozen: bool
    liquidation_protocol_fee_bps: int | None = None
    is_paused: bool | None = None
    flash_loan_enabled: bool | None = None
    liquidation_grace_period_until: int | None = None

    def __post_init__(self) -> None:
        _require_bps("ltv_bps", self.ltv_bps)
        _require_bps("liquidation_threshold_bps", self.liquidation_threshold_bps)
        if self.liquidation_bonus_bps != 0 and self.liquidation_bonus_bps < 10_000:
            raise ValueError(
                "liquidation_bonus_bps must be zero or include principal (at least 10,000)"
            )
        _require_bps("reserve_factor_bps", self.reserve_factor_bps)
        if self.liquidation_protocol_fee_bps is not None:
            _require_bps("liquidation_protocol_fee_bps", self.liquidation_protocol_fee_bps)
        if self.is_paused is not None and not isinstance(self.is_paused, bool):
            raise TypeError("is_paused must be bool or None")
        if self.flash_loan_enabled is not None and not isinstance(self.flash_loan_enabled, bool):
            raise TypeError("flash_loan_enabled must be bool or None")
        if self.liquidation_grace_period_until is not None:
            _require_non_negative(
                "liquidation_grace_period_until",
                self.liquidation_grace_period_until,
            )

    @property
    def ltv(self) -> Decimal:
        return _decimal_ratio(self.ltv_bps, 10_000)

    @property
    def liquidation_threshold(self) -> Decimal:
        return _decimal_ratio(self.liquidation_threshold_bps, 10_000)

    @property
    def liquidation_bonus_multiplier(self) -> Decimal:
        return _decimal_ratio(self.liquidation_bonus_bps, 10_000)

    @property
    def liquidation_bonus(self) -> Decimal:
        """Liquidator bonus rate, e.g. ``0.05`` for an Aave value of 10,500."""

        if self.liquidation_bonus_bps == 0:
            return Decimal(0)
        return self.liquidation_bonus_multiplier - Decimal(1)

    @property
    def liquidation_bonus_rate(self) -> Decimal:
        return self.liquidation_bonus

    @property
    def reserve_factor(self) -> Decimal:
        return _decimal_ratio(self.reserve_factor_bps, 10_000)


@dataclass(frozen=True, slots=True)
class AssetPosition:
    """One reserve in a user's portfolio at a single Ethereum block."""

    asset_address: str
    symbol: str
    decimals: int
    collateral_balance_raw: int
    stable_debt_balance_raw: int
    variable_debt_balance_raw: int
    oracle_price_raw: int
    oracle_base_currency_unit: int
    user_uses_as_collateral: bool
    reserve: ReserveRiskParameters
    price_source_address: str | None = None
    price_source_kind: str | None = None
    oracle_round_id: int | None = None
    oracle_updated_at: int | None = None
    reserve_id: int | None = None
    effective_ltv_bps: int | None = None
    effective_liquidation_threshold_bps: int | None = None
    effective_liquidation_bonus_bps: int | None = None
    a_token_address: str | None = None
    available_liquidity_raw: int | None = None
    a_token_total_supply_raw: int | None = None
    virtual_underlying_balance_raw: int | None = None

    def __post_init__(self) -> None:
        if not self.asset_address:
            raise ValueError("asset_address is required")
        if not self.symbol:
            raise ValueError("symbol is required")
        if not 0 <= self.decimals <= 36:
            raise ValueError("decimals must be between 0 and 36")
        for name in (
            "collateral_balance_raw",
            "stable_debt_balance_raw",
            "variable_debt_balance_raw",
            "oracle_price_raw",
        ):
            _require_non_negative(name, getattr(self, name))
        if self.oracle_base_currency_unit <= 0:
            raise ValueError("oracle_base_currency_unit must be positive")
        if self.reserve_id is not None and self.reserve_id < 0:
            raise ValueError("reserve_id must be non-negative")
        if self.oracle_round_id is not None and self.oracle_round_id < 0:
            raise ValueError("oracle_round_id must be non-negative")
        if self.oracle_updated_at is not None and self.oracle_updated_at < 0:
            raise ValueError("oracle_updated_at must be non-negative")
        if self.available_liquidity_raw is not None:
            _require_non_negative("available_liquidity_raw", self.available_liquidity_raw)
        if self.a_token_total_supply_raw is not None:
            _require_non_negative("a_token_total_supply_raw", self.a_token_total_supply_raw)
        if self.virtual_underlying_balance_raw is not None:
            _require_non_negative(
                "virtual_underlying_balance_raw",
                self.virtual_underlying_balance_raw,
            )
        if self.effective_ltv_bps is not None:
            _require_bps("effective_ltv_bps", self.effective_ltv_bps)
        if self.effective_liquidation_threshold_bps is not None:
            _require_bps(
                "effective_liquidation_threshold_bps",
                self.effective_liquidation_threshold_bps,
            )
        if (
            self.effective_liquidation_bonus_bps is not None
            and self.effective_liquidation_bonus_bps < 10_000
        ):
            raise ValueError("effective_liquidation_bonus_bps must be at least 10,000")

    @property
    def unit(self) -> int:
        return 10**self.decimals

    @property
    def collateral_balance(self) -> Decimal:
        return _decimal_ratio(self.collateral_balance_raw, self.unit)

    @property
    def collateral_units(self) -> Decimal:
        return self.collateral_balance

    @property
    def stable_debt_balance(self) -> Decimal:
        return _decimal_ratio(self.stable_debt_balance_raw, self.unit)

    @property
    def variable_debt_balance(self) -> Decimal:
        return _decimal_ratio(self.variable_debt_balance_raw, self.unit)

    @property
    def total_debt_balance_raw(self) -> int:
        return self.stable_debt_balance_raw + self.variable_debt_balance_raw

    @property
    def debt_balance(self) -> Decimal:
        return _decimal_ratio(self.total_debt_balance_raw, self.unit)

    @property
    def debt_units(self) -> Decimal:
        return self.debt_balance

    @property
    def price_usd(self) -> Decimal:
        return _decimal_ratio(self.oracle_price_raw, self.oracle_base_currency_unit)

    @property
    def collateral_value_usd(self) -> Decimal:
        with localcontext() as context:
            context.prec = 60
            return self.collateral_balance * self.price_usd

    @property
    def debt_value_usd(self) -> Decimal:
        with localcontext() as context:
            context.prec = 60
            return self.debt_balance * self.price_usd

    @property
    def effective_ltv(self) -> Decimal:
        bps = self.effective_ltv_bps
        return _decimal_ratio(self.reserve.ltv_bps if bps is None else bps, 10_000)

    @property
    def effective_liquidation_threshold(self) -> Decimal:
        bps = self.effective_liquidation_threshold_bps
        raw = self.reserve.liquidation_threshold_bps if bps is None else bps
        return _decimal_ratio(raw, 10_000)

    @property
    def applied_liquidation_bonus_bps(self) -> int:
        return (
            self.reserve.liquidation_bonus_bps
            if self.effective_liquidation_bonus_bps is None
            else self.effective_liquidation_bonus_bps
        )

    @property
    def contributes_collateral(self) -> bool:
        return (
            self.collateral_balance_raw > 0
            and self.user_uses_as_collateral
            and self.reserve.usage_as_collateral_enabled
            and self.effective_liquidation_threshold > 0
        )


@dataclass(frozen=True, slots=True)
class AccountSummary:
    """The aggregate values returned by ``Pool.getUserAccountData``."""

    total_collateral_base_raw: int
    total_debt_base_raw: int
    available_borrows_base_raw: int
    current_liquidation_threshold_bps: int
    ltv_bps: int
    health_factor_raw: int
    base_currency_unit: int

    def __post_init__(self) -> None:
        for name in (
            "total_collateral_base_raw",
            "total_debt_base_raw",
            "available_borrows_base_raw",
            "health_factor_raw",
        ):
            _require_non_negative(name, getattr(self, name))
        _require_bps("current_liquidation_threshold_bps", self.current_liquidation_threshold_bps)
        _require_bps("ltv_bps", self.ltv_bps)
        if self.base_currency_unit <= 0:
            raise ValueError("base_currency_unit must be positive")

    @property
    def total_collateral_usd(self) -> Decimal:
        return _decimal_ratio(self.total_collateral_base_raw, self.base_currency_unit)

    @property
    def total_debt_usd(self) -> Decimal:
        return _decimal_ratio(self.total_debt_base_raw, self.base_currency_unit)

    @property
    def available_borrows_usd(self) -> Decimal:
        return _decimal_ratio(self.available_borrows_base_raw, self.base_currency_unit)

    @property
    def current_liquidation_threshold(self) -> Decimal:
        return _decimal_ratio(self.current_liquidation_threshold_bps, 10_000)

    @property
    def ltv(self) -> Decimal:
        return _decimal_ratio(self.ltv_bps, 10_000)

    @property
    def health_factor(self) -> Decimal | None:
        if self.total_debt_base_raw == 0 or self.health_factor_raw == MAX_UINT256:
            return None
        return _decimal_ratio(self.health_factor_raw, WAD)


@dataclass(frozen=True, slots=True)
class BlockMetadata:
    number: int
    hash: str
    timestamp: int

    def __post_init__(self) -> None:
        _require_non_negative("number", self.number)
        _require_non_negative("timestamp", self.timestamp)
        if not self.hash:
            raise ValueError("block hash is required")


@dataclass(frozen=True, slots=True)
class AaveContracts:
    addresses_provider: str
    pool: str
    oracle: str
    pool_data_provider: str
    pool_implementation: str | None = None
    pool_revision: int | None = None


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """A reproducible, same-block snapshot of an Aave V3 account."""

    chain_id: int
    market: str
    user_address: str
    block: BlockMetadata
    contracts: AaveContracts
    base_currency_address: str
    base_currency_unit: int
    positions: tuple[AssetPosition, ...]
    account: AccountSummary
    user_emode_category: int = 0
    flashloan_premium_total_bps: int | None = None
    warnings: tuple[str, ...] = ()
    emode_data_complete: bool = False
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.chain_id <= 0:
            raise ValueError("chain_id must be positive")
        if self.base_currency_unit <= 0:
            raise ValueError("base_currency_unit must be positive")
        if self.account.base_currency_unit != self.base_currency_unit:
            raise ValueError("account and snapshot base currency units differ")
        if self.user_emode_category < 0:
            raise ValueError("user_emode_category must be non-negative")
        if not isinstance(self.emode_data_complete, bool):
            raise TypeError("emode_data_complete must be bool")
        if self.flashloan_premium_total_bps is not None:
            _require_bps("flashloan_premium_total_bps", self.flashloan_premium_total_bps)
        object.__setattr__(self, "positions", tuple(self.positions))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def assets(self) -> tuple[AssetPosition, ...]:
        """Compatibility alias for callers that use ``snapshot.assets``."""

        return self.positions

    def with_warning(self, warning: str) -> PortfolioSnapshot:
        return replace(self, warnings=(*self.warnings, warning))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Keep this at the top level so future schema migrations can route before parsing.
        result["schema_version"] = self.schema_version
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PortfolioSnapshot:
        version = int(payload.get("schema_version", 0))
        if version != 1:
            raise ValueError(f"unsupported snapshot schema_version: {version}")

        contracts = AaveContracts(**dict(payload["contracts"]))
        block = BlockMetadata(**dict(payload["block"]))
        account = AccountSummary(**dict(payload["account"]))
        positions = []
        for item in payload.get("positions", ()):
            position_data = dict(item)
            position_data["reserve"] = ReserveRiskParameters(**dict(position_data["reserve"]))
            positions.append(AssetPosition(**position_data))
        return cls(
            chain_id=int(payload["chain_id"]),
            market=str(payload["market"]),
            user_address=str(payload["user_address"]),
            block=block,
            contracts=contracts,
            base_currency_address=str(payload["base_currency_address"]),
            base_currency_unit=int(payload["base_currency_unit"]),
            positions=tuple(positions),
            account=account,
            user_emode_category=int(payload.get("user_emode_category", 0)),
            emode_data_complete=payload.get("emode_data_complete", False),
            flashloan_premium_total_bps=(
                None
                if payload.get("flashloan_premium_total_bps") is None
                else int(payload["flashloan_premium_total_bps"])
            ),
            warnings=tuple(str(item) for item in payload.get("warnings", ())),
        )


@dataclass(frozen=True, slots=True)
class PortfolioTotals:
    collateral_value_usd: Decimal
    debt_value_usd: Decimal
    liquidation_adjusted_collateral_usd: Decimal
    health_factor: Decimal | None


@dataclass(frozen=True, slots=True)
class HealthFactorReconciliation:
    calculated_health_factor: Decimal | None
    onchain_health_factor: Decimal | None
    absolute_error: Decimal | None
    relative_error: Decimal | None
    difference_bps: Decimal | None
    within_tolerance: bool
    calculated_collateral_usd: Decimal
    onchain_collateral_usd: Decimal
    calculated_debt_usd: Decimal
    onchain_debt_usd: Decimal
    collateral_difference_bps: Decimal | None
    debt_difference_bps: Decimal | None
    health_factor_within_tolerance: bool
    collateral_within_tolerance: bool
    debt_within_tolerance: bool


def calculate_portfolio_totals(positions: Iterable[AssetPosition]) -> PortfolioTotals:
    """Calculate asset-level portfolio values without making any network calls."""

    collateral = Decimal(0)
    debt = Decimal(0)
    adjusted_collateral = Decimal(0)
    with localcontext() as context:
        context.prec = 60
        for position in positions:
            debt += position.debt_value_usd
            if position.contributes_collateral:
                value = position.collateral_value_usd
                collateral += value
                adjusted_collateral += value * position.effective_liquidation_threshold
        health_factor = None if debt == 0 else adjusted_collateral / debt
    return PortfolioTotals(collateral, debt, adjusted_collateral, health_factor)


def calculate_health_factor(positions: Iterable[AssetPosition]) -> Decimal | None:
    """Return ``sum(collateral * effective LT) / sum(debt)``.

    ``None`` represents Aave's no-debt/infinite-health-factor state.
    """

    return calculate_portfolio_totals(positions).health_factor


def reconcile_health_factor(
    snapshot: PortfolioSnapshot,
    *,
    tolerance_bps: Decimal | int | str = Decimal("5"),
) -> HealthFactorReconciliation:
    """Reconcile asset-level HF and value totals with Aave's aggregate account data."""

    tolerance = Decimal(tolerance_bps)
    if tolerance < 0:
        raise ValueError("tolerance_bps must be non-negative")
    totals = calculate_portfolio_totals(snapshot.positions)
    calculated = totals.health_factor
    onchain = snapshot.account.health_factor
    onchain_collateral = snapshot.account.total_collateral_usd
    onchain_debt = snapshot.account.total_debt_usd

    def value_difference_bps(left: Decimal, right: Decimal) -> Decimal | None:
        if right == 0:
            return Decimal(0) if left == 0 else None
        return abs(left - right) / abs(right) * BPS

    collateral_difference = value_difference_bps(
        totals.collateral_value_usd,
        onchain_collateral,
    )
    debt_difference = value_difference_bps(totals.debt_value_usd, onchain_debt)
    collateral_within = collateral_difference is not None and collateral_difference <= tolerance
    debt_within = debt_difference is not None and debt_difference <= tolerance

    if calculated is None or onchain is None:
        health_factor_within = calculated is None and onchain is None
        return HealthFactorReconciliation(
            calculated_health_factor=calculated,
            onchain_health_factor=onchain,
            absolute_error=None,
            relative_error=None,
            difference_bps=None,
            within_tolerance=(health_factor_within and collateral_within and debt_within),
            calculated_collateral_usd=totals.collateral_value_usd,
            onchain_collateral_usd=onchain_collateral,
            calculated_debt_usd=totals.debt_value_usd,
            onchain_debt_usd=onchain_debt,
            collateral_difference_bps=collateral_difference,
            debt_difference_bps=debt_difference,
            health_factor_within_tolerance=health_factor_within,
            collateral_within_tolerance=collateral_within,
            debt_within_tolerance=debt_within,
        )

    absolute_error = abs(calculated - onchain)
    relative_error = absolute_error / abs(onchain) if onchain != 0 else absolute_error
    difference_bps = relative_error * BPS
    return HealthFactorReconciliation(
        calculated_health_factor=calculated,
        onchain_health_factor=onchain,
        absolute_error=absolute_error,
        relative_error=relative_error,
        difference_bps=difference_bps,
        within_tolerance=(difference_bps <= tolerance and collateral_within and debt_within),
        calculated_collateral_usd=totals.collateral_value_usd,
        onchain_collateral_usd=onchain_collateral,
        calculated_debt_usd=totals.debt_value_usd,
        onchain_debt_usd=onchain_debt,
        collateral_difference_bps=collateral_difference,
        debt_difference_bps=debt_difference,
        health_factor_within_tolerance=difference_bps <= tolerance,
        collateral_within_tolerance=collateral_within,
        debt_within_tolerance=debt_within,
    )


__all__ = [
    "AaveContracts",
    "AccountSummary",
    "AssetPosition",
    "BlockMetadata",
    "HealthFactorReconciliation",
    "PortfolioSnapshot",
    "PortfolioTotals",
    "ReserveRiskParameters",
    "calculate_health_factor",
    "calculate_portfolio_totals",
    "reconcile_health_factor",
]
