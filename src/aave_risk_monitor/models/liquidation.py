"""Aave V3.7 liquidation math implemented with contract-style integer arithmetic.

All token amounts are raw ERC-20 integers. Oracle/base-currency amounts are raw
integers in the oracle's own precision (8 decimals for the default Aave V3
Ethereum market constants).  No binary floating-point arithmetic is used.

The implementation mirrors the economically relevant part of Aave V3.7's
``LiquidationLogic`` (Pool revision 11): eligibility, the redesigned close
factor, collateral capping, eMode liquidation bonus selection, protocol fee,
and the minimum-leftover/dust rule. It is a read-only model; it neither builds
nor submits transactions.
"""

from __future__ import annotations

from dataclasses import dataclass

PERCENTAGE_FACTOR = 10_000
WAD = 10**18
HEALTH_FACTOR_LIQUIDATION_THRESHOLD_WAD = WAD
CLOSE_FACTOR_HF_THRESHOLD_WAD = 95 * 10**16  # 0.95e18
DEFAULT_LIQUIDATION_CLOSE_FACTOR_BPS = 5_000
MAX_LIQUIDATION_CLOSE_FACTOR_BPS = 10_000

# These are the V3.7 contract defaults. They assume an 8-decimal USD base
# currency, as the upstream contract comments explicitly note.
MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD_RAW = 2_000 * 10**8
MIN_LEFTOVER_BASE_RAW = MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD_RAW // 2


class LiquidationError(ValueError):
    """Base error for an invalid or non-executable liquidation model input."""


class NotLiquidatableError(LiquidationError):
    """Raised when the account health factor is not strictly below one."""


class DustLeftoverError(LiquidationError):
    """Raised when a partial liquidation would violate Aave's dust rule."""

    def __init__(
        self,
        *,
        leftover_debt_base_raw: int,
        leftover_collateral_base_raw: int,
        minimum_leftover_base_raw: int = MIN_LEFTOVER_BASE_RAW,
    ) -> None:
        self.leftover_debt_base_raw = leftover_debt_base_raw
        self.leftover_collateral_base_raw = leftover_collateral_base_raw
        self.minimum_leftover_base_raw = minimum_leftover_base_raw
        super().__init__(
            "partial liquidation would leave protocol dust: "
            f"debt={leftover_debt_base_raw}, "
            f"collateral={leftover_collateral_base_raw}, "
            f"minimum={minimum_leftover_base_raw} (oracle base raw units)"
        )


def _require_uint(name: str, value: int, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer in raw units")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise LiquidationError(f"{name} must be {qualifier}")


def token_unit(decimals: int) -> int:
    """Return ``10**decimals`` while respecting Solidity uint256 bounds."""

    _require_uint("decimals", decimals)
    if decimals > 77:  # 10**78 exceeds uint256
        raise LiquidationError("decimals must be <= 77 for uint256-compatible units")
    return 10**decimals


def mul_div_ceil(a: int, b: int, denominator: int) -> int:
    """Mirror Aave ``MathUtils.mulDivCeil`` for non-negative integers."""

    _require_uint("a", a)
    _require_uint("b", b)
    _require_uint("denominator", denominator, positive=True)
    product = a * b
    return product // denominator + int(product % denominator != 0)


def percent_mul(value: int, percentage_bps: int) -> int:
    """Aave ``percentMul``: nearest integer, with half rounded upward."""

    _require_uint("value", value)
    _require_uint("percentage_bps", percentage_bps)
    return (value * percentage_bps + PERCENTAGE_FACTOR // 2) // PERCENTAGE_FACTOR


def percent_mul_ceil(value: int, percentage_bps: int) -> int:
    """Aave ``percentMulCeil`` used by V3.7 flash-loan premiums."""

    _require_uint("value", value)
    _require_uint("percentage_bps", percentage_bps)
    product = value * percentage_bps
    return product // PERCENTAGE_FACTOR + int(product % PERCENTAGE_FACTOR != 0)


def percent_mul_floor(value: int, percentage_bps: int) -> int:
    """Aave ``percentMulFloor`` used for V3.7 collateral seizure math."""

    _require_uint("value", value)
    _require_uint("percentage_bps", percentage_bps)
    return value * percentage_bps // PERCENTAGE_FACTOR


def percent_div(value: int, percentage_bps: int) -> int:
    """Aave ``percentDiv``: nearest integer, with half rounded upward."""

    _require_uint("value", value)
    _require_uint("percentage_bps", percentage_bps, positive=True)
    return (value * PERCENTAGE_FACTOR + percentage_bps // 2) // percentage_bps


def percent_div_floor(value: int, percentage_bps: int) -> int:
    """Aave ``percentDivFloor`` used to isolate liquidation bonus collateral."""

    _require_uint("value", value)
    _require_uint("percentage_bps", percentage_bps, positive=True)
    return value * PERCENTAGE_FACTOR // percentage_bps


def percent_div_ceil(value: int, percentage_bps: int) -> int:
    """Mirror Aave ``percentDivCeil``."""

    _require_uint("value", value)
    _require_uint("percentage_bps", percentage_bps, positive=True)
    numerator = value * PERCENTAGE_FACTOR
    return numerator // percentage_bps + int(numerator % percentage_bps != 0)


def debt_to_base_ceil(amount_raw: int, price_base_raw: int, decimals: int) -> int:
    """Convert raw debt to oracle base units, rounding upward like V3.7."""

    return mul_div_ceil(amount_raw, price_base_raw, token_unit(decimals))


def collateral_to_base_floor(amount_raw: int, price_base_raw: int, decimals: int) -> int:
    """Convert raw collateral to oracle base units, rounding downward."""

    _require_uint("amount_raw", amount_raw)
    _require_uint("price_base_raw", price_base_raw, positive=True)
    return amount_raw * price_base_raw // token_unit(decimals)


def is_liquidatable(health_factor_wad: int) -> bool:
    """Return true only for the strict Aave condition ``health factor < 1``."""

    _require_uint("health_factor_wad", health_factor_wad)
    return health_factor_wad < HEALTH_FACTOR_LIQUIDATION_THRESHOLD_WAD


@dataclass(frozen=True, slots=True)
class LiquidationInputs:
    """Inputs for one selected collateral/debt reserve pair.

    ``*_raw`` token fields use the corresponding ERC-20 decimals. ``*_base_raw``
    fields use the Aave oracle base currency. A V3 liquidation bonus is the full
    percentage (for example 10500 means principal plus a 5% bonus), while the
    protocol fee is a conventional portion of only that bonus (1000 means 10%).
    """

    health_factor_wad: int
    total_debt_base_raw: int
    borrower_collateral_balance_raw: int
    borrower_debt_balance_raw: int
    collateral_price_base_raw: int
    debt_price_base_raw: int
    collateral_decimals: int
    debt_decimals: int
    reserve_liquidation_bonus_bps: int
    debt_to_cover_raw: int | None = None
    liquidation_protocol_fee_bps: int = 0
    borrower_emode_category: int = 0
    collateral_enabled_in_emode: bool = False
    emode_liquidation_bonus_bps: int | None = None

    def __post_init__(self) -> None:
        _require_uint("health_factor_wad", self.health_factor_wad)
        _require_uint("total_debt_base_raw", self.total_debt_base_raw, positive=True)
        _require_uint(
            "borrower_collateral_balance_raw",
            self.borrower_collateral_balance_raw,
            positive=True,
        )
        _require_uint("borrower_debt_balance_raw", self.borrower_debt_balance_raw, positive=True)
        _require_uint("collateral_price_base_raw", self.collateral_price_base_raw, positive=True)
        _require_uint("debt_price_base_raw", self.debt_price_base_raw, positive=True)
        token_unit(self.collateral_decimals)
        token_unit(self.debt_decimals)

        if self.debt_to_cover_raw is not None:
            _require_uint("debt_to_cover_raw", self.debt_to_cover_raw, positive=True)

        _validate_liquidation_bonus(
            "reserve_liquidation_bonus_bps", self.reserve_liquidation_bonus_bps
        )
        if self.emode_liquidation_bonus_bps is not None:
            _validate_liquidation_bonus(
                "emode_liquidation_bonus_bps", self.emode_liquidation_bonus_bps
            )
        _require_uint("liquidation_protocol_fee_bps", self.liquidation_protocol_fee_bps)
        if self.liquidation_protocol_fee_bps > PERCENTAGE_FACTOR:
            raise LiquidationError("liquidation_protocol_fee_bps must be <= 10000")
        _require_uint("borrower_emode_category", self.borrower_emode_category)
        if self.borrower_emode_category > 255:
            raise LiquidationError("borrower_emode_category must fit uint8")
        if not isinstance(self.collateral_enabled_in_emode, bool):
            raise TypeError("collateral_enabled_in_emode must be bool")
        if (
            self.borrower_emode_category != 0
            and self.collateral_enabled_in_emode
            and self.emode_liquidation_bonus_bps is None
        ):
            raise LiquidationError(
                "emode_liquidation_bonus_bps is required when the selected "
                "collateral is enabled in the borrower's active eMode category"
            )

    @property
    def effective_liquidation_bonus_bps(self) -> int:
        if self.borrower_emode_category != 0 and self.collateral_enabled_in_emode:
            # __post_init__ guarantees this is present.
            assert self.emode_liquidation_bonus_bps is not None
            return self.emode_liquidation_bonus_bps
        return self.reserve_liquidation_bonus_bps


def _validate_liquidation_bonus(name: str, value: int) -> None:
    _require_uint(name, value, positive=True)
    if value < PERCENTAGE_FACTOR:
        raise LiquidationError(f"{name} must include principal and be >= 10000")


@dataclass(frozen=True, slots=True)
class LiquidationResult:
    """Contract-style liquidation outcome in raw token/oracle units."""

    requested_debt_to_cover_raw: int
    selected_debt_base_raw: int
    selected_collateral_base_raw: int
    close_factor_bps: int
    max_liquidatable_debt_raw: int
    effective_liquidation_bonus_bps: int
    base_collateral_raw: int
    gross_collateral_seized_raw: int
    collateral_to_liquidator_raw: int
    liquidation_protocol_fee_collateral_raw: int
    bonus_collateral_raw: int
    actual_debt_to_liquidate_raw: int
    collateral_seized_base_raw: int
    leftover_debt_base_raw: int
    leftover_collateral_base_raw: int
    exhausts_selected_debt: bool
    exhausts_selected_collateral: bool


def calculate_liquidation(inputs: LiquidationInputs) -> LiquidationResult:
    """Calculate one V3.7 liquidation and enforce the protocol dust rule.

    ``NotLiquidatableError`` models the contract's strict HF validation.
    ``DustLeftoverError`` models ``Errors.MustNotLeaveDust``. Invalid actions
    therefore cannot accidentally be displayed as executable opportunities.
    """

    if not is_liquidatable(inputs.health_factor_wad):
        raise NotLiquidatableError("Aave liquidation requires health_factor_wad < 1e18")

    collateral_unit = token_unit(inputs.collateral_decimals)
    debt_unit = token_unit(inputs.debt_decimals)

    selected_debt_base_raw = debt_to_base_ceil(
        inputs.borrower_debt_balance_raw,
        inputs.debt_price_base_raw,
        inputs.debt_decimals,
    )
    selected_collateral_base_raw = collateral_to_base_floor(
        inputs.borrower_collateral_balance_raw,
        inputs.collateral_price_base_raw,
        inputs.collateral_decimals,
    )

    max_liquidatable_debt_raw = inputs.borrower_debt_balance_raw
    close_factor_bps = MAX_LIQUIDATION_CLOSE_FACTOR_BPS

    uses_default_close_factor = (
        selected_collateral_base_raw >= MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD_RAW
        and selected_debt_base_raw >= MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD_RAW
        and inputs.health_factor_wad > CLOSE_FACTOR_HF_THRESHOLD_WAD
    )
    if uses_default_close_factor:
        close_factor_bps = DEFAULT_LIQUIDATION_CLOSE_FACTOR_BPS
        total_default_liquidatable_debt_base_raw = percent_mul(
            inputs.total_debt_base_raw, DEFAULT_LIQUIDATION_CLOSE_FACTOR_BPS
        )
        if selected_debt_base_raw > total_default_liquidatable_debt_base_raw:
            # Explicit floor rounding from V3.7 LiquidationLogic.
            max_liquidatable_debt_raw = (
                total_default_liquidatable_debt_base_raw * debt_unit
            ) // inputs.debt_price_base_raw

    requested_debt_to_cover_raw = (
        inputs.borrower_debt_balance_raw
        if inputs.debt_to_cover_raw is None
        else inputs.debt_to_cover_raw
    )
    debt_to_cover_raw = min(requested_debt_to_cover_raw, max_liquidatable_debt_raw)
    if debt_to_cover_raw == 0:
        raise LiquidationError("rounding reduced max liquidatable debt to zero")

    effective_bonus_bps = inputs.effective_liquidation_bonus_bps

    # V3.7: base collateral conversion is explicitly floor-rounded.
    base_collateral_raw = (inputs.debt_price_base_raw * debt_to_cover_raw * collateral_unit) // (
        inputs.collateral_price_base_raw * debt_unit
    )
    # V3.7 explicitly rounds collateral paid by the borrower downward.
    max_collateral_to_liquidate_raw = percent_mul_floor(base_collateral_raw, effective_bonus_bps)

    if max_collateral_to_liquidate_raw > inputs.borrower_collateral_balance_raw:
        gross_collateral_seized_raw = inputs.borrower_collateral_balance_raw
        debt_value_for_available_collateral_raw = (
            inputs.collateral_price_base_raw * gross_collateral_seized_raw * debt_unit
        ) // (inputs.debt_price_base_raw * collateral_unit)
        actual_debt_to_liquidate_raw = percent_div_ceil(
            debt_value_for_available_collateral_raw, effective_bonus_bps
        )
    else:
        gross_collateral_seized_raw = max_collateral_to_liquidate_raw
        actual_debt_to_liquidate_raw = debt_to_cover_raw

    collateral_seized_base_raw = (
        gross_collateral_seized_raw * inputs.collateral_price_base_raw
    ) // collateral_unit

    # V3.7 rounds the base portion down, then the treasury's portion of the
    # resulting bonus up. This combination makes the one-wei direction explicit.
    bonus_collateral_raw = gross_collateral_seized_raw - percent_div_floor(
        gross_collateral_seized_raw, effective_bonus_bps
    )
    liquidation_protocol_fee_collateral_raw = 0
    if inputs.liquidation_protocol_fee_bps != 0:
        # The fee is taken only from the bonus, not from principal collateral.
        liquidation_protocol_fee_collateral_raw = percent_mul_ceil(
            bonus_collateral_raw, inputs.liquidation_protocol_fee_bps
        )

    collateral_to_liquidator_raw = (
        gross_collateral_seized_raw - liquidation_protocol_fee_collateral_raw
    )
    leftover_debt_raw = inputs.borrower_debt_balance_raw - actual_debt_to_liquidate_raw
    leftover_collateral_raw = inputs.borrower_collateral_balance_raw - gross_collateral_seized_raw

    exhausts_selected_debt = actual_debt_to_liquidate_raw == inputs.borrower_debt_balance_raw
    exhausts_selected_collateral = (
        gross_collateral_seized_raw == inputs.borrower_collateral_balance_raw
    )

    leftover_debt_base_raw = debt_to_base_ceil(
        leftover_debt_raw, inputs.debt_price_base_raw, inputs.debt_decimals
    )
    leftover_collateral_base_raw = collateral_to_base_floor(
        leftover_collateral_raw,
        inputs.collateral_price_base_raw,
        inputs.collateral_decimals,
    )

    # The on-chain check runs only when neither the selected debt nor the gross
    # collateral (liquidator amount + protocol fee) is fully consumed.
    if (
        not exhausts_selected_debt
        and not exhausts_selected_collateral
        and (
            leftover_debt_base_raw < MIN_LEFTOVER_BASE_RAW
            or leftover_collateral_base_raw < MIN_LEFTOVER_BASE_RAW
        )
    ):
        raise DustLeftoverError(
            leftover_debt_base_raw=leftover_debt_base_raw,
            leftover_collateral_base_raw=leftover_collateral_base_raw,
        )

    return LiquidationResult(
        requested_debt_to_cover_raw=requested_debt_to_cover_raw,
        selected_debt_base_raw=selected_debt_base_raw,
        selected_collateral_base_raw=selected_collateral_base_raw,
        close_factor_bps=close_factor_bps,
        max_liquidatable_debt_raw=max_liquidatable_debt_raw,
        effective_liquidation_bonus_bps=effective_bonus_bps,
        base_collateral_raw=base_collateral_raw,
        gross_collateral_seized_raw=gross_collateral_seized_raw,
        collateral_to_liquidator_raw=collateral_to_liquidator_raw,
        liquidation_protocol_fee_collateral_raw=(liquidation_protocol_fee_collateral_raw),
        bonus_collateral_raw=bonus_collateral_raw,
        actual_debt_to_liquidate_raw=actual_debt_to_liquidate_raw,
        collateral_seized_base_raw=collateral_seized_base_raw,
        leftover_debt_base_raw=leftover_debt_base_raw,
        leftover_collateral_base_raw=leftover_collateral_base_raw,
        exhausts_selected_debt=exhausts_selected_debt,
        exhausts_selected_collateral=exhausts_selected_collateral,
    )


__all__ = [
    "CLOSE_FACTOR_HF_THRESHOLD_WAD",
    "DEFAULT_LIQUIDATION_CLOSE_FACTOR_BPS",
    "DustLeftoverError",
    "HEALTH_FACTOR_LIQUIDATION_THRESHOLD_WAD",
    "LiquidationError",
    "LiquidationInputs",
    "LiquidationResult",
    "MAX_LIQUIDATION_CLOSE_FACTOR_BPS",
    "MIN_BASE_MAX_CLOSE_FACTOR_THRESHOLD_RAW",
    "MIN_LEFTOVER_BASE_RAW",
    "NotLiquidatableError",
    "PERCENTAGE_FACTOR",
    "WAD",
    "calculate_liquidation",
    "collateral_to_base_floor",
    "debt_to_base_ceil",
    "is_liquidatable",
    "mul_div_ceil",
    "percent_div",
    "percent_div_ceil",
    "percent_div_floor",
    "percent_mul",
    "percent_mul_ceil",
    "percent_mul_floor",
    "token_unit",
]
