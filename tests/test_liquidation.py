from __future__ import annotations

import pytest

from aave_risk_monitor.models.liquidation import (
    CLOSE_FACTOR_HF_THRESHOLD_WAD,
    DEFAULT_LIQUIDATION_CLOSE_FACTOR_BPS,
    MAX_LIQUIDATION_CLOSE_FACTOR_BPS,
    MIN_LEFTOVER_BASE_RAW,
    WAD,
    DustLeftoverError,
    LiquidationError,
    LiquidationInputs,
    NotLiquidatableError,
    calculate_liquidation,
    collateral_to_base_floor,
    debt_to_base_ceil,
    percent_div,
    percent_div_ceil,
    percent_div_floor,
    percent_mul,
    percent_mul_ceil,
    percent_mul_floor,
)

USD8 = 10**8


def test_health_factor_must_be_strictly_below_one() -> None:
    inputs = LiquidationInputs(
        health_factor_wad=WAD,
        total_debt_base_raw=2_000 * USD8,
        borrower_collateral_balance_raw=1,
        borrower_debt_balance_raw=1,
        collateral_price_base_raw=2_000 * USD8,
        debt_price_base_raw=2_000 * USD8,
        collateral_decimals=0,
        debt_decimals=0,
        reserve_liquidation_bonus_bps=10_000,
    )

    with pytest.raises(NotLiquidatableError):
        calculate_liquidation(inputs)


def test_default_close_factor_caps_at_half_of_total_portfolio_debt() -> None:
    result = calculate_liquidation(
        LiquidationInputs(
            health_factor_wad=CLOSE_FACTOR_HF_THRESHOLD_WAD + 1,
            total_debt_base_raw=4_000 * USD8,
            borrower_collateral_balance_raw=2,
            borrower_debt_balance_raw=2,
            collateral_price_base_raw=2_000 * USD8,
            debt_price_base_raw=2_000 * USD8,
            collateral_decimals=0,
            debt_decimals=0,
            debt_to_cover_raw=2,
            reserve_liquidation_bonus_bps=10_000,
        )
    )

    assert result.close_factor_bps == DEFAULT_LIQUIDATION_CLOSE_FACTOR_BPS
    assert result.max_liquidatable_debt_raw == 1
    assert result.actual_debt_to_liquidate_raw == 1
    assert result.gross_collateral_seized_raw == 1


def test_hf_exactly_point_95_allows_full_selected_debt() -> None:
    result = calculate_liquidation(
        LiquidationInputs(
            health_factor_wad=CLOSE_FACTOR_HF_THRESHOLD_WAD,
            total_debt_base_raw=4_000 * USD8,
            borrower_collateral_balance_raw=2,
            borrower_debt_balance_raw=2,
            collateral_price_base_raw=2_000 * USD8,
            debt_price_base_raw=2_000 * USD8,
            collateral_decimals=0,
            debt_decimals=0,
            debt_to_cover_raw=2,
            reserve_liquidation_bonus_bps=10_000,
        )
    )

    assert result.close_factor_bps == MAX_LIQUIDATION_CLOSE_FACTOR_BPS
    assert result.max_liquidatable_debt_raw == 2
    assert result.actual_debt_to_liquidate_raw == 2


def test_close_factor_threshold_uses_floor_collateral_base_value() -> None:
    # One raw oracle unit below $2,000 disables the 50%-of-total cap.
    result = calculate_liquidation(
        LiquidationInputs(
            health_factor_wad=CLOSE_FACTOR_HF_THRESHOLD_WAD + 1,
            total_debt_base_raw=2_000 * USD8,
            borrower_collateral_balance_raw=1,
            borrower_debt_balance_raw=1,
            collateral_price_base_raw=2_000 * USD8 - 1,
            debt_price_base_raw=2_000 * USD8,
            collateral_decimals=0,
            debt_decimals=0,
            debt_to_cover_raw=1,
            reserve_liquidation_bonus_bps=10_000,
        )
    )

    assert result.selected_collateral_base_raw == 2_000 * USD8 - 1
    assert result.close_factor_bps == MAX_LIQUIDATION_CLOSE_FACTOR_BPS


def test_collateral_cap_recomputes_debt_and_fee_is_only_on_bonus() -> None:
    result = calculate_liquidation(
        LiquidationInputs(
            health_factor_wad=90 * 10**16,
            total_debt_base_raw=1_000 * USD8,
            borrower_collateral_balance_raw=500,  # 5.00 collateral
            borrower_debt_balance_raw=1_000,  # 10.00 debt
            collateral_price_base_raw=100 * USD8,
            debt_price_base_raw=100 * USD8,
            collateral_decimals=2,
            debt_decimals=2,
            debt_to_cover_raw=1_000,
            reserve_liquidation_bonus_bps=10_500,
            liquidation_protocol_fee_bps=1_000,
        )
    )

    assert result.gross_collateral_seized_raw == 500
    assert result.actual_debt_to_liquidate_raw == 477  # percentDivCeil
    assert result.bonus_collateral_raw == 24  # base portion is floor-rounded
    assert result.liquidation_protocol_fee_collateral_raw == 3  # fee rounds up
    assert result.collateral_to_liquidator_raw == 497
    assert result.liquidation_protocol_fee_collateral_raw < 50


def test_emode_bonus_overrides_reserve_bonus_only_for_enabled_collateral() -> None:
    common = dict(
        health_factor_wad=90 * 10**16,
        total_debt_base_raw=4_000 * USD8,
        borrower_collateral_balance_raw=5_000,
        borrower_debt_balance_raw=4_000,
        collateral_price_base_raw=100 * USD8,
        debt_price_base_raw=100 * USD8,
        collateral_decimals=2,
        debt_decimals=2,
        debt_to_cover_raw=1_000,
        reserve_liquidation_bonus_bps=10_500,
        borrower_emode_category=1,
        emode_liquidation_bonus_bps=10_100,
    )

    emode = calculate_liquidation(LiquidationInputs(**common, collateral_enabled_in_emode=True))
    reserve = calculate_liquidation(LiquidationInputs(**common, collateral_enabled_in_emode=False))

    assert emode.effective_liquidation_bonus_bps == 10_100
    assert emode.gross_collateral_seized_raw == 1_010
    assert reserve.effective_liquidation_bonus_bps == 10_500
    assert reserve.gross_collateral_seized_raw == 1_050


def test_active_emode_requires_category_bonus() -> None:
    with pytest.raises(LiquidationError, match="emode_liquidation_bonus_bps"):
        LiquidationInputs(
            health_factor_wad=90 * 10**16,
            total_debt_base_raw=1,
            borrower_collateral_balance_raw=1,
            borrower_debt_balance_raw=1,
            collateral_price_base_raw=1,
            debt_price_base_raw=1,
            collateral_decimals=0,
            debt_decimals=0,
            reserve_liquidation_bonus_bps=10_500,
            borrower_emode_category=1,
            collateral_enabled_in_emode=True,
        )


def test_v37_one_wei_rounding_directions() -> None:
    assert percent_mul(1, 5_000) == 1  # close factor remains half-up
    assert percent_mul_floor(1, 5_000) == 0
    assert percent_mul_ceil(1, 1) == 1
    assert percent_div(1, 20_000) == 1
    assert percent_div_floor(1, 20_000) == 0
    assert percent_div_ceil(1, 20_000) == 1
    assert debt_to_base_ceil(1, 1, 1) == 1
    assert collateral_to_base_floor(1, 1, 1) == 0


def test_v37_liquidation_rounds_seized_collateral_down_one_wei() -> None:
    result = calculate_liquidation(
        LiquidationInputs(
            health_factor_wad=90 * 10**16,
            total_debt_base_raw=1,
            borrower_collateral_balance_raw=10,
            borrower_debt_balance_raw=1,
            collateral_price_base_raw=1,
            debt_price_base_raw=1,
            collateral_decimals=0,
            debt_decimals=0,
            debt_to_cover_raw=1,
            reserve_liquidation_bonus_bps=15_000,
        )
    )

    # 1 * 150% = 1.5: V3.7 pays 1, not the legacy half-up result 2.
    assert result.base_collateral_raw == 1
    assert result.gross_collateral_seized_raw == 1


def test_v37_bonus_floor_then_protocol_fee_ceil_changes_one_wei() -> None:
    result = calculate_liquidation(
        LiquidationInputs(
            health_factor_wad=90 * 10**16,
            total_debt_base_raw=2,
            borrower_collateral_balance_raw=10,
            borrower_debt_balance_raw=2,
            collateral_price_base_raw=1,
            debt_price_base_raw=1,
            collateral_decimals=0,
            debt_decimals=0,
            debt_to_cover_raw=2,
            reserve_liquidation_bonus_bps=10_500,
            liquidation_protocol_fee_bps=1,
        )
    )

    assert result.gross_collateral_seized_raw == 2
    assert result.bonus_collateral_raw == 1  # 2 - floor(2 / 1.05)
    assert result.liquidation_protocol_fee_collateral_raw == 1
    assert result.collateral_to_liquidator_raw == 1


def test_partial_liquidation_must_not_leave_dust() -> None:
    with pytest.raises(DustLeftoverError) as caught:
        calculate_liquidation(
            LiquidationInputs(
                health_factor_wad=90 * 10**16,
                total_debt_base_raw=1_998 * USD8,
                borrower_collateral_balance_raw=2,
                borrower_debt_balance_raw=2,
                collateral_price_base_raw=999 * USD8,
                debt_price_base_raw=999 * USD8,
                collateral_decimals=0,
                debt_decimals=0,
                debt_to_cover_raw=1,
                reserve_liquidation_bonus_bps=10_000,
            )
        )

    assert caught.value.leftover_debt_base_raw == 999 * USD8
    assert caught.value.leftover_collateral_base_raw == 999 * USD8


def test_leftover_equal_to_minimum_is_allowed() -> None:
    result = calculate_liquidation(
        LiquidationInputs(
            health_factor_wad=90 * 10**16,
            total_debt_base_raw=2_000 * USD8,
            borrower_collateral_balance_raw=2,
            borrower_debt_balance_raw=2,
            collateral_price_base_raw=1_000 * USD8,
            debt_price_base_raw=1_000 * USD8,
            collateral_decimals=0,
            debt_decimals=0,
            debt_to_cover_raw=1,
            reserve_liquidation_bonus_bps=10_000,
        )
    )

    assert result.leftover_debt_base_raw == MIN_LEFTOVER_BASE_RAW
    assert result.leftover_collateral_base_raw == MIN_LEFTOVER_BASE_RAW


@pytest.mark.parametrize(
    ("collateral_balance", "debt_balance", "debt_to_cover", "exhausted_field"),
    [
        (2, 2, 2, "exhausts_selected_debt"),
        (1, 2, 2, "exhausts_selected_collateral"),
    ],
)
def test_dust_check_is_skipped_when_debt_or_collateral_is_exhausted(
    collateral_balance: int,
    debt_balance: int,
    debt_to_cover: int,
    exhausted_field: str,
) -> None:
    result = calculate_liquidation(
        LiquidationInputs(
            health_factor_wad=90 * 10**16,
            total_debt_base_raw=debt_balance * 999 * USD8,
            borrower_collateral_balance_raw=collateral_balance,
            borrower_debt_balance_raw=debt_balance,
            collateral_price_base_raw=999 * USD8,
            debt_price_base_raw=999 * USD8,
            collateral_decimals=0,
            debt_decimals=0,
            debt_to_cover_raw=debt_to_cover,
            reserve_liquidation_bonus_bps=10_000,
        )
    )

    assert getattr(result, exhausted_field) is True
