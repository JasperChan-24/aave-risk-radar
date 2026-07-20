"""Deterministic offline portfolio and history used by the public demo."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from aave_risk_monitor.domain import (
    AaveContracts,
    AccountSummary,
    AssetPosition,
    BlockMetadata,
    PortfolioSnapshot,
    ReserveRiskParameters,
)

WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
WBTC = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
USDC = "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
DEMO_USER = "0x000000000000000000000000000000000000dEaD"
DEMO_WARNING = (
    "Synthetic deterministic portfolio for offline demonstration; values are not a live "
    "wallet and must not be interpreted as an executable liquidation."
)


def _reserve(
    *,
    ltv_bps: int,
    threshold_bps: int,
    bonus_bps: int,
    collateral: bool,
) -> ReserveRiskParameters:
    return ReserveRiskParameters(
        ltv_bps=ltv_bps,
        liquidation_threshold_bps=threshold_bps,
        liquidation_bonus_bps=bonus_bps,
        reserve_factor_bps=1_000,
        usage_as_collateral_enabled=collateral,
        borrowing_enabled=True,
        stable_borrow_rate_enabled=False,
        is_active=True,
        is_frozen=False,
        liquidation_protocol_fee_bps=1_000,
        is_paused=False,
        flash_loan_enabled=True,
        liquidation_grace_period_until=0,
    )


def build_demo_snapshot() -> PortfolioSnapshot:
    """Return a fixed multi-collateral, single-debt account near enough to be interesting."""

    base_unit = 10**8
    positions = (
        AssetPosition(
            asset_address=WETH,
            symbol="WETH",
            decimals=18,
            collateral_balance_raw=5 * 10**18,
            stable_debt_balance_raw=0,
            variable_debt_balance_raw=0,
            oracle_price_raw=3_600 * base_unit,
            oracle_base_currency_unit=base_unit,
            user_uses_as_collateral=True,
            reserve=_reserve(
                ltv_bps=8_000,
                threshold_bps=8_250,
                bonus_bps=10_500,
                collateral=True,
            ),
            price_source_address="demo:aave-oracle-weth",
            price_source_kind="synthetic",
            reserve_id=0,
            a_token_address="demo:a-weth",
            available_liquidity_raw=100_000 * 10**18,
            a_token_total_supply_raw=150_000 * 10**18,
            virtual_underlying_balance_raw=100_000 * 10**18,
        ),
        AssetPosition(
            asset_address=WBTC,
            symbol="WBTC",
            decimals=8,
            collateral_balance_raw=10_000_000,  # 0.1 WBTC
            stable_debt_balance_raw=0,
            variable_debt_balance_raw=0,
            oracle_price_raw=110_000 * base_unit,
            oracle_base_currency_unit=base_unit,
            user_uses_as_collateral=True,
            reserve=_reserve(
                ltv_bps=7_500,
                threshold_bps=8_000,
                bonus_bps=10_500,
                collateral=True,
            ),
            price_source_address="demo:aave-oracle-wbtc",
            price_source_kind="synthetic",
            reserve_id=1,
            a_token_address="demo:a-wbtc",
            available_liquidity_raw=1_000 * 10**8,
            a_token_total_supply_raw=2_000 * 10**8,
            virtual_underlying_balance_raw=1_000 * 10**8,
        ),
        AssetPosition(
            asset_address=USDC,
            symbol="USDC",
            decimals=6,
            collateral_balance_raw=0,
            stable_debt_balance_raw=0,
            variable_debt_balance_raw=20_000 * 10**6,
            oracle_price_raw=base_unit,
            oracle_base_currency_unit=base_unit,
            user_uses_as_collateral=False,
            reserve=_reserve(
                ltv_bps=7_800,
                threshold_bps=8_100,
                bonus_bps=10_450,
                collateral=True,
            ),
            price_source_address="demo:aave-oracle-usdc",
            price_source_kind="synthetic",
            reserve_id=2,
            a_token_address="demo:a-usdc",
            available_liquidity_raw=100_000_000 * 10**6,
            a_token_total_supply_raw=150_000_000 * 10**6,
            virtual_underlying_balance_raw=100_000_000 * 10**6,
        ),
    )
    # Adjusted collateral = 5*3600*.825 + .1*110000*.80 = $23,650.
    # Debt = $20,000, so HF = 1.1825.
    account = AccountSummary(
        total_collateral_base_raw=29_000 * base_unit,
        total_debt_base_raw=20_000 * base_unit,
        available_borrows_base_raw=0,
        current_liquidation_threshold_bps=8_155,
        ltv_bps=7_810,
        health_factor_raw=1_182_500_000_000_000_000,
        base_currency_unit=base_unit,
    )
    return PortfolioSnapshot(
        chain_id=1,
        market="Synthetic Ethereum V3-style fixture",
        user_address=DEMO_USER,
        block=BlockMetadata(
            number=0,
            hash="synthetic-demo-fixture-v1",
            timestamp=int(datetime(2026, 7, 20, tzinfo=UTC).timestamp()),
        ),
        contracts=AaveContracts(
            addresses_provider="0x2f39D218133AFaB8F2B819B1066c7E434Ad94E9e",
            pool="0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2",
            oracle="demo:aave-oracle",
            pool_data_provider="demo:protocol-data-provider",
            pool_implementation="demo:pool-v3.7",
            pool_revision=11,
        ),
        base_currency_address="0x0000000000000000000000000000000000000000",
        base_currency_unit=base_unit,
        positions=positions,
        account=account,
        user_emode_category=0,
        emode_data_complete=True,
        flashloan_premium_total_bps=5,
        warnings=(DEMO_WARNING,),
    )


def build_demo_history(days: int = 366) -> pd.DataFrame:
    """Generate a fixed, correlated daily history ending at the demo spot prices.

    This fixture exercises calibration and statistics offline. It is deliberately
    labelled synthetic in the UI; live research should use cached Aave-oracle
    observations produced by the capture script.
    """

    if days < 30:
        raise ValueError("demo history needs at least 30 daily observations")
    annual_volatility = np.asarray([0.68, 0.58, 0.02], dtype=np.float64)
    correlation = np.asarray(
        [
            [1.00, 0.78, -0.04],
            [0.78, 1.00, -0.02],
            [-0.04, -0.02, 1.00],
        ],
        dtype=np.float64,
    )
    daily_covariance = correlation * np.outer(annual_volatility, annual_volatility) / 365.0
    factor = np.linalg.cholesky(daily_covariance)
    random = np.random.default_rng(20260720)
    log_returns = random.standard_normal((days - 1, 3)) @ factor.T
    log_prices = np.vstack([np.zeros(3), np.cumsum(log_returns, axis=0)])
    paths = np.exp(log_prices)
    target_spot = np.asarray([3_600.0, 110_000.0, 1.0])
    paths *= target_spot / paths[-1]
    index = pd.date_range(end="2026-07-20", periods=days, freq="D")
    return pd.DataFrame(
        paths,
        index=index,
        columns=[WETH.lower(), WBTC.lower(), USDC.lower()],
    )


__all__ = [
    "DEMO_USER",
    "DEMO_WARNING",
    "USDC",
    "WBTC",
    "WETH",
    "build_demo_history",
    "build_demo_snapshot",
]
