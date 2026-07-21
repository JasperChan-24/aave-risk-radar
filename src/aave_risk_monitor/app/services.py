"""Pure orchestration helpers shared by Streamlit and smoke tests."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

from aave_risk_monitor.domain import (
    AssetPosition,
    PortfolioSnapshot,
    PortfolioTotals,
    calculate_portfolio_totals,
    reconcile_health_factor,
)
from aave_risk_monitor.models import (
    LiquidationInputs,
    LiquidationResult,
    SwapQuote,
    token_unit,
)
from aave_risk_monitor.simulation import (
    MarketCalibration,
    SimulationConfig,
    SimulationResult,
    calibrate_log_returns,
    make_linear_portfolio_revaluer,
    simulate_from_calibration,
)

SUPPORTED_LIQUIDATION_POOL_REVISIONS = frozenset({11})


def active_positions(snapshot: PortfolioSnapshot) -> tuple[AssetPosition, ...]:
    """Return reserves that materially affect equity or health factor."""

    return tuple(
        position
        for position in snapshot.positions
        if position.collateral_balance_raw > 0 or position.total_debt_balance_raw > 0
    )


def position_key(position: AssetPosition) -> str:
    """Stable history-column key independent of token symbol changes."""

    return position.asset_address.lower()


def snapshot_table(snapshot: PortfolioSnapshot) -> pd.DataFrame:
    """Build a human-readable per-asset table without losing the source address."""

    rows: list[dict[str, object]] = []
    for position in active_positions(snapshot):
        rows.append(
            {
                "asset": position.symbol,
                "address": position.asset_address,
                "supplied": float(position.collateral_balance),
                "debt": float(position.debt_balance),
                "collateral_enabled": position.contributes_collateral,
                "price_usd": float(position.price_usd),
                "supplied_value_usd": float(position.collateral_value_usd),
                "debt_value_usd": float(position.debt_value_usd),
                "ltv_pct": float(position.effective_ltv * 100),
                "liquidation_threshold_pct": float(position.effective_liquidation_threshold * 100),
                "liquidation_bonus_pct": (
                    0
                    if position.applied_liquidation_bonus_bps == 0
                    else position.applied_liquidation_bonus_bps - 10_000
                )
                / 100,
                "protocol_fee_pct_of_bonus": (
                    None
                    if position.reserve.liquidation_protocol_fee_bps is None
                    else position.reserve.liquidation_protocol_fee_bps / 100
                ),
                "oracle_source_kind": position.price_source_kind or "unknown",
                "oracle_source": position.price_source_address or "unknown",
                "reserve_active": position.reserve.is_active,
                "reserve_paused": position.reserve.is_paused,
                "liquidation_grace_period_until": (position.reserve.liquidation_grace_period_until),
                "flash_loan_enabled": position.reserve.flash_loan_enabled,
                "available_liquidity": (
                    None
                    if position.available_liquidity_raw is None
                    else float(
                        Decimal(position.available_liquidity_raw)
                        / Decimal(token_unit(position.decimals))
                    )
                ),
                "a_token_total_supply": (
                    None
                    if position.a_token_total_supply_raw is None
                    else float(
                        Decimal(position.a_token_total_supply_raw)
                        / Decimal(token_unit(position.decimals))
                    )
                ),
                "virtual_underlying_balance": (
                    None
                    if position.virtual_underlying_balance_raw is None
                    else float(
                        Decimal(position.virtual_underlying_balance_raw)
                        / Decimal(token_unit(position.decimals))
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def apply_price_shocks(
    snapshot: PortfolioSnapshot,
    shocks_pct: Mapping[str, float | int | Decimal],
) -> PortfolioSnapshot:
    """Return a new snapshot with address-keyed percentage price shocks.

    A price belongs to the asset, so the same shocked price revalues both its
    supplied and borrowed balances. The original block snapshot is untouched.
    """

    shocked_positions: list[AssetPosition] = []
    for position in snapshot.positions:
        raw_shock = shocks_pct.get(position_key(position), 0)
        shock = Decimal(str(raw_shock)) / Decimal(100)
        multiplier = Decimal(1) + shock
        if multiplier <= 0:
            raise ValueError("a price shock cannot reduce an asset price to zero or below")
        shocked_price = int((Decimal(position.oracle_price_raw) * multiplier).to_integral_value())
        if shocked_price <= 0:
            raise ValueError("rounded shocked price must remain positive")
        shocked_positions.append(replace(position, oracle_price_raw=shocked_price))
    return replace(snapshot, positions=tuple(shocked_positions))


def shocked_totals(
    snapshot: PortfolioSnapshot,
    shocks_pct: Mapping[str, float | int | Decimal],
) -> PortfolioTotals:
    return calculate_portfolio_totals(apply_price_shocks(snapshot, shocks_pct).positions)


def load_price_history(
    source: str | Path | BinaryIO | TextIO,
    *,
    snapshot: PortfolioSnapshot | None = None,
) -> pd.DataFrame:
    """Load a daily cache and, when supplied, verify snapshot provenance."""

    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError("history CSV is empty")
    if "date" not in frame.columns:
        raise ValueError("history CSV is missing the date column")

    dates = pd.to_datetime(frame["date"], utc=True, errors="raise", format="mixed")
    if snapshot is not None:
        required_metadata = {
            "block_number",
            "block_hash",
            "block_timestamp_utc",
            "oracle_address",
        }
        missing_metadata = required_metadata.difference(frame.columns)
        if missing_metadata:
            raise ValueError(
                "history CSV is missing reproducibility columns: "
                + ", ".join(sorted(missing_metadata))
            )

        numeric_blocks = pd.to_numeric(frame["block_number"], errors="raise")
        if not np.equal(numeric_blocks, np.floor(numeric_blocks)).all():
            raise ValueError("history block numbers must be integers")
        block_numbers = numeric_blocks.astype(np.int64)
        if block_numbers.duplicated().any() or not block_numbers.is_monotonic_increasing:
            raise ValueError("history block numbers must be unique and increasing")
        if int(block_numbers.max()) > snapshot.block.number:
            raise ValueError("history contains a block after the portfolio snapshot")

        block_hashes = frame["block_hash"].astype(str)
        if not block_hashes.str.fullmatch(r"0x[0-9a-fA-F]{64}").all():
            raise ValueError("history contains an invalid Ethereum block hash")
        snapshot_rows = block_numbers == snapshot.block.number
        if (
            snapshot_rows.any()
            and not (block_hashes[snapshot_rows].str.lower() == snapshot.block.hash.lower()).all()
        ):
            raise ValueError("history block hash does not match the portfolio snapshot")

        oracle_addresses = {str(value).lower() for value in frame["oracle_address"]}
        if oracle_addresses != {snapshot.contracts.oracle.lower()}:
            raise ValueError("history oracle address does not match the portfolio snapshot")

        block_timestamps = pd.to_datetime(
            frame["block_timestamp_utc"],
            utc=True,
            errors="raise",
            format="mixed",
        )
        snapshot_timestamp = pd.Timestamp(snapshot.block.timestamp, unit="s", tz="UTC")
        if block_timestamps.max() > snapshot_timestamp:
            raise ValueError("history contains a block timestamp after the portfolio snapshot")
        if not np.array_equal(dates.dt.date.to_numpy(), block_timestamps.dt.date.to_numpy()):
            raise ValueError("history date does not match block_timestamp_utc")

    frame["date"] = dates.dt.tz_localize(None)
    frame = frame.set_index("date")
    frame.columns = [str(column).lower() for column in frame.columns]
    if frame.columns.has_duplicates:
        raise ValueError("history contains duplicate asset-address columns")
    return frame.sort_index()


def history_fingerprint(history: pd.DataFrame) -> str:
    """Stable content key so Streamlit never reuses results for a replaced CSV."""

    digest = hashlib.sha256()
    digest.update("\x1f".join(str(column) for column in history.columns).encode())
    digest.update(pd.util.hash_pandas_object(history, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def prepare_price_history(
    snapshot: PortfolioSnapshot,
    history: pd.DataFrame,
    *,
    calibration_days: int = 365,
    maximum_staleness_days: float = 2.0,
) -> pd.DataFrame:
    """Validate chronology and return exactly the trailing calibration window."""

    if calibration_days < 2:
        raise ValueError("calibration_days must be at least 2")
    if maximum_staleness_days < 0:
        raise ValueError("maximum_staleness_days must be non-negative")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError("price history must use a DatetimeIndex")
    prepared = history.copy().sort_index()
    prepared_index = pd.DatetimeIndex(prepared.index)
    if prepared_index.tz is not None:
        prepared_index = prepared_index.tz_convert("UTC").tz_localize(None)
    prepared.index = prepared_index
    snapshot_time = pd.Timestamp(
        datetime.fromtimestamp(snapshot.block.timestamp, tz=UTC)
    ).tz_localize(None)
    if prepared.empty:
        raise ValueError("price history is empty")
    observation_times = prepared_index
    if "block_timestamp_utc" in prepared.columns:
        parsed_block_times = pd.to_datetime(
            prepared["block_timestamp_utc"],
            utc=True,
            errors="raise",
            format="mixed",
        )
        observation_times = pd.DatetimeIndex(parsed_block_times).tz_localize(None)
        if not np.array_equal(observation_times.date, prepared_index.date):
            raise ValueError("history dates do not match block_timestamp_utc")
        observation_nanoseconds = observation_times.to_numpy(dtype="datetime64[ns]").astype(
            np.int64
        )
        gaps_hours = np.diff(observation_nanoseconds) / 3_600_000_000_000
        if gaps_hours.size and np.any((gaps_hours < 23.0) | (gaps_hours > 25.0)):
            raise ValueError("history block timestamps must be approximately 24 hours apart")
    if "oracle_address" in prepared.columns:
        oracle_addresses = {str(value).lower() for value in prepared["oracle_address"].dropna()}
        if oracle_addresses != {snapshot.contracts.oracle.lower()}:
            raise ValueError("history oracle address does not match the snapshot")
    if "block_number" in prepared.columns:
        block_numbers = pd.to_numeric(prepared["block_number"], errors="raise")
        if int(block_numbers.max()) > snapshot.block.number:
            raise ValueError("price history contains a block after the snapshot")
        snapshot_rows = block_numbers == snapshot.block.number
        if snapshot_rows.any():
            snapshot_row = prepared.loc[snapshot_rows].iloc[-1]
            for position in active_positions(snapshot):
                key = position_key(position)
                if key not in prepared.columns:
                    continue
                try:
                    observed = Decimal(str(snapshot_row[key]))
                except InvalidOperation as exc:
                    raise ValueError(
                        f"history price at the snapshot block is invalid for {position.symbol}"
                    ) from exc
                tolerance = Decimal(1) / Decimal(2 * position.oracle_base_currency_unit)
                if not observed.is_finite() or abs(observed - position.price_usd) > tolerance:
                    raise ValueError(
                        "history price at the snapshot block does not match the "
                        f"Aave-oracle snapshot for {position.symbol}"
                    )
    latest_observation = pd.Timestamp(observation_times.max())
    if latest_observation > snapshot_time:
        raise ValueError("price history contains observations after the snapshot timestamp")
    staleness_days = (snapshot_time - latest_observation).total_seconds() / 86_400
    if staleness_days > maximum_staleness_days:
        raise ValueError(
            f"price history ends {staleness_days:.2f} days before the snapshot; "
            f"maximum is {maximum_staleness_days:.2f}"
        )
    required_prices = calibration_days + 1
    if len(prepared) < required_prices:
        raise ValueError(
            f"price history needs at least {required_prices} daily prices "
            f"for {calibration_days} returns"
        )
    return prepared.iloc[-required_prices:]


def missing_history_assets(
    snapshot: PortfolioSnapshot,
    history: pd.DataFrame,
) -> tuple[AssetPosition, ...]:
    columns = set(str(column).lower() for column in history.columns)
    return tuple(
        position for position in active_positions(snapshot) if position_key(position) not in columns
    )


def simulate_snapshot(
    snapshot: PortfolioSnapshot,
    history: pd.DataFrame,
    *,
    config: SimulationConfig | None = None,
    reconciliation_tolerance_bps: Decimal | int | str = Decimal("5"),
) -> tuple[MarketCalibration, SimulationResult]:
    """Calibrate and simulate all active collateral and debt assets.

    Missing assets fail closed instead of receiving a guessed volatility or
    correlation. History must already be daily, aligned, and cleaned by the
    data layer.
    """

    _require_complete_emode_data(snapshot)
    reconciliation = reconcile_health_factor(
        snapshot,
        tolerance_bps=reconciliation_tolerance_bps,
    )
    if not reconciliation.within_tolerance:
        raise ValueError(
            "asset-level HF/collateral/debt values do not reconcile with "
            "Pool.getUserAccountData; risk simulation is disabled"
        )
    positions = active_positions(snapshot)
    if not positions:
        raise ValueError("snapshot has no supplied or borrowed positions")
    missing = missing_history_assets(snapshot, history)
    if missing:
        symbols = ", ".join(position.symbol for position in missing)
        raise ValueError(f"price history is missing active assets: {symbols}")
    history = prepare_price_history(snapshot, history)

    keys = [position_key(position) for position in positions]
    aligned_history = history.loc[:, keys]
    calibration = calibrate_log_returns(aligned_history)
    collateral_units = np.asarray(
        [float(position.collateral_balance) for position in positions],
        dtype=np.float64,
    )
    debt_units = np.asarray(
        [float(position.debt_balance) for position in positions],
        dtype=np.float64,
    )
    thresholds = np.asarray(
        [
            float(position.effective_liquidation_threshold)
            if position.contributes_collateral
            else 0.0
            for position in positions
        ],
        dtype=np.float64,
    )
    spot = np.asarray([float(position.price_usd) for position in positions])
    revalue = make_linear_portfolio_revaluer(
        collateral_units,
        debt_units,
        thresholds,
    )
    result = simulate_from_calibration(
        spot_prices_usd=spot,
        calibration=calibration,
        revalue_fn=revalue,
        config=config,
    )
    return calibration, result


def _require_complete_emode_data(snapshot: PortfolioSnapshot) -> None:
    if not snapshot.emode_data_complete:
        raise ValueError("snapshot eMode metadata is incomplete; risk model execution is disabled")


def _total_variable_debt_base_raw(snapshot: PortfolioSnapshot) -> int:
    total_debt_base_raw = 0
    for position in snapshot.positions:
        if position.variable_debt_balance_raw <= 0:
            continue
        unit = token_unit(position.decimals)
        debt_numerator = position.variable_debt_balance_raw * position.oracle_price_raw
        total_debt_base_raw += debt_numerator // unit + int(debt_numerator % unit != 0)
    return total_debt_base_raw


def _validate_liquidation_snapshot(snapshot: PortfolioSnapshot) -> None:
    revision = snapshot.contracts.pool_revision
    if revision not in SUPPORTED_LIQUIDATION_POOL_REVISIONS:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_LIQUIDATION_POOL_REVISIONS))
        raise ValueError(
            f"unsupported Aave Pool revision {revision!r}; validated revisions: {supported}"
        )
    if snapshot.base_currency_unit != 10**8:
        raise ValueError(
            "V3.7 close-factor and dust thresholds require an 8-decimal USD oracle base"
        )
    _require_complete_emode_data(snapshot)
    if any(position.stable_debt_balance_raw > 0 for position in snapshot.positions):
        raise ValueError(
            "Pool revision 11 liquidation modelling does not support non-zero stable debt; "
            "V3.7 liquidation logic operates on variable debt"
        )


def liquidation_health_factor_wad(snapshot: PortfolioSnapshot) -> int:
    """Rebuild V3.7's integer health factor for liquidation eligibility."""

    _validate_liquidation_snapshot(snapshot)
    adjusted_collateral_base_bps = 0
    for position in snapshot.positions:
        if not position.contributes_collateral:
            continue
        unit = token_unit(position.decimals)
        collateral_base_raw = position.collateral_balance_raw * position.oracle_price_raw // unit
        threshold_bps = (
            position.reserve.liquidation_threshold_bps
            if position.effective_liquidation_threshold_bps is None
            else position.effective_liquidation_threshold_bps
        )
        adjusted_collateral_base_bps += collateral_base_raw * threshold_bps

    total_debt_base_raw = _total_variable_debt_base_raw(snapshot)
    if total_debt_base_raw == 0:
        raise ValueError("a debt-free account cannot be liquidated")

    # GenericLogic first applies half-up wadDiv, then divides the weighted
    # liquidation-threshold numerator by 10_000.
    return (
        (adjusted_collateral_base_bps * 10**18 + total_debt_base_raw // 2) // total_debt_base_raw
    ) // 10_000


def liquidation_inputs_for_pair(
    snapshot: PortfolioSnapshot,
    collateral: AssetPosition,
    debt: AssetPosition,
    *,
    debt_to_cover_raw: int | None = None,
) -> LiquidationInputs:
    """Adapt a (possibly shocked) domain snapshot to exact V3.7 pair math."""

    health_factor_wad = liquidation_health_factor_wad(snapshot)

    if collateral.collateral_balance_raw <= 0 or not collateral.contributes_collateral:
        raise ValueError("selected collateral is unavailable or not enabled")
    if debt.variable_debt_balance_raw <= 0:
        raise ValueError("selected debt asset has no borrower balance")
    _validate_liquidation_reserve(snapshot, collateral, role="collateral")
    _validate_liquidation_reserve(snapshot, debt, role="debt")

    total_debt_base_raw = _total_variable_debt_base_raw(snapshot)
    effective_bonus = getattr(
        collateral,
        "effective_liquidation_bonus_bps",
        None,
    )
    reserve_bonus = collateral.reserve.liquidation_bonus_bps
    uses_emode_bonus = effective_bonus is not None and effective_bonus != reserve_bonus
    protocol_fee = collateral.reserve.liquidation_protocol_fee_bps
    if protocol_fee is None:
        raise ValueError("selected collateral liquidation protocol fee is unavailable")
    return LiquidationInputs(
        health_factor_wad=health_factor_wad,
        total_debt_base_raw=total_debt_base_raw,
        borrower_collateral_balance_raw=collateral.collateral_balance_raw,
        borrower_debt_balance_raw=debt.variable_debt_balance_raw,
        collateral_price_base_raw=collateral.oracle_price_raw,
        debt_price_base_raw=debt.oracle_price_raw,
        collateral_decimals=collateral.decimals,
        debt_decimals=debt.decimals,
        reserve_liquidation_bonus_bps=reserve_bonus,
        debt_to_cover_raw=debt_to_cover_raw,
        liquidation_protocol_fee_bps=protocol_fee,
        borrower_emode_category=(snapshot.user_emode_category if uses_emode_bonus else 0),
        collateral_enabled_in_emode=uses_emode_bonus,
        emode_liquidation_bonus_bps=(effective_bonus if uses_emode_bonus else None),
    )


def _validate_liquidation_reserve(
    snapshot: PortfolioSnapshot,
    position: AssetPosition,
    *,
    role: str,
) -> None:
    reserve = position.reserve
    if not reserve.is_active:
        raise ValueError(f"selected {role} reserve is inactive")
    if reserve.is_paused is None:
        raise ValueError(f"selected {role} reserve paused state is unavailable")
    if reserve.is_paused:
        raise ValueError(f"selected {role} reserve is paused")
    grace_period = reserve.liquidation_grace_period_until
    if grace_period is None:
        raise ValueError(f"selected {role} liquidation grace period is unavailable")
    if grace_period >= snapshot.block.timestamp:
        raise ValueError(f"selected {role} reserve is inside its liquidation grace period")


def validate_aave_flash_loan(debt: AssetPosition, amount_raw: int) -> None:
    """Fail closed unless the modelled Aave flash loan is currently feasible."""

    if amount_raw <= 0:
        raise ValueError("flash-loan amount must be positive")
    enabled = debt.reserve.flash_loan_enabled
    if enabled is None:
        raise ValueError("debt reserve flash-loan state is unavailable")
    if not enabled:
        raise ValueError("debt reserve has Aave flash loans disabled")
    if debt.available_liquidity_raw is None:
        raise ValueError("debt reserve available liquidity is unavailable")
    if debt.available_liquidity_raw < amount_raw:
        raise ValueError("debt reserve available liquidity is below the required flash principal")
    if debt.a_token_total_supply_raw is None:
        raise ValueError("debt reserve aToken total supply is unavailable")
    if debt.a_token_total_supply_raw < amount_raw:
        raise ValueError("debt reserve aToken total supply is below the required flash principal")
    if debt.virtual_underlying_balance_raw is None:
        raise ValueError("debt reserve virtual underlying balance is unavailable")
    if debt.virtual_underlying_balance_raw < amount_raw:
        raise ValueError(
            "debt reserve virtual underlying balance is below the required flash principal"
        )


def validate_liquidation_route_liquidity(
    collateral: AssetPosition,
    debt: AssetPosition,
    liquidation: LiquidationResult,
) -> None:
    """Validate flash-out and receive-underlying liquidity in callback order.

    The model assumes ``receiveAToken=False``. A cross-asset route needs the
    debt flash principal and seized collateral to be independently available.
    For a same-asset route the actual token balance must cover principal plus
    seizure because collateral is transferred before the final flash repayment.
    Aave's virtual balance credits the debt repayment before subtracting seized
    collateral, so principal and seizure are validated independently there.
    """

    principal = liquidation.actual_debt_to_liquidate_raw
    seized = liquidation.collateral_to_liquidator_raw
    validate_aave_flash_loan(debt, principal)

    same_asset = collateral.asset_address.lower() == debt.asset_address.lower()
    required_actual_liquidity = seized + principal if same_asset else seized
    if collateral.available_liquidity_raw is None:
        raise ValueError("collateral reserve available liquidity is unavailable")
    if collateral.available_liquidity_raw < required_actual_liquidity:
        qualifier = (
            "combined flash principal and seized collateral" if same_asset else "seized collateral"
        )
        raise ValueError(
            f"collateral reserve available liquidity is below the required {qualifier}"
        )
    if collateral.virtual_underlying_balance_raw is None:
        raise ValueError("collateral reserve virtual underlying balance is unavailable")
    if collateral.virtual_underlying_balance_raw < seized:
        raise ValueError(
            "collateral reserve virtual underlying balance is below the required seized collateral"
        )


def fixed_execution_quote(
    liquidation: LiquidationResult,
    collateral: AssetPosition,
    debt: AssetPosition,
    *,
    price_impact_bps: int = 30,
    slippage_buffer_bps: int = 50,
) -> SwapQuote:
    """Build a transparent offline pricing assumption for the seized collateral.

    A same-asset liquidation needs no swap, so proceeds are the seized raw amount
    with no price-impact or slippage haircut. Cross-asset pairs use Aave-oracle
    relative value and the caller's explicit haircuts.
    """

    for name, value in (
        ("price_impact_bps", price_impact_bps),
        ("slippage_buffer_bps", slippage_buffer_bps),
    ):
        if not 0 <= value < 10_000:
            raise ValueError(f"{name} must be between 0 and 9,999")
    if collateral.asset_address.lower() == debt.asset_address.lower():
        if collateral.decimals != debt.decimals:
            raise ValueError("the same asset cannot have conflicting token decimals")
        return SwapQuote(
            sell_asset=collateral.asset_address,
            buy_asset=debt.asset_address,
            sell_amount_raw=liquidation.collateral_to_liquidator_raw,
            expected_buy_amount_raw=liquidation.collateral_to_liquidator_raw,
            minimum_buy_amount_raw=liquidation.collateral_to_liquidator_raw,
            source="same-asset-no-swap",
            block_number=None,
        )
    oracle_debt_raw = (
        liquidation.collateral_to_liquidator_raw
        * collateral.oracle_price_raw
        * token_unit(debt.decimals)
        // (token_unit(collateral.decimals) * debt.oracle_price_raw)
    )
    expected = oracle_debt_raw * (10_000 - price_impact_bps) // 10_000
    minimum = expected * (10_000 - slippage_buffer_bps) // 10_000
    if minimum <= 0:
        raise ValueError("execution assumptions reduce swap proceeds to zero")
    return SwapQuote(
        sell_asset=collateral.asset_address,
        buy_asset=debt.asset_address,
        sell_amount_raw=liquidation.collateral_to_liquidator_raw,
        expected_buy_amount_raw=expected,
        minimum_buy_amount_raw=minimum,
        source="fixed-aave-oracle-haircut",
        block_number=None,
        issues=("offline assumption: Aave-oracle relative value is not a live route quote",),
    )


def gas_cost_in_debt_raw(
    *,
    gas_units: int,
    gas_price_gwei: float,
    native_price_usd: float,
    debt_price_usd: float,
    debt_decimals: int,
) -> int:
    """Convert an EIP-1559 gas budget into raw debt-token units."""

    if gas_units < 0 or gas_price_gwei < 0 or native_price_usd <= 0 or debt_price_usd <= 0:
        raise ValueError("gas inputs must be non-negative and prices must be positive")
    debt_tokens = (
        Decimal(gas_units)
        * Decimal(str(gas_price_gwei))
        * Decimal("1e-9")
        * Decimal(str(native_price_usd))
        / Decimal(str(debt_price_usd))
    )
    debt_raw = debt_tokens * Decimal(token_unit(debt_decimals))
    return int(debt_raw.to_integral_value(rounding=ROUND_CEILING))


def validate_liquidator_taker(snapshot: PortfolioSnapshot, taker: str) -> None:
    """Reject Aave's forbidden self-liquidation caller relationship."""

    if taker.strip().lower() == snapshot.user_address.strip().lower():
        raise ValueError("liquidator/taker cannot equal the borrower (self-liquidation)")


def validate_quote_block_lag(
    *,
    quote_block_number: int | None,
    snapshot_block_number: int,
    maximum_lag_blocks: int = 5,
) -> int:
    """Fail closed when a live quote does not identify a nearby pricing block."""

    if quote_block_number is None:
        raise ValueError("0x quote did not identify its pricing block")
    if min(quote_block_number, snapshot_block_number, maximum_lag_blocks) < 0:
        raise ValueError("quote/snapshot blocks and maximum lag must be non-negative")
    block_lag = abs(quote_block_number - snapshot_block_number)
    if block_lag > maximum_lag_blocks:
        raise ValueError(
            f"0x quote block differs from the portfolio snapshot by {block_lag} blocks"
        )
    return block_lag


def validate_quote_network_fee_budget(
    *,
    quote_network_fee_native_raw: int | None,
    gas_units: int,
    gas_price_gwei: float,
) -> int:
    """Ensure the manual full-transaction gas budget covers a quoted route fee."""

    if gas_units < 0 or gas_price_gwei < 0:
        raise ValueError("gas units and gas price must be non-negative")
    manual_budget_native_raw = int(
        Decimal(gas_units) * Decimal(str(gas_price_gwei)) * Decimal("1e9")
    )
    if quote_network_fee_native_raw is None:
        return manual_budget_native_raw
    if quote_network_fee_native_raw < 0:
        raise ValueError("quote network fee must be non-negative")
    if manual_budget_native_raw < quote_network_fee_native_raw:
        raise ValueError(
            "manual full-transaction gas budget is below the 0x route network-fee estimate"
        )
    return manual_budget_native_raw


__all__ = [
    "SUPPORTED_LIQUIDATION_POOL_REVISIONS",
    "active_positions",
    "apply_price_shocks",
    "load_price_history",
    "fixed_execution_quote",
    "gas_cost_in_debt_raw",
    "history_fingerprint",
    "liquidation_health_factor_wad",
    "liquidation_inputs_for_pair",
    "missing_history_assets",
    "position_key",
    "prepare_price_history",
    "shocked_totals",
    "simulate_snapshot",
    "snapshot_table",
    "validate_aave_flash_loan",
    "validate_liquidator_taker",
    "validate_liquidation_route_liquidity",
    "validate_quote_block_lag",
    "validate_quote_network_fee_budget",
]
