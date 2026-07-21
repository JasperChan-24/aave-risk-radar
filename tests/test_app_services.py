from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aave_risk_monitor.app.demo import (
    DEMO_WARNING,
    USDC,
    WBTC,
    WETH,
    build_demo_history,
    build_demo_snapshot,
)
from aave_risk_monitor.app.services import (
    SUPPORTED_LIQUIDATION_POOL_REVISIONS,
    active_positions,
    apply_price_shocks,
    fixed_execution_quote,
    gas_cost_in_debt_raw,
    history_fingerprint,
    liquidation_health_factor_wad,
    liquidation_inputs_for_pair,
    load_price_history,
    missing_history_assets,
    position_key,
    prepare_price_history,
    simulate_snapshot,
    snapshot_table,
    validate_aave_flash_loan,
    validate_liquidation_route_liquidity,
    validate_liquidator_taker,
    validate_quote_block_lag,
    validate_quote_network_fee_budget,
)
from aave_risk_monitor.data import load_snapshot
from aave_risk_monitor.domain import PortfolioSnapshot, calculate_portfolio_totals
from aave_risk_monitor.models import (
    LiquidatorPnLInputs,
    calculate_liquidation,
    calculate_liquidator_pnl,
)
from aave_risk_monitor.simulation import SimulationConfig

ROOT = Path(__file__).resolve().parents[1]


def _snapshot_with_revision(
    snapshot: PortfolioSnapshot,
    revision: int | None = 11,
) -> PortfolioSnapshot:
    return replace(
        snapshot,
        contracts=replace(snapshot.contracts, pool_revision=revision),
    )


def _position(snapshot: PortfolioSnapshot, symbol: str):
    return next(position for position in snapshot.positions if position.symbol == symbol)


def test_demo_fixture_uses_asset_level_balances_prices_and_thresholds() -> None:
    snapshot = build_demo_snapshot()
    totals = calculate_portfolio_totals(snapshot.positions)

    assert [position.symbol for position in active_positions(snapshot)] == [
        "WETH",
        "WBTC",
        "USDC",
    ]
    assert _position(snapshot, "WETH").effective_liquidation_threshold == Decimal("0.825")
    assert _position(snapshot, "WBTC").effective_liquidation_threshold == Decimal("0.8")
    assert _position(snapshot, "USDC").debt_balance == Decimal("20000")
    assert totals.collateral_value_usd == Decimal("29000")
    assert totals.debt_value_usd == Decimal("20000")
    assert totals.liquidation_adjusted_collateral_usd == Decimal("23650.0000")
    assert totals.health_factor == Decimal("1.1825")
    assert snapshot.account.health_factor == totals.health_factor
    assert snapshot.warnings == (DEMO_WARNING,)

    table = snapshot_table(snapshot).set_index("asset")
    assert table.loc["WETH", "address"] == WETH
    assert table.loc["WBTC", "supplied_value_usd"] == pytest.approx(11_000.0)
    assert table.loc["USDC", "debt_value_usd"] == pytest.approx(20_000.0)
    assert table.loc["WETH", "liquidation_threshold_pct"] == pytest.approx(82.5)
    assert table.loc["WBTC", "liquidation_threshold_pct"] == pytest.approx(80.0)
    assert table.loc["WETH", "liquidation_bonus_pct"] == pytest.approx(5.0)
    assert table.loc["WETH", "protocol_fee_pct_of_bonus"] == pytest.approx(10.0)
    assert table.loc["WETH", "oracle_source_kind"] == "synthetic"
    assert table.loc["WETH", "a_token_total_supply"] == pytest.approx(150_000)
    assert table.loc["WETH", "virtual_underlying_balance"] == pytest.approx(100_000)


def test_address_keyed_shocks_only_revalue_the_selected_asset() -> None:
    snapshot = build_demo_snapshot()
    shocked = apply_price_shocks(snapshot, {WETH.lower(): -50})

    assert _position(snapshot, "WETH").price_usd == Decimal("3600")
    assert _position(shocked, "WETH").price_usd == Decimal("1800")
    assert _position(shocked, "WBTC") == _position(snapshot, "WBTC")
    assert _position(shocked, "USDC") == _position(snapshot, "USDC")

    totals = calculate_portfolio_totals(shocked.positions)
    assert totals.collateral_value_usd == Decimal("20000")
    assert totals.debt_value_usd == Decimal("20000")
    assert totals.liquidation_adjusted_collateral_usd == Decimal("16225.0000")
    assert totals.health_factor == Decimal("0.81125")

    debt_shocked = apply_price_shocks(snapshot, {USDC.lower(): 10})
    debt_totals = calculate_portfolio_totals(debt_shocked.positions)
    assert debt_totals.collateral_value_usd == Decimal("29000")
    assert debt_totals.debt_value_usd == Decimal("22000")
    assert debt_totals.health_factor == Decimal("1.075")

    with pytest.raises(ValueError, match="zero or below"):
        apply_price_shocks(snapshot, {WETH.lower(): -100})


def test_simulation_wrapper_preserves_portfolio_asset_order_and_is_seeded() -> None:
    snapshot = build_demo_snapshot()
    history = build_demo_history(days=366).loc[:, [USDC.lower(), WBTC.lower(), WETH.lower()]]
    config = SimulationConfig(
        n_paths=257,
        horizon_steps=5,
        step_size_days=1.0,
        seed=20260720,
        chunk_size=53,
        stored_paths=11,
        bootstrap_resamples=20,
    )

    calibration, result = simulate_snapshot(snapshot, history, config=config)
    second_calibration, second_result = simulate_snapshot(snapshot, history, config=config)

    expected_order = tuple(position_key(position) for position in active_positions(snapshot))
    assert calibration.asset_names == expected_order
    assert calibration.asset_names == second_calibration.asset_names
    assert calibration.observations == 365
    assert result.sampled_price_paths_usd.shape == (11, 6, 3)
    assert result.sampled_valuation.health_factor.shape == (11, 6)
    initial_prices = result.sampled_price_paths_usd[:, 0, :]
    np.testing.assert_allclose(
        initial_prices,
        np.broadcast_to([3600, 110000, 1], initial_prices.shape),
    )
    np.testing.assert_array_equal(
        result.first_liquidation_step,
        second_result.first_liquidation_step,
    )
    np.testing.assert_array_equal(
        result.sampled_price_paths_usd,
        second_result.sampled_price_paths_usd,
    )
    probability = result.summary.liquidation_probability
    assert probability.confidence_interval is not None
    assert probability.confidence_interval.lower <= probability.value
    assert probability.value <= probability.confidence_interval.upper
    assert len(result.summary.tail_risk) == 2

    incomplete = history.drop(columns=[WBTC.lower()])
    assert [position.symbol for position in missing_history_assets(snapshot, incomplete)] == [
        "WBTC"
    ]
    with pytest.raises(ValueError, match="missing active assets: WBTC"):
        simulate_snapshot(snapshot, incomplete, config=config)


def test_history_window_rejects_lookahead_staleness_and_short_samples() -> None:
    snapshot = build_demo_snapshot()
    history = build_demo_history()
    prepared = prepare_price_history(snapshot, history)
    assert len(prepared) == 366
    assert history_fingerprint(prepared) == history_fingerprint(prepared.copy())
    changed = prepared.copy()
    changed.iloc[-1, 0] *= 1.01
    assert history_fingerprint(changed) != history_fingerprint(prepared)

    with pytest.raises(ValueError, match="after the snapshot"):
        prepare_price_history(
            snapshot,
            history.rename(
                index={history.index[-1]: history.index[-1] + pd.Timedelta(1, unit="D")}
            ),
        )
    with pytest.raises(ValueError, match="needs at least 366"):
        prepare_price_history(snapshot, history.iloc[-100:])
    with pytest.raises(ValueError, match="price history ends"):
        prepare_price_history(snapshot, history.shift(freq="-10D"))

    metadata_history = history.copy()
    metadata_history["block_timestamp_utc"] = metadata_history.index.tz_localize("UTC").astype(str)
    metadata_history["block_number"] = np.arange(len(metadata_history))
    metadata_history["oracle_address"] = snapshot.contracts.oracle
    metadata_snapshot = replace(
        snapshot,
        block=replace(snapshot.block, number=len(metadata_history) - 1),
    )
    assert len(prepare_price_history(metadata_snapshot, metadata_history)) == 366

    same_date_lookahead = metadata_history.copy()
    same_date_lookahead.iloc[-1, same_date_lookahead.columns.get_loc("block_timestamp_utc")] = (
        "2026-07-20T01:00:00+00:00"
    )
    with pytest.raises(ValueError, match="after the snapshot timestamp"):
        prepare_price_history(metadata_snapshot, same_date_lookahead)


def test_checked_in_history_cache_verifies_snapshot_provenance() -> None:
    snapshot_path = (
        ROOT / "snapshots" / "1-ed0c6079229e2d407672a117c22b62064f4a4312-block-25573974.json"
    )
    history_path = ROOT / "data" / "history" / "ethereum-v3-whale-block-25573974.csv"
    snapshot = load_snapshot(snapshot_path)

    history = load_price_history(history_path, snapshot=snapshot)

    assert len(history) == 366
    assert history.index[0] == pd.Timestamp("2025-07-20")
    assert history.index[-1] == pd.Timestamp("2026-07-20")
    assert int(history.iloc[-1]["block_number"]) == snapshot.block.number
    assert history.iloc[-1]["block_hash"] == snapshot.block.hash

    tampered_price = history.copy()
    tampered_price.loc[tampered_price.index[-1], WETH.lower()] *= 10
    with pytest.raises(ValueError, match="does not match.*WETH"):
        prepare_price_history(snapshot, tampered_price)

    mismatched_hash = history_path.read_text(encoding="utf-8").replace(
        snapshot.block.hash,
        "0x" + "00" * 32,
    )
    with pytest.raises(ValueError, match="block hash does not match"):
        load_price_history(StringIO(mismatched_hash), snapshot=snapshot)


def test_snapshot_history_loader_requires_integral_blocks_and_metadata() -> None:
    snapshot = replace(
        build_demo_snapshot(),
        block=replace(
            build_demo_snapshot().block,
            number=2,
            hash="0x" + "aa" * 32,
        ),
    )
    valid = pd.DataFrame(
        {
            "date": ["2026-07-19", "2026-07-20"],
            "block_number": [1, 2],
            "block_hash": ["0x" + "bb" * 32, snapshot.block.hash],
            "block_timestamp_utc": [
                "2026-07-19T00:00:00+00:00",
                "2026-07-20T00:00:00+00:00",
            ],
            "oracle_address": [snapshot.contracts.oracle] * 2,
            WETH.lower(): [3_500, 3_600],
        }
    )

    fractional = valid.copy()
    fractional["block_number"] = fractional["block_number"].astype(float)
    fractional.loc[0, "block_number"] = 1.5
    with pytest.raises(ValueError, match="must be integers"):
        load_price_history(StringIO(fractional.to_csv(index=False)), snapshot=snapshot)

    missing = valid.drop(columns="oracle_address")
    with pytest.raises(ValueError, match="missing reproducibility columns"):
        load_price_history(StringIO(missing.to_csv(index=False)), snapshot=snapshot)


@pytest.mark.parametrize("revision", [None, 10, 12])
def test_liquidation_adapter_fails_closed_for_unvalidated_pool_revisions(
    revision: int | None,
) -> None:
    snapshot = _snapshot_with_revision(build_demo_snapshot(), revision)
    collateral = _position(snapshot, "WETH")
    debt = _position(snapshot, "USDC")

    with pytest.raises(ValueError, match="unsupported Aave Pool revision"):
        liquidation_inputs_for_pair(snapshot, collateral, debt)


def test_liquidation_adapter_requires_v37_usd_base_assumption() -> None:
    snapshot = _snapshot_with_revision(build_demo_snapshot())
    unsupported_base = replace(
        snapshot,
        base_currency_unit=10**18,
        account=replace(snapshot.account, base_currency_unit=10**18),
    )

    with pytest.raises(ValueError, match="8-decimal USD oracle base"):
        liquidation_inputs_for_pair(
            unsupported_base,
            _position(unsupported_base, "WETH"),
            _position(unsupported_base, "USDC"),
        )


def test_incomplete_emode_metadata_disables_all_risk_model_entrypoints() -> None:
    snapshot = replace(build_demo_snapshot(), emode_data_complete=False)
    collateral = _position(snapshot, "WETH")
    debt = _position(snapshot, "USDC")

    with pytest.raises(ValueError, match="eMode metadata is incomplete"):
        simulate_snapshot(snapshot, build_demo_history())
    with pytest.raises(ValueError, match="eMode metadata is incomplete"):
        liquidation_health_factor_wad(snapshot)
    with pytest.raises(ValueError, match="eMode metadata is incomplete"):
        liquidation_inputs_for_pair(snapshot, collateral, debt)


def test_simulation_entrypoint_rejects_unreconciled_asset_level_totals() -> None:
    snapshot = build_demo_snapshot()
    scaled = replace(
        snapshot,
        positions=tuple(
            replace(
                position,
                collateral_balance_raw=position.collateral_balance_raw // 2,
                variable_debt_balance_raw=position.variable_debt_balance_raw // 2,
            )
            for position in snapshot.positions
        ),
    )

    with pytest.raises(ValueError, match="do not reconcile"):
        simulate_snapshot(scaled, build_demo_history())


def test_integer_liquidation_health_factor_helper_matches_adapter() -> None:
    snapshot = apply_price_shocks(build_demo_snapshot(), {WETH.lower(): -25})
    inputs = liquidation_inputs_for_pair(
        snapshot,
        _position(snapshot, "WETH"),
        _position(snapshot, "USDC"),
    )

    assert liquidation_health_factor_wad(snapshot) == 996_875_000_000_000_000
    assert liquidation_health_factor_wad(snapshot) == inputs.health_factor_wad


def test_liquidation_adapter_rejects_nonzero_stable_debt_for_v37() -> None:
    snapshot = build_demo_snapshot()
    debt = _position(snapshot, "USDC")
    stable_debt = replace(
        debt,
        stable_debt_balance_raw=1,
        variable_debt_balance_raw=debt.variable_debt_balance_raw - 1,
    )
    snapshot = replace(
        snapshot,
        positions=tuple(
            stable_debt if position.symbol == "USDC" else position
            for position in snapshot.positions
        ),
    )

    with pytest.raises(ValueError, match="non-zero stable debt"):
        liquidation_inputs_for_pair(
            snapshot,
            _position(snapshot, "WETH"),
            stable_debt,
        )


def test_liquidation_adapter_fails_closed_on_pause_or_grace_period() -> None:
    snapshot = build_demo_snapshot()
    collateral = _position(snapshot, "WETH")
    debt = _position(snapshot, "USDC")

    paused_collateral = replace(
        collateral,
        reserve=replace(collateral.reserve, is_paused=True),
    )
    paused_snapshot = replace(
        snapshot,
        positions=tuple(
            paused_collateral if position.symbol == "WETH" else position
            for position in snapshot.positions
        ),
    )
    with pytest.raises(ValueError, match="collateral reserve is paused"):
        liquidation_inputs_for_pair(paused_snapshot, paused_collateral, debt)

    grace_debt = replace(
        debt,
        reserve=replace(
            debt.reserve,
            liquidation_grace_period_until=snapshot.block.timestamp,
        ),
    )
    grace_snapshot = replace(
        snapshot,
        positions=tuple(
            grace_debt if position.symbol == "USDC" else position for position in snapshot.positions
        ),
    )
    with pytest.raises(ValueError, match="debt reserve is inside"):
        liquidation_inputs_for_pair(grace_snapshot, collateral, grace_debt)


def test_aave_flash_loan_guard_checks_flag_metadata_and_liquidity() -> None:
    debt = _position(build_demo_snapshot(), "USDC")
    validate_aave_flash_loan(debt, 20_000 * 10**6)

    with pytest.raises(ValueError, match="disabled"):
        validate_aave_flash_loan(
            replace(debt, reserve=replace(debt.reserve, flash_loan_enabled=False)),
            1,
        )
    with pytest.raises(ValueError, match="state is unavailable"):
        validate_aave_flash_loan(
            replace(debt, reserve=replace(debt.reserve, flash_loan_enabled=None)),
            1,
        )
    with pytest.raises(ValueError, match="available liquidity is below"):
        validate_aave_flash_loan(replace(debt, available_liquidity_raw=10), 11)
    with pytest.raises(ValueError, match="aToken total supply is unavailable"):
        validate_aave_flash_loan(replace(debt, a_token_total_supply_raw=None), 1)
    with pytest.raises(ValueError, match="aToken total supply is below"):
        validate_aave_flash_loan(replace(debt, a_token_total_supply_raw=10), 11)
    with pytest.raises(ValueError, match="virtual underlying balance is unavailable"):
        validate_aave_flash_loan(
            replace(debt, virtual_underlying_balance_raw=None),
            1,
        )
    with pytest.raises(ValueError, match="virtual underlying balance is below"):
        validate_aave_flash_loan(
            replace(debt, virtual_underlying_balance_raw=10),
            11,
        )


def test_liquidation_adapter_fixed_quote_and_gas_feed_complete_pnl() -> None:
    assert frozenset({11}) == SUPPORTED_LIQUIDATION_POOL_REVISIONS
    snapshot = _snapshot_with_revision(build_demo_snapshot())
    scenario = apply_price_shocks(snapshot, {WETH.lower(): -25})
    collateral = _position(scenario, "WETH")
    debt = _position(scenario, "USDC")

    inputs = liquidation_inputs_for_pair(scenario, collateral, debt)
    assert inputs.health_factor_wad == 996_875_000_000_000_000
    assert inputs.total_debt_base_raw == 20_000 * 10**8
    assert inputs.borrower_collateral_balance_raw == 5 * 10**18
    assert inputs.borrower_debt_balance_raw == 20_000 * 10**6

    liquidation = calculate_liquidation(inputs)
    assert liquidation.close_factor_bps == 5_000
    assert liquidation.actual_debt_to_liquidate_raw == 10_000 * 10**6

    quote = fixed_execution_quote(
        liquidation,
        collateral,
        debt,
        price_impact_bps=30,
        slippage_buffer_bps=50,
    )
    oracle_debt_raw = (
        liquidation.collateral_to_liquidator_raw
        * collateral.oracle_price_raw
        * 10**debt.decimals
        // (10**collateral.decimals * debt.oracle_price_raw)
    )
    assert quote.expected_buy_amount_raw == oracle_debt_raw * 9_970 // 10_000
    assert quote.minimum_buy_amount_raw == quote.expected_buy_amount_raw * 9_950 // 10_000
    assert quote.source == "fixed-aave-oracle-haircut"
    assert quote.issues == (
        "offline assumption: Aave-oracle relative value is not a live route quote",
    )

    gas_cost_raw = gas_cost_in_debt_raw(
        gas_units=650_000,
        gas_price_gwei=20,
        native_price_usd=3_600,
        debt_price_usd=1,
        debt_decimals=6,
    )
    assert gas_cost_raw == 46_800_000

    pnl_inputs = LiquidatorPnLInputs.from_liquidation(
        liquidation,
        collateral_asset=collateral.asset_address,
        debt_asset=debt.asset_address,
        collateral_decimals=collateral.decimals,
        debt_decimals=debt.decimals,
        flash_loan_premium_bps=5,
        gas_cost_debt_raw=gas_cost_raw,
    )
    pnl = calculate_liquidator_pnl(pnl_inputs, quote)
    assert pnl.flash_principal_raw == liquidation.actual_debt_to_liquidate_raw
    assert pnl.flash_premium_raw == 5_000_000
    assert pnl.gas_cost_raw == 46_800_000
    assert pnl.swap_proceeds_raw == quote.minimum_buy_amount_raw
    assert pnl.net_pnl_raw == (
        quote.minimum_buy_amount_raw
        - liquidation.actual_debt_to_liquidate_raw
        - pnl.flash_premium_raw
        - gas_cost_raw
    )
    assert pnl.profitable

    validate_liquidation_route_liquidity(collateral, debt, liquidation)
    with pytest.raises(ValueError, match="required seized collateral"):
        validate_liquidation_route_liquidity(
            replace(collateral, available_liquidity_raw=0),
            debt,
            liquidation,
        )
    with pytest.raises(ValueError, match="virtual underlying.*seized collateral"):
        validate_liquidation_route_liquidity(
            replace(collateral, virtual_underlying_balance_raw=0),
            debt,
            liquidation,
        )


def test_same_asset_liquidation_uses_one_to_one_proceeds_without_swap_haircuts() -> None:
    snapshot = build_demo_snapshot()
    collateral_and_debt = replace(
        _position(snapshot, "WETH"),
        variable_debt_balance_raw=7 * 10**18,
    )
    no_usdc_debt = replace(
        _position(snapshot, "USDC"),
        variable_debt_balance_raw=0,
    )
    scenario = replace(
        snapshot,
        positions=tuple(
            collateral_and_debt
            if position.symbol == "WETH"
            else no_usdc_debt
            if position.symbol == "USDC"
            else position
            for position in snapshot.positions
        ),
    )

    liquidation = calculate_liquidation(
        liquidation_inputs_for_pair(
            scenario,
            collateral_and_debt,
            collateral_and_debt,
        )
    )
    quote = fixed_execution_quote(
        liquidation,
        collateral_and_debt,
        collateral_and_debt,
        price_impact_bps=9_000,
        slippage_buffer_bps=9_000,
    )

    assert quote.expected_buy_amount_raw == liquidation.collateral_to_liquidator_raw
    assert quote.minimum_buy_amount_raw == liquidation.collateral_to_liquidator_raw
    assert quote.source == "same-asset-no-swap"
    assert quote.issues == ()

    principal = liquidation.actual_debt_to_liquidate_raw
    seized = liquidation.collateral_to_liquidator_raw
    assert seized > principal
    combined = principal + seized
    validate_liquidation_route_liquidity(
        replace(
            collateral_and_debt,
            available_liquidity_raw=combined,
            virtual_underlying_balance_raw=seized,
        ),
        replace(
            collateral_and_debt,
            available_liquidity_raw=combined,
            virtual_underlying_balance_raw=seized,
        ),
        liquidation,
    )
    constrained = replace(
        collateral_and_debt,
        available_liquidity_raw=combined - 1,
        virtual_underlying_balance_raw=combined,
    )
    with pytest.raises(ValueError, match="combined flash principal"):
        validate_liquidation_route_liquidity(
            constrained,
            constrained,
            liquidation,
        )
    virtual_constrained = replace(
        collateral_and_debt,
        available_liquidity_raw=combined,
        virtual_underlying_balance_raw=seized - 1,
    )
    with pytest.raises(ValueError, match="virtual underlying.*seized collateral"):
        validate_liquidation_route_liquidity(
            virtual_constrained,
            virtual_constrained,
            liquidation,
        )


def test_gas_adapter_rounds_fractional_raw_debt_cost_up() -> None:
    assert (
        gas_cost_in_debt_raw(
            gas_units=1,
            gas_price_gwei=1,
            native_price_usd=1,
            debt_price_usd=3,
            debt_decimals=6,
        )
        == 1
    )


def test_live_quote_execution_context_guards_are_fail_closed() -> None:
    snapshot = build_demo_snapshot()

    with pytest.raises(ValueError, match="self-liquidation"):
        validate_liquidator_taker(snapshot, snapshot.user_address.upper())
    validate_liquidator_taker(
        snapshot,
        "0x000000000000000000000000000000000000bEEF",
    )

    assert (
        validate_quote_block_lag(
            quote_block_number=105,
            snapshot_block_number=100,
        )
        == 5
    )
    with pytest.raises(ValueError, match="did not identify"):
        validate_quote_block_lag(
            quote_block_number=None,
            snapshot_block_number=100,
        )
    with pytest.raises(ValueError, match="differs.*6 blocks"):
        validate_quote_block_lag(
            quote_block_number=106,
            snapshot_block_number=100,
        )

    manual_budget = validate_quote_network_fee_budget(
        quote_network_fee_native_raw=20_000_000_000,
        gas_units=2,
        gas_price_gwei=10,
    )
    assert manual_budget == 20_000_000_000
    with pytest.raises(ValueError, match="below the 0x route"):
        validate_quote_network_fee_budget(
            quote_network_fee_native_raw=20_000_000_001,
            gas_units=2,
            gas_price_gwei=10,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"gas_units": -1},
        {"gas_price_gwei": -1},
        {"native_price_usd": 0},
        {"debt_price_usd": 0},
    ],
)
def test_gas_adapter_rejects_invalid_inputs(overrides: dict[str, float | int]) -> None:
    inputs: dict[str, float | int] = {
        "gas_units": 650_000,
        "gas_price_gwei": 20,
        "native_price_usd": 3_600,
        "debt_price_usd": 1,
        "debt_decimals": 6,
    }
    inputs.update(overrides)

    with pytest.raises(ValueError, match="gas inputs"):
        gas_cost_in_debt_raw(**inputs)  # type: ignore[arg-type]
