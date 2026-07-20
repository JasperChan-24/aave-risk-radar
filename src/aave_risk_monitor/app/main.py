"""Bilingual Streamlit dashboard for the research models."""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from web3 import Web3

from aave_risk_monitor.app.demo import build_demo_history, build_demo_snapshot
from aave_risk_monitor.app.i18n import translate
from aave_risk_monitor.app.services import (
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
    simulate_snapshot,
    snapshot_table,
    validate_liquidation_route_liquidity,
    validate_liquidator_taker,
    validate_quote_block_lag,
    validate_quote_network_fee_budget,
)
from aave_risk_monitor.config import AaveV3MarketConfig, Settings
from aave_risk_monitor.data import AaveV3DataSource, AaveV3DataSourceError
from aave_risk_monitor.domain import (
    AssetPosition,
    PortfolioSnapshot,
    calculate_portfolio_totals,
    reconcile_health_factor,
)
from aave_risk_monitor.models import (
    LiquidatorPnLInputs,
    QuoteError,
    ZeroXQuoteProvider,
    calculate_liquidation,
    calculate_liquidator_pnl,
    estimate_liquidator_pnl,
    raw_to_decimal,
    token_unit,
)
from aave_risk_monitor.simulation import SimulationConfig

DEFAULT_WHALE = "0xed0c6079229e2d407672a117c22b62064f4a4312"


def _money(value: Decimal | float) -> str:
    return f"${float(value):,.2f}"


def _hf(value: Decimal | None) -> str:
    return "∞" if value is None else f"{value:.4f}"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _position_label(position: AssetPosition) -> str:
    return f"{position.symbol} · {position.asset_address[:6]}…{position.asset_address[-4:]}"


@st.cache_resource(show_spinner=False)
def _web3(rpc_url: str) -> Web3:
    return Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 25}))


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_live_snapshot(
    rpc_url: str,
    address: str,
    addresses_provider: str,
    reconciliation_tolerance_bps: int,
) -> PortfolioSnapshot:
    web3 = _web3(rpc_url)
    if not web3.is_connected():
        raise AaveV3DataSourceError("Ethereum RPC connection failed")
    return AaveV3DataSource(
        web3,
        AaveV3MarketConfig(addresses_provider=addresses_provider),
        reconciliation_tolerance_bps=reconciliation_tolerance_bps,
    ).fetch_portfolio(address)


def _uploaded_history(uploaded: Any, snapshot: PortfolioSnapshot) -> pd.DataFrame:
    return load_price_history(uploaded, snapshot=snapshot)


def _render_portfolio(
    snapshot: PortfolioSnapshot,
    language: str,
    reconciliation_tolerance_bps: int = 5,
) -> None:
    table = snapshot_table(snapshot)
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "price_usd": st.column_config.NumberColumn(format="$%.4f"),
            "supplied_value_usd": st.column_config.NumberColumn(format="$%.2f"),
            "debt_value_usd": st.column_config.NumberColumn(format="$%.2f"),
            "ltv_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "liquidation_threshold_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "liquidation_bonus_pct": st.column_config.NumberColumn(format="%.2f%%"),
            "protocol_fee_pct_of_bonus": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )
    reconciliation = reconcile_health_factor(
        snapshot,
        tolerance_bps=reconciliation_tolerance_bps,
    )
    if reconciliation.within_tolerance:
        st.success(translate("reconciled", language))
    else:

        def display_bps(value: Decimal | None) -> str:
            return "n/a" if value is None else f"{value:.4f} bps"

        detail = "; ".join(
            (
                f"HF {display_bps(reconciliation.difference_bps)}",
                f"collateral {display_bps(reconciliation.collateral_difference_bps)}",
                f"debt {display_bps(reconciliation.debt_difference_bps)}",
            )
        )
        st.warning(f"{translate('not_reconciled', language)} · {detail}")

    with st.expander("Snapshot provenance / 快照来源"):
        contracts = snapshot.contracts
        st.json(
            {
                "chain_id": snapshot.chain_id,
                "market": snapshot.market,
                "block_number": snapshot.block.number,
                "block_hash": snapshot.block.hash,
                "block_timestamp": snapshot.block.timestamp,
                "user": snapshot.user_address,
                "pool": contracts.pool,
                "pool_implementation": getattr(contracts, "pool_implementation", None),
                "pool_revision": getattr(contracts, "pool_revision", None),
                "oracle": contracts.oracle,
                "flashloan_premium_total_bps": getattr(
                    snapshot, "flashloan_premium_total_bps", None
                ),
                "emode_category": snapshot.user_emode_category,
            }
        )
    for warning in snapshot.warnings:
        st.warning(warning)


def _render_simulation_result(result: Any, calibration: Any, language: str) -> None:
    probability = result.summary.liquidation_probability
    interval = probability.confidence_interval
    probability_text = _pct(probability.value)
    if interval is not None:
        probability_text += f" · 95% CI [{_pct(interval.lower)}, {_pct(interval.upper)}]"

    ttl_summary = result.summary.time_to_liquidation
    ttl = ttl_summary.conditional_median_days
    ttl_text = "not observed / 未观测"
    if ttl is not None:
        ttl_text = f"{ttl.value:.1f} days"
        ttl_interval = ttl.confidence_interval
        if ttl_interval is not None:
            ttl_text += f" · 95% CI [{ttl_interval.lower:.1f}, {ttl_interval.upper:.1f}]"
    col1, col2 = st.columns(2)
    col1.metric(translate("liq_probability", language), probability_text)
    col2.metric(translate("ttl", language), ttl_text)
    st.caption(
        "TTL first passages: "
        f"{ttl_summary.observed_liquidations:,} observed; "
        f"{ttl_summary.censored_paths:,} right-censored."
    )

    selected_horizon = int(result.config.horizon_days)
    horizons = list(
        dict.fromkeys(day for day in (1, 7, selected_horizon) if day <= selected_horizon)
    )
    horizon_columns = st.columns(len(horizons))
    for column, day in zip(horizon_columns, horizons, strict=True):
        step = int(day / result.config.step_size_days)
        point = float(result.cumulative_liquidation_probability[step])
        lower = float(result.cumulative_liquidation_probability_lower[step])
        upper = float(result.cumulative_liquidation_probability_upper[step])
        column.metric(
            f"P(liquidation ≤ {day}d)",
            f"{_pct(point)} · 95% CI [{_pct(lower)}, {_pct(upper)}]",
        )

    tail_rows: list[dict[str, object]] = []
    for estimate in result.summary.tail_risk:
        var_ci = estimate.value_at_risk_usd.confidence_interval
        es_ci = estimate.expected_shortfall_usd.confidence_interval
        tail_rows.append(
            {
                "level": f"{estimate.confidence_level:.0%}",
                "VaR_USD": estimate.value_at_risk_usd.value,
                "VaR_95pct_CI": (
                    "n/a" if var_ci is None else f"[{var_ci.lower:,.2f}, {var_ci.upper:,.2f}]"
                ),
                "ES_USD": estimate.expected_shortfall_usd.value,
                "ES_95pct_CI": (
                    "n/a" if es_ci is None else f"[{es_ci.lower:,.2f}, {es_ci.upper:,.2f}]"
                ),
            }
        )
    st.dataframe(pd.DataFrame(tail_rows), width="stretch", hide_index=True)

    curve = pd.DataFrame(
        {
            "liquidation_probability": result.cumulative_liquidation_probability,
            "interval_lower": result.cumulative_liquidation_probability_lower,
            "interval_upper": result.cumulative_liquidation_probability_upper,
        },
        index=np.arange(result.cumulative_liquidation_probability.size)
        * result.config.step_size_days,
    )
    curve.index.name = "day"
    st.line_chart(curve)

    sample_hf = result.sampled_valuation.health_factor[:50].T
    if sample_hf.size:
        st.caption("Sampled health-factor paths (first 50 only)")
        st.line_chart(pd.DataFrame(sample_hf))

    with st.expander("Calibration / 校准结果"):
        volatility = pd.DataFrame(
            {
                "asset": calibration.asset_names,
                "annualized_volatility": calibration.annualized_volatility,
            }
        )
        st.dataframe(volatility, width="stretch", hide_index=True)
        st.caption(
            f"{calibration.estimator}; {calibration.observations} daily returns; "
            f"shrinkage={calibration.shrinkage:.4f}; drift={result.config.drift_mode}"
        )
        st.dataframe(
            pd.DataFrame(
                calibration.correlation,
                index=calibration.asset_names,
                columns=calibration.asset_names,
            ),
            width="stretch",
        )


def _render_liquidator(
    snapshot: PortfolioSnapshot,
    scenario_snapshot: PortfolioSnapshot,
    shocks: dict[str, float],
    language: str,
) -> None:
    is_hypothetical = any(abs(value) > 1e-12 for value in shocks.values())
    if is_hypothetical:
        try:
            health_factor_wad = liquidation_health_factor_wad(scenario_snapshot)
        except ValueError as exc:
            st.error(f"Pair model unavailable: {exc}")
            return
    else:
        health_factor_wad = snapshot.account.health_factor_raw
    if health_factor_wad >= 10**18:
        st.info(translate("current_safe", language))
        return

    if is_hypothetical:
        st.warning(translate("hypothetical", language))

    collateral_positions = [
        position
        for position in active_positions(scenario_snapshot)
        if position.collateral_balance_raw > 0 and position.contributes_collateral
    ]
    debt_positions = [
        position
        for position in active_positions(scenario_snapshot)
        if position.total_debt_balance_raw > 0
    ]
    if not collateral_positions or not debt_positions:
        st.info("No executable collateral/debt pair in this snapshot.")
        return

    left, right = st.columns(2)
    collateral = left.selectbox(
        "Collateral to seize",
        collateral_positions,
        format_func=_position_label,
    )
    debt = right.selectbox(
        "Debt to repay",
        debt_positions,
        format_func=_position_label,
    )
    gas_col, fee_col, impact_col = st.columns(3)
    gas_units = int(gas_col.number_input("Gas units", min_value=0, value=650_000, step=10_000))
    gas_price_gwei = float(
        fee_col.number_input("Max fee (gwei)", min_value=0.0, value=20.0, step=1.0)
    )
    price_impact_bps = int(
        impact_col.number_input(
            "Execution impact (bps)", min_value=0, max_value=9_000, value=30, step=5
        )
    )
    slippage_bps = int(
        st.number_input(
            "Additional minimum-output buffer (bps)",
            min_value=0,
            max_value=9_000,
            value=50,
            step=5,
        )
    )
    other_cost_debt = Decimal(
        str(
            st.number_input(
                f"Other execution / MEV cost ({debt.symbol})",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )
        )
    )
    if is_hypothetical:
        quote_mode = "Fixed oracle haircut"
        st.info(
            "Live 0x pricing is disabled for shocked oracle scenarios because current-market "
            "quotes and hypothetical prices do not describe the same state."
        )
    else:
        quote_mode = st.radio(
            "Swap pricing",
            ["Fixed oracle haircut", "0x live read-only pricing quote"],
            horizontal=True,
        )
    taker = st.text_input(
        "Quote taker (read-only; no transaction is sent)",
        value="0x000000000000000000000000000000000000bEEF",
    )

    native_position = next(
        (
            position
            for position in scenario_snapshot.positions
            if position.symbol.upper() in {"WETH", "ETH"}
        ),
        None,
    )
    native_price = (
        float(native_position.price_usd)
        if native_position is not None
        else float(
            st.number_input(
                "Native token price (USD)",
                min_value=0.01,
                value=3_600.0,
            )
        )
    )

    if not st.button("Calculate pair economics", type="primary"):
        return

    try:
        if not Web3.is_address(taker):
            raise ValueError("liquidator/taker must be a valid EVM address")
        validate_liquidator_taker(scenario_snapshot, taker)
        liquidation_inputs = liquidation_inputs_for_pair(
            scenario_snapshot,
            collateral,
            debt,
        )
        liquidation = calculate_liquidation(liquidation_inputs)
        validate_liquidation_route_liquidity(collateral, debt, liquidation)

        gas_cost_raw = gas_cost_in_debt_raw(
            gas_units=gas_units,
            gas_price_gwei=gas_price_gwei,
            native_price_usd=native_price,
            debt_price_usd=float(debt.price_usd),
            debt_decimals=debt.decimals,
        )
        flash_premium = getattr(snapshot, "flashloan_premium_total_bps", None)
        if flash_premium is None:
            raise ValueError("Pool flash premium is unavailable at the snapshot block")
        pnl_inputs = LiquidatorPnLInputs.from_liquidation(
            liquidation,
            collateral_asset=collateral.asset_address,
            debt_asset=debt.asset_address,
            collateral_decimals=collateral.decimals,
            debt_decimals=debt.decimals,
            flash_loan_premium_bps=int(flash_premium),
            gas_cost_debt_raw=gas_cost_raw,
            other_cost_debt_raw=int(other_cost_debt * Decimal(token_unit(debt.decimals))),
            chain_id=scenario_snapshot.chain_id,
            taker=taker,
        )

        same_asset = collateral.asset_address.lower() == debt.asset_address.lower()
        if same_asset:
            quote = fixed_execution_quote(
                liquidation,
                collateral,
                debt,
                price_impact_bps=price_impact_bps,
                slippage_buffer_bps=slippage_bps,
            )
            pnl = calculate_liquidator_pnl(pnl_inputs, quote)
        elif quote_mode.startswith("0x"):
            pnl = estimate_liquidator_pnl(
                pnl_inputs,
                ZeroXQuoteProvider(
                    api_key=os.getenv("ZEROX_API_KEY"),
                    default_taker=taker,
                    slippage_bps=slippage_bps,
                ),
            )
            validate_quote_block_lag(
                quote_block_number=pnl.quote_block_number,
                snapshot_block_number=scenario_snapshot.block.number,
            )
        else:
            quote = fixed_execution_quote(
                liquidation,
                collateral,
                debt,
                price_impact_bps=price_impact_bps,
                slippage_buffer_bps=slippage_bps,
            )
            pnl = calculate_liquidator_pnl(pnl_inputs, quote)

        validate_quote_network_fee_budget(
            quote_network_fee_native_raw=pnl.quote_network_fee_native_raw,
            gas_units=gas_units,
            gas_price_gwei=gas_price_gwei,
        )

        token = debt.symbol
        proceeds_label = (
            "Collateral available for repayment" if same_asset else "Gross swap proceeds"
        )
        rows = [
            ("Debt repaid", raw_to_decimal(pnl.flash_principal_raw, debt.decimals)),
            (proceeds_label, raw_to_decimal(pnl.gross_swap_proceeds_raw, debt.decimals)),
            ("Flash premium", raw_to_decimal(pnl.flash_premium_raw, debt.decimals)),
            ("Gas", raw_to_decimal(pnl.gas_cost_raw, debt.decimals)),
            ("Other execution / MEV", raw_to_decimal(pnl.other_cost_raw, debt.decimals)),
            ("Slippage buffer", raw_to_decimal(pnl.slippage_cost_raw, debt.decimals)),
            ("Quote fees", raw_to_decimal(pnl.quote_fee_raw, debt.decimals)),
            ("Net P&L", pnl.net_pnl),
        ]
        st.dataframe(
            pd.DataFrame(
                [{"component": name, f"amount_{token}": float(value)} for name, value in rows]
            ),
            width="stretch",
            hide_index=True,
        )
        outcome = f"{pnl.net_pnl:,.6f} {token}"
        if pnl.profitable and not pnl.quote_issues:
            st.success(f"Positive modelled net P&L: {outcome}")
        elif pnl.profitable:
            st.warning(
                f"Positive arithmetic net P&L, but unresolved quote issues prevent an "
                f"opportunity classification: {outcome}"
            )
        else:
            st.error(f"Negative modelled net P&L: {outcome}")
        st.caption(
            f"close factor={liquidation.close_factor_bps / 100:.0f}%; "
            f"quote={pnl.quote_source}; break-even={pnl.break_even_debt_per_collateral:,.8f} "
            f"{token}/{collateral.symbol}"
        )
        if pnl.quote_network_fee_native_raw is not None:
            st.caption(
                "0x reported route network fee: "
                f"{pnl.quote_network_fee_native_raw:,} native raw units. It must fit inside "
                "the manual full-transaction gas budget and is not charged twice."
            )
        for issue in pnl.quote_issues:
            st.warning(issue)
    except (ValueError, QuoteError) as exc:
        st.error(f"Pair model unavailable: {exc}")


def main() -> None:
    st.set_page_config(page_title="Aave Whale Risk Monitor", page_icon="🌊", layout="wide")
    load_dotenv(Path.cwd() / ".env", override=False)

    language = st.sidebar.radio("Language / 语言", ["English", "中文"], horizontal=True)
    tr = lambda key: translate(key, language)  # noqa: E731
    st.title(f"🌊 {tr('title')}")
    st.caption(tr("subtitle"))
    st.sidebar.header("Data & model / 数据与模型")
    mode = st.sidebar.radio(tr("data_mode"), [tr("demo"), tr("live")])
    is_demo = mode == tr("demo")

    snapshot: PortfolioSnapshot
    reconciliation_tolerance_bps = 5
    uploaded_history = None
    if is_demo:
        snapshot = build_demo_snapshot()
    else:
        address = st.sidebar.text_input(tr("address"), DEFAULT_WHALE)
        rpc_url = os.getenv("ALCHEMY_RPC_URL") or os.getenv("ETHEREUM_RPC_URL") or ""
        if not rpc_url:
            st.error(tr("rpc_missing"))
            st.stop()
        try:
            settings = Settings.from_env()
            reconciliation_tolerance_bps = settings.hf_reconciliation_tolerance_bps
            with st.spinner("Reading one block-consistent Aave snapshot…"):
                snapshot = _fetch_live_snapshot(
                    rpc_url,
                    address,
                    settings.market.addresses_provider,
                    settings.hf_reconciliation_tolerance_bps,
                )
        except (ValueError, AaveV3DataSourceError) as exc:
            st.error(f"Live snapshot failed: {exc}")
            st.stop()

    horizon = int(st.sidebar.select_slider(tr("horizon"), options=[1, 7, 30], value=30))
    paths = int(
        st.sidebar.select_slider(tr("paths"), options=[2_000, 5_000, 10_000, 20_000], value=20_000)
    )
    seed = int(st.sidebar.number_input(tr("seed"), min_value=0, value=20_260_720))

    totals = calculate_portfolio_totals(snapshot.positions)
    snapshot_reconciliation = reconcile_health_factor(
        snapshot,
        tolerance_bps=reconciliation_tolerance_bps,
    )
    model_inputs_valid = snapshot_reconciliation.within_tolerance and snapshot.emode_data_complete
    metric_columns = st.columns(4)
    metric_columns[0].metric(tr("collateral"), _money(totals.collateral_value_usd))
    metric_columns[1].metric(tr("debt"), _money(totals.debt_value_usd))
    metric_columns[2].metric(tr("health_factor"), _hf(totals.health_factor))
    metric_columns[3].metric(tr("onchain_hf"), _hf(snapshot.account.health_factor))
    st.caption(
        f"{snapshot.market} · block {snapshot.block.number:,} · "
        f"Pool revision {getattr(snapshot.contracts, 'pool_revision', None) or 'fixture'}"
    )

    portfolio_tab, stress_tab, simulation_tab, liquidator_tab, methodology_tab = st.tabs(
        [tr("portfolio"), tr("stress"), tr("simulation"), tr("liquidator"), tr("methodology")]
    )

    with portfolio_tab:
        _render_portfolio(snapshot, language, reconciliation_tolerance_bps)

    scenario_snapshot = snapshot
    shocks: dict[str, float] = {position_key(position): 0.0 for position in snapshot.positions}
    with stress_tab:
        if not snapshot.emode_data_complete:
            st.error("Stress analysis is disabled because required eMode metadata is incomplete.")
        elif not snapshot_reconciliation.within_tolerance:
            st.error(
                "Stress analysis is disabled because asset-level values do not reconcile "
                "with Pool.getUserAccountData."
            )
        else:
            st.write(tr("shock_help"))
            stress_rows = [
                {
                    "asset": position.symbol,
                    "address": position_key(position),
                    "current_price_usd": float(position.price_usd),
                    "shock_pct": 0.0,
                }
                for position in active_positions(snapshot)
            ]
            edited = st.data_editor(
                pd.DataFrame(stress_rows),
                width="stretch",
                hide_index=True,
                disabled=["asset", "address", "current_price_usd"],
                column_config={
                    "shock_pct": st.column_config.NumberColumn(
                        min_value=-99.0,
                        max_value=500.0,
                        step=1.0,
                        format="%.1f%%",
                    )
                },
                key=f"stress-{snapshot.user_address}-{snapshot.block.number}",
            )
            shocks = {
                str(row["address"]): float(row["shock_pct"])
                for row in edited.to_dict(orient="records")
            }
            scenario_snapshot = apply_price_shocks(snapshot, shocks)
            scenario_totals = calculate_portfolio_totals(scenario_snapshot.positions)
            st.metric(
                "Stressed HF / 压力后 HF",
                _hf(scenario_totals.health_factor),
                delta=(
                    None
                    if totals.health_factor is None or scenario_totals.health_factor is None
                    else f"{scenario_totals.health_factor - totals.health_factor:+.4f}"
                ),
            )
            if scenario_totals.health_factor is not None and scenario_totals.health_factor < 1:
                st.error(tr("hypothetical"))

    with simulation_tab:
        if not snapshot.emode_data_complete:
            st.error("Risk simulation is disabled because required eMode metadata is incomplete.")
        elif not snapshot_reconciliation.within_tolerance:
            st.error(
                "Risk simulation is disabled because the asset-level snapshot does not "
                "reconcile with Pool.getUserAccountData."
            )
        if is_demo:
            history = build_demo_history()
            st.warning(
                "Offline demo uses a deterministic synthetic return fixture. "
                "Live research should upload/cache common-block Aave-oracle history."
            )
        else:
            uploaded_history = st.file_uploader(
                "Daily history CSV: date + one lower-case asset-address column per active reserve",
                type=["csv"],
            )
            history = None
            if uploaded_history is not None:
                try:
                    history = _uploaded_history(uploaded_history, snapshot)
                except (TypeError, UnicodeError, ValueError) as exc:
                    st.error(str(exc))

        if history is not None:
            history_key = history_fingerprint(history)
            missing = missing_history_assets(snapshot, history)
            if missing:
                st.error(
                    f"{tr('missing_history')} Missing: "
                    + ", ".join(position.symbol for position in missing)
                )
            elif model_inputs_valid and st.button(tr("run_simulation"), type="primary"):
                try:
                    with st.spinner("Calibrating covariance and simulating paths…"):
                        calibration, result = simulate_snapshot(
                            snapshot,
                            history,
                            config=SimulationConfig(
                                n_paths=paths,
                                horizon_steps=horizon,
                                step_size_days=1.0,
                                seed=seed,
                                drift_mode="zero",
                            ),
                            reconciliation_tolerance_bps=(reconciliation_tolerance_bps),
                        )
                    st.session_state["risk_result"] = (calibration, result)
                    st.session_state["risk_result_key"] = (
                        snapshot.user_address,
                        snapshot.block.number,
                        history_key,
                        paths,
                        horizon,
                        seed,
                    )
                except (ValueError, FloatingPointError) as exc:
                    st.session_state.pop("risk_result", None)
                    st.session_state.pop("risk_result_key", None)
                    st.error(f"Simulation unavailable: {exc}")
            expected_key = (
                snapshot.user_address,
                snapshot.block.number,
                history_key,
                paths,
                horizon,
                seed,
            )
            if model_inputs_valid and st.session_state.get("risk_result_key") == expected_key:
                calibration, result = st.session_state["risk_result"]
                _render_simulation_result(result, calibration, language)
        else:
            st.info(
                "Upload a reproducible daily cache. The app refuses to invent volatility for "
                "unsupported assets. See scripts/capture_history.py."
            )

    with liquidator_tab:
        if model_inputs_valid:
            _render_liquidator(snapshot, scenario_snapshot, shocks, language)
        elif not snapshot.emode_data_complete:
            st.error(
                "Liquidator economics are disabled because required eMode metadata is incomplete."
            )
        else:
            st.error("Liquidator economics are disabled until the asset-level snapshot reconciles.")

    with methodology_tab:
        st.markdown(
            """
### Model contract

- Canonical spot prices come from the Aave Oracle at one block; Chainlink/SVR/adapter sources
  are provenance, not hard-coded substitutes.
- Health factor is recomputed from every enabled collateral and every debt asset.
- The default 30-day model uses daily log returns, sample per-asset volatility, a
  Ledoit-Wolf-shrunk correlation matrix, zero arithmetic drift, 20,000 seeded paths,
  Wilson intervals for simulated future-event probabilities, deterministic intervals for the
  fixed initial state, and bootstrap VaR/ES intervals.
- Liquidation math validates Aave V3.7/revision 11 unscaled amount calculations and fails
  closed when a pair violates eligibility or dust constraints; scaled aToken/index settlement
  rounding is outside this research model.
- P&L is a read-only estimate. Quotes expire, MEV and reverts are not eliminated, and no
  transaction is built, signed, or submitted.

Read `reports/research_report.md` for equations, validation design, and limitations.
"""
        )
        st.info(tr("research_only"))
        st.markdown(
            "[Aave V3.7 liquidation source](https://github.com/aave-dao/aave-v3-origin/blob/main/src/contracts/protocol/libraries/logic/LiquidationLogic.sol) · "
            "[Aave oracle docs](https://aave.com/docs/aave-v3/smart-contracts/oracles) · "
            "[Chainlink historical data](https://docs.chain.link/data-feeds/historical-data)"
        )


if __name__ == "__main__":
    main()
