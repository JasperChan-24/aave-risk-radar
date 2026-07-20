# Asset-Level Liquidation Risk on Aave V3

## Abstract

This project estimates liquidation risk for an Aave V3 account from the account's actual
collateral and debt composition. It replaces a scalar health-factor shock and uncalibrated
single-factor paths with a block-consistent portfolio snapshot, asset-level oracle prices,
historically calibrated cross-asset dynamics, first-passage liquidation metrics, and a
cost-explicit liquidator profit-and-loss model. The software is a research and monitoring
tool. It does not submit transactions and its outputs are not trading advice.

The first release deliberately targets the Ethereum Aave V3 Core market. Every snapshot
records the block, contract addresses, Pool implementation revision, oracle base unit, and
asset-level protocol state so that its inputs remain interpretable after protocol upgrades.
Simulation configuration and seed belong to the simulation result and UI state; they are not
stored inside the snapshot JSON.

## 1. Protocol state and portfolio valuation

The Ethereum `PoolAddressesProvider` is the only bootstrap address. The application discovers
the active Pool, Price Oracle, and Protocol Data Provider at runtime, then reads all values at
one block tag. Per-reserve balances come from `getUserReserveData`; collateral parameters and
the liquidation protocol fee come from the reserve configuration; the Pool supplies the
user's eMode category and aggregate account data used for reconciliation. These interfaces are
documented in the [Aave view-contract reference](https://aave.com/docs/aave-v3/smart-contracts/view-contracts)
and [PoolAddressesProvider reference](https://aave.com/docs/aave-v3/smart-contracts/pool-addresses-provider).

For collateral asset \(i\) and debt asset \(j\), the reconstructed health factor is

\[
HF_t = \frac{\sum_i q^C_i P_{i,t} LT_i^{\mathrm{effective}}}
              {\sum_j q^D_j P_{j,t}}.
\]

Only balances enabled as collateral enter the numerator. The effective liquidation threshold
uses the user's eMode category when the reserve belongs to that category; otherwise it uses
the reserve threshold. A debt-free account has an infinite health factor by convention.
Integer token units and the oracle base-currency unit are preserved until display conversion.

The canonical valuation input is `AaveOracle.getAssetPrice`, because that is the price consumed
by Aave's own health-factor and liquidation logic. `getSourceOfAsset` is retained as provenance;
the source may be a standard Chainlink feed, an SVR feed, or a wrapper-specific adapter, so a
hard-coded ETH/USD aggregator cannot reproduce an arbitrary portfolio. See the
[Aave oracle reference](https://aave.com/docs/aave-v3/smart-contracts/oracles) and the
[Chainlink AggregatorV3 interface](https://docs.chain.link/data-feeds/api-reference).

## 2. Historical calibration and simulation

### 2.1 Default research configuration

- Horizon: 30 calendar days, with 1-day observation steps.
- Reported sub-horizons: 1, 7, and 30 days.
- Calibration window: 366 daily UTC price observations, yielding 365 aligned daily log returns.
- Price process: correlated geometric process over every active collateral and debt asset.
- Drift: zero by default; short-window sample means are too unstable to treat as forecasts.
- Volatility and dependence: sample per-asset volatility from aligned log returns, combined
  with a Ledoit-Wolf-shrunk correlation matrix and projected to a positive semidefinite
  covariance matrix if numerical noise remains. Standardizing before shrinkage prevents a
  low-volatility asset from inheriting another asset's scale.
- Monte Carlo: 20,000 paths by default, deterministic seed, generated in chunks.
- Tail levels: 95% and 99%; interval confidence level: 95%.

Historical Aave-oracle observations are sampled at common blocks and the capture utility caches
the corresponding block hashes. The spot snapshot also retains AggregatorV3-compatible round
metadata when the Aave source exposes that interface. Interface compatibility alone does not
prove that a source is a direct Chainlink feed, so the snapshot avoids that stronger label.
For a historical window, the capture utility queries the oracle address pinned by the snapshot
at each past block. It does not re-resolve `PoolAddressesProvider.getPriceOracle()` at every
observation, so the resulting panel assumes that the provider's oracle address remained
unchanged across the window. A period spanning an oracle migration must be split into
oracle-consistent segments and validated before calibration.

Chainlink rounds are event-driven by heartbeat and
deviation thresholds and can span multiple aggregator phases, so raw rounds must be resampled
before covariance estimation. The
[Chainlink historical-data guide](https://docs.chain.link/data-feeds/historical-data) describes
these phase-aware round identifiers.

For step covariance \(\Sigma\), Cholesky-like factor \(L\), and independent standard-normal
draw \(z_t\), simulated log prices follow

\[
\log P_{t+1}=\log P_t - \tfrac{1}{2}\operatorname{diag}(\Sigma)+Lz_t.
\]

The current model holds supplied and borrowed token quantities fixed over the 30-day horizon;
interest accrual is therefore outside this release's model scope. At every step the engine
revalues every asset and recomputes health factor. The first observation with \(HF<1\) is
the discrete first-passage time. To prevent binary floating-point portfolio summation from
turning an algebraically exact \(HF=1\) into a false breach, the implementation uses an explicit
eight-ULP guard immediately below the barrier; economically material breaches remain strict.
This daily model should not be interpreted as an intraday liquidation clock; time-step
sensitivity remains on the future validation roadmap.

### 2.2 Risk outputs

- **Liquidation probability:** fraction of paths whose health factor is first observed strictly
  below one within the selected horizon, with a Wilson binomial confidence interval for simulated
  future events. If the fixed initial snapshot is already liquidatable, the interval is
  deterministically `[1,1]` instead.
- **Time to liquidation:** first-passage distribution conditional on liquidation; non-hitting
  paths remain right-censored and are not assigned an artificial time.
- **Value at Risk:** quantile of terminal mark-to-market loss in borrower net equity,
  \(L_T=(C_0-D_0)-(C_T-D_T)\).
- **Expected Shortfall:** mean of the requested worst tail mass of the same loss variable.
- **Uncertainty:** bootstrap confidence intervals for VaR and ES. These quantify Monte Carlo
  sampling uncertainty, not full structural model uncertainty.

These tail metrics keep supplied and borrowed token quantities fixed and mark them to simulated
prices at the horizon. A path that crosses the liquidation boundary is still carried to the
horizon for this calculation: the engine does not seize collateral, repay debt, accrue execution
costs, or transition the borrower into a post-liquidation state. Liquidation probability/TTL and
terminal VaR/ES should therefore be interpreted as separate views of the same pre-execution path
set, not as a single end-to-end liquidation loss model.

## 3. Liquidation mechanics and liquidator net P&L

The implementation is version-aware and validates the economically relevant **unscaled**
integer amount arithmetic used to estimate revision-11 liquidator economics. For the current
Aave V3 liquidation logic, an account must have \(HF<1\). A selected debt reserve is generally
fully coverable, except when the selected collateral and debt legs are each at least 2,000 units
of the oracle base currency and \(HF>0.95\); in that case the call is capped at the lesser of the
selected debt balance and 50% of total account debt. Collateral capacity, the 1,000-unit leftover
rule, reserve or eMode liquidation bonus, protocol fee, decimals, and the modeled floor/ceil
rounding can further reduce the estimated executable amount.

This is not an exact `liquidationCall` state-transition emulator. In particular, the current
snapshot and model do not reproduce scaled aToken balance settlement, liquidity-index/Ray
rounding, or the fee-cap end state. Consequently, raw-unit results can differ from an on-chain
execution even when the unscaled economic inputs agree. The authoritative implementation is
[Aave V3 LiquidationLogic](https://github.com/aave-dao/aave-v3-origin/blob/main/src/contracts/protocol/libraries/logic/LiquidationLogic.sol).
The [v3.7 changelog](https://github.com/aave-dao/aave-v3-origin/blob/main/docs/3.7/Aave-v3.7-changelog.md)
documents the explicit floor/ceil rounding changes and Pool revision 11.

For a concrete collateral/debt pair, the model reports

\[
\text{Net P\&L} = Q_{swap} - D_{repaid} - F_{flash} - C_{gas} - C_{other},
\]

where \(Q_{swap}\) is a quoted amount for selling the collateral received after the protocol's
share of the bonus. Flash-loan premium is read from the Pool for each live snapshot;
gas uses a stated gas-unit estimate and EIP-1559 fee assumption. Fixed quotes are used in
offline tests, with explicit haircut and minimum-output assumptions. The optional 0x adapter
requests an AllowanceHolder **firm pricing** quote in read-only mode and parses expected/minimum
buy amounts plus supported fee fields. It does not expose a separately validated price-impact
measure. The adapter intentionally discards transaction calldata, sets no allowance, and submits
no transaction. The quote is an input to the P&L model, not proof of later execution. No output
is called “risk-free”: competition, MEV, state changes, reverts, and quote expiry remain execution
risks. The application rejects self-liquidation, requires actual and virtual reserve liquidity,
checks same-asset flash-principal-plus-seizure ordering, and requires the user-entered full gas
budget to cover any native network fee reported by 0x without charging that fee twice.

## 4. Validation status and roadmap

### 4.1 Implemented checks

The checked-in tests and fixed artifacts currently cover:

1. Recomputing health factor from per-reserve data and reconciling it with
   `Pool.getUserAccountData` at the same block.
2. Mixed collateral, debt-price shocks, eMode, disabled collateral, no-debt accounts,
   non-18-decimal tokens, stale/invalid prices, and controlled boundary-rounding cases.
3. Calibration from sample per-asset volatility and Ledoit-Wolf-shrunk correlation on aligned
   daily log returns, positive-semidefinite covariance, simulated
   covariance/correlation reproduction, deterministic seeding, chunk-size invariance, and
   zero-volatility/perfect-correlation cases.
4. Strict `HF<1` first passage, right-censored non-hits, Wilson intervals, empirical VaR/ES,
   and deterministic bootstrap intervals on controlled samples.
5. Revision-11 unscaled liquidation amount arithmetic for the 0.95 health-factor boundary, the
   2,000/1,000 base-currency rules, collateral capacity, eMode bonus, protocol fee, and modeled
   raw-unit rounding.
6. P&L cost decomposition, flash-premium rounding, break-even output, and validation/failure
   paths for fixed quotes and read-only 0x firm-pricing inputs.
7. Offline Streamlit smoke coverage and an opt-in, fixed-block Ethereum RPC integration check.

The concrete fixtures, test-file mapping, and commands are recorded in
[validation_report.md](validation_report.md).

For the checked-in block `25,573,974`, the reproducible default run uses 365 daily returns,
20,000 paths, 30 one-day steps, and seed `20260720`. No path crossed below one for this account
whose reconstructed starting health factor is approximately 4.2733; the 30-day point estimate
is therefore zero, while the Wilson 95% upper bound is approximately 0.0192%. Full calibration,
input hashes, censoring counts, and VaR/ES intervals are stored in
[`fixed_snapshot_analysis.json`](fixed_snapshot_analysis.json). A zero observed count is not a
claim of zero true risk.

### 4.2 Future validation work

The following are research objectives, not claims about the current release:

1. Compare single-asset first-passage estimates with analytical and finer-grid benchmarks.
2. Publish path-count, calibration-window, time-step, and bootstrap-resample convergence studies.
3. Backtest VaR exceptions on rolling out-of-sample windows.
4. Compare Gaussian Monte Carlo against historical and heavy-tailed resampling for crash,
   stablecoin-depeg, and oracle-latency episodes.
5. Run systematic local/global sensitivity analysis across volatility, correlation, drift,
   liquidity, gas, and execution-haircut assumptions.
6. Compare liquidation outputs with forked `liquidationCall` executions across more markets and
   future Pool revisions, including scaled aToken settlement, liquidity-index/Ray rounding, and
   fee-cap end-state differences.

## 5. Limitations

The model assumes the snapshot's collateral settings and token quantities remain constant over
the forecast horizon, so interest accrual is omitted. Gaussian innovations underrepresent jumps,
liquidity spirals, oracle latency, and stablecoin depegs. Formal historical stress tests and
sensitivity studies have not yet been implemented; they are future validation work, not
companions to the current base estimate. Daily monitoring can miss intraday barrier crossings.
The optional 0x input is a short-lived, read-only firm pricing quote whose calldata is discarded;
it is neither a submitted trade nor an execution guarantee. VaR/ES are fixed-quantity terminal
mark-to-market metrics and do not model post-trigger balance changes. The historical panel
assumes the snapshot oracle address remained active throughout its window, and the liquidator
adapter does not simulate scaled aToken/liquidity-index/Ray settlement or a complete transaction
end state. Finally, protocol upgrades can alter liquidation semantics, so unknown Pool revisions
fail closed until their arithmetic is validated.
