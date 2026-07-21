# Validation Evidence and Open Work

This document separates checks implemented in the repository from validation that would be
needed for a stronger empirical claim. It is an evidence index, not an external audit.

## Reproducible artifacts

- The checked-in Ethereum Aave V3 snapshot
  [`snapshots/1-ed0c6079229e2d407672a117c22b62064f4a4312-block-25573974.json`](../snapshots/1-ed0c6079229e2d407672a117c22b62064f4a4312-block-25573974.json)
  pins chain ID, block `25,573,974`, block hash and timestamp, discovered Aave contracts and Pool
  revision, reserve configuration, raw user balances, Aave-oracle prices/source addresses, eMode,
  flash-loan metadata, actual/virtual underlying liquidity, aToken total supply, and aggregate
  account values.
- The snapshot's reconstructed health factor differs from same-block
  `Pool.getUserAccountData` by approximately `3.65e-12` basis points. This is a reconciliation
  check for that account/block, not a claim that every possible reserve configuration is covered.
- The matching
  [`data/history/ethereum-v3-whale-block-25573974.csv`](../data/history/ethereum-v3-whale-block-25573974.csv)
  contains 366 consecutive daily observations ending at the snapshot block. Each row records the
  UTC date, block number/hash/timestamp, oracle address, and address-keyed Aave-oracle prices for
  the active WETH, WBTC, and USDT assets.
- The machine-readable
  [`reports/fixed_snapshot_analysis.json`](fixed_snapshot_analysis.json) records SHA-256 hashes
  for both inputs, the model source tree, and the artifact-generator script, plus runtime package
  versions, the complete default simulation configuration, sample per-asset volatility with
  Ledoit-Wolf-shrunk correlation calibration,
  probability/TTL results, and terminal mark-to-market VaR/ES with intervals. With 20,000 paths,
  30 daily steps, and seed `20260720`, this high-HF account had zero observed first passages;
  the Wilson 95% interval was `[0, 0.0001920361]`. All 20,000 TTL observations were therefore
  right-censored. This finite-sample result does not establish zero true liquidation risk.
- Snapshot JSON does not contain simulation configuration or a seed. Those values are attached to
  the simulation result and shown by the app, so reproducing an exported numerical result also
  requires recording those displayed settings.
- The machine-readable
  [`reports/empirical_validation.json`](empirical_validation.json) binds the same snapshot and
  366-observation history by SHA-256. It compares 1/0.5/0.25-day grids, 2,000/8,000/32,000 paths,
  and 90/180/365-day calibration windows; runs 185 strictly rolling one-day 95% VaR forecasts;
  and records permanent USDT price shocks plus covariance-matched Student-t innovations. The
  observed block-`25,573,974` account is the high-HF control (`HF=4.2733`). The convergence and
  stress portfolio at `HF=1.05` is explicitly synthetic and is not presented as an observed user.
- The fixed
  [`reports/liquidation_replay.json`](liquidation_replay.json) and
  [`reports/liquidation_replay.md`](liquidation_replay.md) bind an observed revision-11 borrower
  at block `25,572,181`, with `HF=0.999565096476407496`, to the successful liquidation in
  Ethereum transaction
  [`0xd138…ab62f`](https://etherscan.io/tx/0xd138a0455f087ad399820fb42f4fd35ca8f8986c223609afabf1cba1ffdab62f)
  one block later. The model matches the event's `15,001,159` raw USDT repayment and
  `8,495,915,601,721,874` raw WETH delivered to the liquidator exactly. The original signed
  transaction also succeeds on a block-minus-one Anvil mainnet fork with an identical
  `LiquidationCall` payload; the fork receipt's gas usage is evidence, not a gas-model target.

## Implemented automated checks

| Claim under test | Evidence |
| --- | --- |
| Asset-level valuation and same-block reconciliation | [`tests/test_portfolio.py`](../tests/test_portfolio.py) covers mixed thresholds, disabled collateral, debt-free accounts, independent HF/collateral-total/debt-total reconciliation, fail-closed legacy eMode persistence, fixed snapshot round-trip, historical CSV alignment, mocked same-block contract reads, and reorg rejection. |
| Historical-cache integrity | [`tests/test_history.py`](../tests/test_history.py) covers price-matrix invariants, CSV filtering/errors, archive-RPC sequential and batch reads, rate-limit retry, and block/hash provenance. |
| Calibrated correlated simulation | [`tests/test_simulation.py`](../tests/test_simulation.py) covers sample-volatility and shrunk-correlation calibration from log returns, PSD covariance, invalid/irregular history, covariance reproduction, zero drift, deterministic seeds, chunk invariance, and bounded stored paths. |
| First-passage and sampling metrics | [`tests/test_simulation.py`](../tests/test_simulation.py) covers strict `HF<1`, the eight-ULP numerical boundary guard, deterministic `[1,1]` intervals for an initially liquidatable snapshot, right censoring, Wilson intervals, empirical VaR/ES, and reproducible bootstrap intervals on controlled data. |
| Revision-11 unscaled liquidation amount arithmetic | [`tests/test_liquidation.py`](../tests/test_liquidation.py) covers close factor, the 0.95 and 2,000/1,000 base-currency boundaries, collateral caps, eMode, protocol fee, dust rules, and modeled raw-unit rounding. It does not constitute a full scaled-balance state-transition proof. |
| Liquidator P&L and quote boundaries | [`tests/test_pnl.py`](../tests/test_pnl.py) covers cost decomposition, optional execution/MEV cost, break-even, flash-premium rounding, fixed quotes, read-only 0x firm-pricing response parsing, fee handling, unavailable liquidity, and failure propagation. |
| App orchestration and fail-closed guards | [`tests/test_app_services.py`](../tests/test_app_services.py) covers asset-keyed shocks, seeded simulation wiring, history integrity, incomplete eMode metadata, integer HF eligibility, unsupported Pool revisions, oracle-base assumptions, pause/grace/flash-loan/actual-and-virtual-liquidity guards, same-asset callback ordering, conservative gas rounding, and P&L inputs. [`tests/test_app_smoke.py`](../tests/test_app_smoke.py) renders the default demo offline. |
| Fixed-block RPC consistency | [`tests/test_mainnet_integration.py`](../tests/test_mainnet_integration.py) is opt-in and compares live archive-RPC reads at the pinned block with the checked-in snapshot. The reader also rejects a block hash that changes during capture. The integration test is skipped unless explicitly enabled. |
| Empirical convergence, backtest, and stress evidence | [`tests/test_empirical_validation.py`](../tests/test_empirical_validation.py) covers deterministic grid/path/window studies, look-ahead rejection, known Kupiec and Christoffersen behavior, covariance-matched Student-t sampling, role-sensitive stablecoin shocks, and artifact/input hash integrity. |
| Observed liquidation replay and mainnet fork | [`tests/test_liquidation_replay.py`](../tests/test_liquidation_replay.py) decodes the fixed raw log, reconciles the near-HF and high-HF controls, and requires zero raw-unit event/model deltas. [`tests/test_mainnet_fork_liquidation.py`](../tests/test_mainnet_fork_liquidation.py) fails closed without its runtime and, when explicitly enabled, replays the original signed transaction on block-minus-one state and compares the fork event with both history and model. |
| Content-addressed release gate | [`tests/test_evidence_gate.py`](../tests/test_evidence_gate.py) verifies that artifact, source, test, screenshot, account/block, event/model, and fork bindings fail closed when missing or modified. The same verifier runs in [Evidence Gate](../.github/workflows/evidence-gate.yml) for pull requests, `main`, and published releases. |

Run the offline suite and static checks with:

```bash
python -m pip check
ruff check .
ruff format --check .
mypy src/aave_risk_monitor
pytest --cov=aave_risk_monitor --cov-report=term-missing
python scripts/verify_evidence.py
```

Local release validation on 2026-07-21 completed with `148 passed, 2 skipped`, total branch
coverage `81.16%`, and no Ruff, formatting, mypy, dependency-consistency, deterministic-artifact,
or Evidence Gate failures. One skip is the opt-in fixed-block RPC check and one is the opt-in
mainnet-fork check; each was also enabled and passed separately. CI repeats the offline suite,
static checks, dependency check, and content-addressed verifier independently on `main`.

Run the archive-RPC check only when a suitable Ethereum RPC URL is configured:

```bash
RUN_RPC_INTEGRATION=1 pytest -m integration tests/test_mainnet_integration.py
```

Regenerate the fixed numerical evidence with:

```bash
python scripts/run_fixed_analysis.py \
  --snapshot snapshots/1-ed0c6079229e2d407672a117c22b62064f4a4312-block-25573974.json \
  --history data/history/ethereum-v3-whale-block-25573974.csv \
  --output reports/fixed_snapshot_analysis.json
```

Regenerate the empirical evidence and execute the opt-in fork replay with:

```bash
python scripts/run_empirical_validation.py
RUN_MAINNET_FORK=1 \
ANVIL_COMMAND="npx --yes @foundry-rs/anvil@1.7.1" \
python scripts/run_fork_validation.py
```

For the exact package versions from the local macOS/Python 3.12 validation run, install
[`requirements-lock.txt`](../requirements-lock.txt) and then install the local project with
`--no-deps`.

## Interpretation boundaries

- Liquidation probability and TTL use discrete daily first passage. They can miss an intraday
  crossing and do not establish continuous-time barrier accuracy. The implementation applies an
  eight-ULP guard below `HF=1` solely to avoid a floating-point summation false positive at an
  algebraically exact boundary. The checked-in 1/0.5/0.25-day experiment demonstrates numerical
  stability at its recorded tolerances, not convergence to an analytical continuous-time limit.
- VaR and ES are based on fixed supplied/borrowed quantities and terminal mark-to-market net
  equity. A liquidated path is not transitioned through debt repayment, collateral seizure,
  execution costs, or subsequent account evolution.
- Wilson and bootstrap intervals quantify finite-simulation sampling uncertainty; a fixed
  initially liquidatable state instead has a deterministic `[1,1]` interval. None of these
  intervals includes parameter uncertainty, data-window choice, model misspecification, jumps,
  oracle latency, or liquidity feedback.
- The rolling VaR backtest fixes the current account's token quantities and revalues them through
  historical prices; it does not reconstruct the borrower's historical balances. Of 185 one-day
  forecasts it records 11 breaches versus 9.25 expected. Kupiec unconditional coverage passes
  (`p=0.566`), but Christoffersen independence fails (`p=0.00135`), so the aggregate statistical
  acceptance result is intentionally false and indicates clustered exceptions.
- The USDT shocks are role-sensitive permanent price jumps; because USDT is debt in this account,
  a downward price break improves HF while a `+10%` break is adverse and makes the synthetic
  portfolio initially liquidatable. The Student-t(df=5) experiment matches innovation covariance
  but is not a fitted return-distribution claim; in this seeded run its 99% ES is about 0.99%
  above the Gaussian baseline while its VaR is slightly lower.
- The optional 0x adapter obtains a read-only AllowanceHolder firm pricing quote, parses pricing
  fields, and discards transaction calldata. No allowance or transaction is submitted, and the
  result is not an execution guarantee. The adapter does not claim a separately validated
  price-impact measure.
- The historical cache queries the oracle address pinned in the snapshot at prior blocks. It
  assumes `PoolAddressesProvider` did not switch oracle addresses within the window; a window
  spanning an oracle migration must be segmented and validated separately.
- The revision-11 liquidator model validates economically relevant unscaled repayment and
  seizure amount math. It does not reproduce scaled aToken settlement, liquidity-index/Ray
  rounding, or the fee-cap end state. One observed small-position/full-close WETH/USDT event and
  its fork replay match exactly, but this does not establish equivalence for other branches.
- The fixed snapshot validates one account, one market, one Pool revision, and one block. It does
  not substitute for cross-market or post-upgrade validation.

## Validation roadmap

Remaining work:

1. Analytical first-passage benchmarks or grids finer than 0.25 day to establish a stronger
   continuous-time reference.
2. Bootstrap-resample convergence and explicit parameter-uncertainty intervals.
3. Rolling backtests with historically reconstructed account balances, alternative VaR models,
   and remediation of the observed breach clustering.
4. Historical crash/oracle-latency episode replay and fitted regime or jump models; the current
   stablecoin and Student-t exercises are disclosed sensitivity scenarios.
5. Systematic local/global sensitivity analysis.
6. Forked `liquidationCall` comparisons across more reserve pairs, accounts, markets, and Pool
   revisions, including scaled-balance, liquidity-index/Ray-rounding, and fee-cap end-state
   comparisons.
