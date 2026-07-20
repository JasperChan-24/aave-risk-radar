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

Run the offline suite and static checks with:

```bash
python -m pip check
ruff check .
ruff format --check .
mypy src/aave_risk_monitor
pytest --cov=aave_risk_monitor --cov-report=term-missing
```

Local release validation on 2026-07-20 completed with `127 passed, 1 skipped`, total branch
coverage `84.46%`, and no Ruff, formatting, mypy, or dependency-consistency failures. The
separately enabled fixed-block RPC integration check completed with `1 passed`.

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

For the exact package versions from the local macOS/Python 3.12 validation run, install
[`requirements-lock.txt`](../requirements-lock.txt) and then install the local project with
`--no-deps`.

## Interpretation boundaries

- Liquidation probability and TTL use discrete daily first passage. They can miss an intraday
  crossing and do not establish continuous-time barrier accuracy. The implementation applies an
  eight-ULP guard below `HF=1` solely to avoid a floating-point summation false positive at an
  algebraically exact boundary.
- VaR and ES are based on fixed supplied/borrowed quantities and terminal mark-to-market net
  equity. A liquidated path is not transitioned through debt repayment, collateral seizure,
  execution costs, or subsequent account evolution.
- Wilson and bootstrap intervals quantify finite-simulation sampling uncertainty; a fixed
  initially liquidatable state instead has a deterministic `[1,1]` interval. None of these
  intervals includes parameter uncertainty, data-window choice, model misspecification, jumps,
  oracle latency, or liquidity feedback.
- The optional 0x adapter obtains a read-only AllowanceHolder firm pricing quote, parses pricing
  fields, and discards transaction calldata. No allowance or transaction is submitted, and the
  result is not an execution guarantee. The adapter does not claim a separately validated
  price-impact measure.
- The historical cache queries the oracle address pinned in the snapshot at prior blocks. It
  assumes `PoolAddressesProvider` did not switch oracle addresses within the window; a window
  spanning an oracle migration must be segmented and validated separately.
- The revision-11 liquidator model validates economically relevant unscaled repayment and
  seizure amount math. It does not reproduce scaled aToken settlement, liquidity-index/Ray
  rounding, or the fee-cap end state, so forked execution may differ at raw-unit boundaries.
- The fixed snapshot validates one account, one market, one Pool revision, and one block. It does
  not substitute for cross-market or post-upgrade validation.

## Validation roadmap

Not yet implemented:

1. Analytical or fine-grid first-passage benchmarks and explicit time-step convergence.
2. Path-count, calibration-window, and bootstrap-resample convergence reports.
3. Rolling out-of-sample VaR exception backtests.
4. Historical/heavy-tailed resampling for crash, stablecoin-depeg, and oracle-latency regimes.
5. Systematic local/global sensitivity analysis.
6. Forked `liquidationCall` comparisons across more reserve pairs, accounts, markets, and Pool
   revisions, including scaled-balance, liquidity-index/Ray-rounding, and fee-cap end-state
   comparisons.
