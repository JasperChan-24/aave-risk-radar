# Aave Whale Risk Monitor

Asset-level liquidation-risk research for Aave V3 on Ethereum.

The dashboard reconstructs a borrower's position reserve by reserve, calibrates a correlated
multi-asset price model from fixed historical observations, estimates first-passage liquidation
risk and tail loss, and evaluates liquidator economics after protocol fees, flash-loan premium,
gas, and read-only firm-pricing inputs. It is designed as a reproducible research portfolio
project, not a liquidation bot.

> 中文概述：项目按用户真实的逐资产抵押与负债重算健康因子，并输出 1/7/30 天清算概率、
> 首次触线时间、VaR/ES、置信区间，以及逐项列示主要成本的清算人净收益。

## Why the prototype was rebuilt

| Prototype | Current design |
| --- | --- |
| Scaled aggregate health factor by one unnamed shock | Revalues every collateral and debt asset with its own oracle price and effective threshold |
| Used fixed `u=1.1`, `d=0.95`, `p=1/3` | Uses an explicit 30-day horizon, daily step, historical volatility, and cross-asset covariance |
| Drew 100 unseeded paths | Runs deterministic, chunked Monte Carlo and computes liquidation probability, TTL, VaR/ES, and confidence intervals |
| Reported `50% debt × 5%` as “risk-free profit” | Applies revision-aware Aave amount rules and itemizes bonus, protocol fee, flash fee, gas, quoted proceeds, and net P&L |
| Mixed RPC, math, and UI in one file | Separates `data`, `models`, `simulation`, and `app`, with offline fixtures and unit tests |

## Architecture

```mermaid
flowchart LR
    RPC["Ethereum RPC"] --> DATA["data: block-consistent snapshot"]
    ORACLE["Aave Oracle / price sources"] --> DATA
    HISTORY["Cached historical observations"] --> CAL["simulation: calibration"]
    DATA --> PORT["models: portfolio + HF"]
    DATA --> LIQ["models: liquidation math"]
    CAL --> MC["simulation: correlated paths"]
    PORT --> MC
    LIQ --> PNL["liquidator net P&L"]
    QUOTE["Optional read-only firm pricing"] --> PNL
    PORT --> UI["Streamlit app"]
    MC --> UI
    PNL --> UI
```

The only fixed protocol bootstrap is Ethereum's Aave V3 `PoolAddressesProvider`. The active
Pool, Price Oracle, and Data Provider are discovered at runtime, and every call in a snapshot
uses the same block tag. The app uses the canonical Aave oracle price because that is what the
protocol consumes; the underlying source address is retained for Chainlink/SVR/adapter
provenance.

## Risk outputs

- Per-asset supplied balance, debt balance, collateral flag, price, LTV, liquidation threshold,
  liquidation bonus, protocol fee, and position value.
- Reconstructed health factor with a same-block comparison to `getUserAccountData`.
- User-specified, independent asset shocks for deterministic scenario analysis. This is not a
  calibrated historical stress-testing or sensitivity-analysis suite.
- 1-, 7-, and 30-day liquidation probability with a 95% Wilson interval for simulated future
  events; an initially liquidatable snapshot instead has the deterministic interval `[1,1]`.
  First passage remains strict below `HF=1`, with a disclosed eight-ULP guard against binary
  summation noise exactly at the boundary.
- First observed time-to-liquidation distribution with non-hitting paths treated as censored.
- 95% and 99% VaR/Expected Shortfall on fixed-quantity, terminal mark-to-market borrower
  net-equity loss, with bootstrap confidence intervals. Paths are not transitioned into a
  post-liquidation portfolio after first passage.
- Pair-specific liquidator P&L with close factor, collateral capacity, dust constraints,
  protocol fee, current Pool flash premium, gas, optional execution/MEV cost, and fixed or
  optional read-only firm pricing. The route guard checks actual and virtual reserve liquidity,
  aToken supply for the flash principal, same-asset callback ordering, self-liquidation, quote
  block lag, and whether the manual gas budget covers 0x's reported network fee.

The deployed Pool revision is recorded in every snapshot. The implementation targets Aave V3
revision 11/v3.7 and fails closed for unsupported semantics. Its liquidation adapter validates
the economically relevant **unscaled** debt-repayment and collateral-seizure amount arithmetic
used by the P&L model. It is not a byte-for-byte `liquidationCall` emulator: the snapshot and
model do not reproduce scaled aToken balance settlement, liquidity-index/Ray rounding, or the
fee-cap end state. Those mechanics can matter at raw-unit boundaries and remain targets for
fork-based validation. Aave's
[view contracts](https://aave.com/docs/aave-v3/smart-contracts/view-contracts),
[oracle interface](https://aave.com/docs/aave-v3/smart-contracts/oracles), and
[v3.7 liquidation changes](https://github.com/aave-dao/aave-v3-origin/blob/main/docs/3.7/Aave-v3.7-changelog.md)
are the primary protocol references.

## Run locally

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
streamlit run app.py
```

To install the exact package versions used by the local macOS/Python 3.12 validation run:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

`ALCHEMY_RPC_URL` (or `ETHEREUM_RPC_URL`) enables live portfolio reads. Archive access is
needed only when rebuilding historical Aave-oracle caches. The offline demo and checked-in
fixtures keep the UI and tests usable without network access.

`ZEROX_API_KEY` is optional. When present, the P&L layer can request a timestamped 0x
AllowanceHolder **firm pricing** quote in read-only mode. The adapter validates expected and
minimum buy amounts plus supported fee fields and
discards transaction calldata; it does not claim that the quoted trade is executable at a later
block. Without it, the optional 0x mode fails explicitly and the separately selected,
clearly-labelled fixed/assumption mode remains available; the app never silently turns an oracle
valuation into sale proceeds.

## Reproduce and verify

```bash
python -m pip check
ruff check .
ruff format --check .
mypy src/aave_risk_monitor
pytest --cov=aave_risk_monitor --cov-report=term-missing
```

Unit tests are offline and deterministic. RPC integration tests are marked `integration` and
operate against a fixed block when explicitly enabled. Snapshot JSON stores chain ID, block
number/hash/timestamp, contract addresses and Pool revision, eMode, raw balances, reserve
configuration, oracle price/source, actual and virtual reserve liquidity, aToken total supply,
and Aave's aggregate account values.

Snapshot JSON intentionally contains only block-pinned protocol and provenance data. Simulation
configuration and random seed are attached to each in-memory simulation result and displayed by
the app; they are not embedded in the snapshot artifact. See the
[validation evidence](reports/validation_report.md) for the boundary between implemented checks
and the research roadmap.

The repository includes a real read-only [Ethereum snapshot](snapshots/README.md) at block
`25,573,974`. To capture a new fixed block:

```bash
python scripts/capture_snapshot.py --user 0xYOUR_BORROWER --block latest --output snapshots
```

The checked-in fixed-block snapshot has a matching 366-observation Aave-oracle cache at
[`data/history/ethereum-v3-whale-block-25573974.csv`](data/history/ethereum-v3-whale-block-25573974.csv).
To rebuild a trailing 365-return window from an archive RPC, ask the capture utility for 366
daily observations:

```bash
python scripts/capture_history.py \
  --snapshot snapshots/1-ed0c6079229e2d407672a117c22b62064f4a4312-block-25573974.json \
  --days 366 \
  --output data/history/ethereum-v3-whale-block-25573974.csv
```

For a manually curated sample, provide one explicit archive block per consecutive UTC date.
This keeps the sampling decision reviewable:

```bash
python scripts/capture_history.py \
  --snapshot snapshots/1-USER-block-N.json \
  --blocks-file data/daily-blocks.txt \
  --output data/prices.csv
```

The CSV records the date, block number/hash, block timestamp, oracle address, and an
address-keyed price column for every active collateral or debt asset. Historical capture calls
the oracle address recorded in the snapshot at each past block; it does not rediscover the
provider's active oracle independently at every observation. The cache therefore assumes that
the provider's oracle address did not change across the requested window. A window spanning an
oracle migration must be segmented by oracle era and validated separately.

Reproduce the checked-in 20,000-path numerical result (including input hashes, calibration,
probability intervals, TTL censoring, and terminal mark-to-market VaR/ES) with:

```bash
python scripts/run_fixed_analysis.py \
  --snapshot snapshots/1-ed0c6079229e2d407672a117c22b62064f4a4312-block-25573974.json \
  --history data/history/ethereum-v3-whale-block-25573974.csv \
  --output reports/fixed_snapshot_analysis.json
```

The resulting [machine-readable analysis artifact](reports/fixed_snapshot_analysis.json) has a
zero 30-day liquidation-frequency point estimate for this high-HF account, but its Wilson 95%
upper bound is non-zero; this is not evidence that liquidation risk is literally zero.

## Project layout

```text
src/aave_risk_monitor/
├── data/          # Aave/Oracle reads, historical observations, snapshots
├── models/        # Portfolio health, liquidation rules, liquidator P&L
├── simulation/    # Calibration, correlated paths, risk metrics
└── app/           # Streamlit presentation and orchestration
tests/             # Golden cases, statistical tests, offline fixtures
reports/           # Research methodology, validation evidence, limitations
scripts/           # Reproducible snapshot/history capture utilities
snapshots/         # Immutable public-chain snapshot artifacts
```

## Research boundaries

This project does not sign or submit transactions. Daily paths can miss intraday barrier
crossings; Gaussian dynamics can understate jumps, stablecoin depegs, and liquidity spirals;
quotes can expire; and MEV competition or state changes can make a theoretically profitable
liquidation fail. Confidence intervals describe simulation sampling uncertainty, not complete
model uncertainty. VaR/ES stop at the horizon mark-to-market state and do not model the
balance-sheet transition, execution loss, or feedback effects after liquidation. Formal
historical stress studies, global sensitivity analysis, out-of-sample VaR backtests, and
time-step convergence studies remain future work. The revision-11 P&L adapter models unscaled
amounts, not scaled aToken/liquidity-index/Ray settlement or a forked transaction end state.
Read the full
[research report](reports/research_report.md) before interpreting results.

## License

MIT. See [LICENSE](LICENSE).
