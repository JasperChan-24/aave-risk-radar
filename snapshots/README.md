# Fixed snapshots

Two observed Ethereum Aave V3 accounts are pinned for complementary validation:

- `1-ed0c6079229e2d407672a117c22b62064f4a4312-block-25573974.json` is the
  high-health control (`HF=4.273322048891862594`) used by the fixed simulation and rolling VaR
  evidence.
- `1-eb75251694d4d1a71af99b1f96c819dc5e9ed3ea-block-25572181.json` is the
  observed near-liquidation state (`HF=0.999565096476407496`) one block before the successful
  revision-11 liquidation in Ethereum transaction
  [`0xd138…ab62f`](https://etherscan.io/tx/0xd138a0455f087ad399820fb42f4fd35ca8f8986c223609afabf1cba1ffdab62f).

The second snapshot is bound to the raw event, exact model replay, and executed mainnet-fork
evidence in [`../reports/liquidation_replay.md`](../reports/liquidation_replay.md).

`1-ed0c6079229e2d407672a117c22b62064f4a4312-block-25573974.json` is a read-only
Ethereum mainnet Aave V3 snapshot captured at block `25,573,974`
(`2026-07-20T12:37:59Z`). It contains public on-chain state only; no RPC URL,
API key, private key, or transaction payload is stored.

The asset-level health factor reconstructed from the JSON differs from
`Pool.getUserAccountData` by approximately `3.65e-12` basis points. WETH exposes
AggregatorV3-compatible round metadata; compatibility alone is not treated as
proof of a direct Chainlink feed. The WBTC and USDT Aave price sources are
labelled as oracle adapters because their source contracts do not expose
`latestRoundData` directly; their canonical Aave-oracle prices are still stored.
The snapshot also pins eMode completeness and each reserve's paused state,
liquidation grace period, flash-loan flag, aToken address and total supply,
actual underlying balance, and virtual underlying balance at the same block.
Capture re-reads the resolved block hash after all calls and discards the result
if that hash changes during the read.

This file is a reproducibility artifact, not current account state. Re-capture a
new block with:

```bash
python scripts/capture_snapshot.py \
  --user 0xed0c6079229e2d407672a117c22b62064f4a4312 \
  --output snapshots
```

Its matching 366-day Aave-oracle cache is
[`../data/history/ethereum-v3-whale-block-25573974.csv`](../data/history/ethereum-v3-whale-block-25573974.csv),
and the default seeded simulation output is
[`../reports/fixed_snapshot_analysis.json`](../reports/fixed_snapshot_analysis.json).
The history builder queries this snapshot's oracle address at every past block and assumes the
provider did not change oracle addresses during the window. History spanning an oracle migration
must be split into oracle-consistent segments before calibration.
