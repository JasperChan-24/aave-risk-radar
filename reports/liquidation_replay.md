# Aave V3 revision-11 liquidation replay

## Result

The fixed historical replay passes exactly at raw-token precision. Ethereum
transaction
[`0xd138…ab62f`](https://etherscan.io/tx/0xd138a0455f087ad399820fb42f4fd35ca8f8986c223609afabf1cba1ffdab62f)
succeeded in block `25,572,182`. The account snapshot is pinned to block
`25,572,181` (Pool revision `11`) and has observed health factor
`0.999565096476407496`.

| Quantity | LiquidationCall event | Model | Raw delta |
|---|---:|---:|---:|
| USDT debt liquidated (6 decimals) | `15,001,159` | `15,001,159` | `0` |
| WETH delivered to liquidator (18 decimals) | `8,495,915,601,721,874` | `8,495,915,601,721,874` | `0` |

The model separately calculates gross WETH seizure
`8,536,565,915,605,711` and protocol-fee WETH `40,650,313,883,837`.
Their difference is the amount emitted by `LiquidationCall` for the liquidator.

The original signed transaction was also submitted to an Anvil `1.7.1`
mainnet fork at block `25,572,181`. It succeeded in fork block `25,572,182`
with the same transaction hash, and all four event topics plus the 128-byte
event data matched mainnet exactly. Fork gas was `465,276`, versus historical
gas `464,316` (delta `+960`): the fork did not replay the target block's first
23 transactions, so warm/cold state differs while liquidation semantics do not.

## Observed account controls

- Near-liquidation case: `0xeB7525…eD3Ea`, block `25,572,181`, HF
  `0.999565096476407496`.
- High-health control: `0xEd0C60…4a4312`, block `25,573,974`, HF
  `4.273322048891862594`.

Both asset-level snapshots reconcile to `Pool.getUserAccountData`. These are
observed accounts, not synthetic debt rescalings.

## Reproduce

```bash
python scripts/capture_liquidation_evidence.py
pytest -q tests/test_liquidation_replay.py tests/test_mainnet_fork_liquidation.py
```

To execute the original transaction on block-minus-one mainnet state:

```bash
RUN_MAINNET_FORK=1 \
ANVIL_COMMAND="anvil" \
python scripts/run_fork_validation.py
```

The runner requires an archive-capable RPC in `ETHEREUM_ARCHIVE_RPC_URL`,
`ETHEREUM_RPC_URL`, or `ALCHEMY_RPC_URL`. It writes
`reports/mainnet_fork_liquidation.json` only after Anvil starts, the replayed
transaction succeeds, one matching `LiquidationCall` is found, and all event ↔
model comparisons pass. Missing Anvil/RPC, a revert, an ambiguous event, or any
raw-unit mismatch returns a non-zero exit and does not create a success artifact.

## Scope limits

The offline arithmetic replay is conditioned on the debt amount emitted by the
event. A block-minus-one snapshot cannot reproduce preceding transactions within
the liquidation block. This single WETH/USDT case exercises the revision-11
small-position/full-close branch; it does not validate every reserve, eMode,
dust, or collateral-capped branch. The checked-in fork artifact records the
executed result; re-running requires the same archive state and
Anvil-compatible EVM behavior.
