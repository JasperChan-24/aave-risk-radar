from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from web3 import Web3

from aave_risk_monitor.data import AaveV3DataSource, load_snapshot

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "snapshots" / "1-ed0c6079229e2d407672a117c22b62064f4a4312-block-25573974.json"


def test_fixed_mainnet_block_matches_checked_in_snapshot() -> None:
    if os.getenv("RUN_RPC_INTEGRATION") != "1":
        pytest.skip("set RUN_RPC_INTEGRATION=1 to enable fixed-block RPC verification")
    load_dotenv(ROOT / ".env", override=False)
    rpc_url = os.getenv("ALCHEMY_RPC_URL") or os.getenv("ETHEREUM_RPC_URL")
    if not rpc_url:
        pytest.skip("Ethereum RPC URL is not configured")

    expected = load_snapshot(SNAPSHOT)
    web3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))
    actual = AaveV3DataSource(web3).fetch_portfolio(
        expected.user_address,
        expected.block.number,
    )

    assert actual.block == expected.block
    assert actual.contracts == expected.contracts
    assert actual.account == expected.account
    assert actual.positions == expected.positions
    assert actual.emode_data_complete == expected.emode_data_complete
    assert actual.flashloan_premium_total_bps == expected.flashloan_premium_total_bps
