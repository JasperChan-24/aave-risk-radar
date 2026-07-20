"""Minimal read-only ABIs used by the data adapters.

Keeping these ABIs local makes the snapshot reader independent of generated
contract bindings while still documenting every on-chain field consumed.
"""

from __future__ import annotations

from typing import Any


def _view(
    name: str,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "stateMutability": "view",
        "inputs": inputs,
        "outputs": outputs,
    }


POOL_ADDRESSES_PROVIDER_ABI = [
    _view("getMarketId", [], [{"name": "", "type": "string"}]),
    _view("getPool", [], [{"name": "", "type": "address"}]),
    _view("getPriceOracle", [], [{"name": "", "type": "address"}]),
    _view("getPoolDataProvider", [], [{"name": "", "type": "address"}]),
]

POOL_ABI = [
    _view(
        "getUserAccountData",
        [{"name": "user", "type": "address"}],
        [
            {"name": "totalCollateralBase", "type": "uint256"},
            {"name": "totalDebtBase", "type": "uint256"},
            {"name": "availableBorrowsBase", "type": "uint256"},
            {"name": "currentLiquidationThreshold", "type": "uint256"},
            {"name": "ltv", "type": "uint256"},
            {"name": "healthFactor", "type": "uint256"},
        ],
    ),
    _view(
        "getUserEMode",
        [{"name": "user", "type": "address"}],
        [{"name": "", "type": "uint256"}],
    ),
    _view("getReservesCount", [], [{"name": "", "type": "uint256"}]),
    _view(
        "getReserveAddressById",
        [{"name": "id", "type": "uint16"}],
        [{"name": "", "type": "address"}],
    ),
    _view(
        "getEModeCategoryCollateralConfig",
        [{"name": "id", "type": "uint8"}],
        [
            {
                "name": "",
                "type": "tuple",
                "components": [
                    {"name": "ltv", "type": "uint16"},
                    {"name": "liquidationThreshold", "type": "uint16"},
                    {"name": "liquidationBonus", "type": "uint16"},
                ],
            }
        ],
    ),
    _view(
        "getEModeCategoryCollateralBitmap",
        [{"name": "id", "type": "uint8"}],
        [{"name": "", "type": "uint128"}],
    ),
    _view(
        "getEModeCategoryLtvzeroBitmap",
        [{"name": "id", "type": "uint8"}],
        [{"name": "", "type": "uint128"}],
    ),
    _view(
        "getIsEModeCategoryIsolated",
        [{"name": "id", "type": "uint8"}],
        [{"name": "", "type": "bool"}],
    ),
    _view(
        "getEModeCategoryData",
        [{"name": "id", "type": "uint8"}],
        [
            {
                "name": "",
                "type": "tuple",
                "components": [
                    {"name": "ltv", "type": "uint16"},
                    {"name": "liquidationThreshold", "type": "uint16"},
                    {"name": "liquidationBonus", "type": "uint16"},
                    {"name": "priceSource", "type": "address"},
                    {"name": "label", "type": "string"},
                ],
            }
        ],
    ),
    _view("POOL_REVISION", [], [{"name": "", "type": "uint256"}]),
    _view("FLASHLOAN_PREMIUM_TOTAL", [], [{"name": "", "type": "uint128"}]),
    _view(
        "getLiquidationGracePeriod",
        [{"name": "asset", "type": "address"}],
        [{"name": "", "type": "uint40"}],
    ),
    _view(
        "getVirtualUnderlyingBalance",
        [{"name": "asset", "type": "address"}],
        [{"name": "", "type": "uint128"}],
    ),
]

POOL_DATA_PROVIDER_ABI = [
    _view(
        "getAllReservesTokens",
        [],
        [
            {
                "name": "",
                "type": "tuple[]",
                "components": [
                    {"name": "symbol", "type": "string"},
                    {"name": "tokenAddress", "type": "address"},
                ],
            }
        ],
    ),
    _view(
        "getUserReserveData",
        [
            {"name": "asset", "type": "address"},
            {"name": "user", "type": "address"},
        ],
        [
            {"name": "currentATokenBalance", "type": "uint256"},
            {"name": "currentStableDebt", "type": "uint256"},
            {"name": "currentVariableDebt", "type": "uint256"},
            {"name": "principalStableDebt", "type": "uint256"},
            {"name": "scaledVariableDebt", "type": "uint256"},
            {"name": "stableBorrowRate", "type": "uint256"},
            {"name": "liquidityRate", "type": "uint256"},
            {"name": "stableRateLastUpdated", "type": "uint40"},
            {"name": "usageAsCollateralEnabled", "type": "bool"},
        ],
    ),
    _view(
        "getReserveConfigurationData",
        [{"name": "asset", "type": "address"}],
        [
            {"name": "decimals", "type": "uint256"},
            {"name": "ltv", "type": "uint256"},
            {"name": "liquidationThreshold", "type": "uint256"},
            {"name": "liquidationBonus", "type": "uint256"},
            {"name": "reserveFactor", "type": "uint256"},
            {"name": "usageAsCollateralEnabled", "type": "bool"},
            {"name": "borrowingEnabled", "type": "bool"},
            {"name": "stableBorrowRateEnabled", "type": "bool"},
            {"name": "isActive", "type": "bool"},
            {"name": "isFrozen", "type": "bool"},
        ],
    ),
    _view(
        "getReserveEModeCategory",
        [{"name": "asset", "type": "address"}],
        [{"name": "", "type": "uint256"}],
    ),
    _view(
        "getLiquidationProtocolFee",
        [{"name": "asset", "type": "address"}],
        [{"name": "", "type": "uint256"}],
    ),
    _view(
        "getPaused",
        [{"name": "asset", "type": "address"}],
        [{"name": "", "type": "bool"}],
    ),
    _view(
        "getFlashLoanEnabled",
        [{"name": "asset", "type": "address"}],
        [{"name": "", "type": "bool"}],
    ),
    _view(
        "getReserveTokensAddresses",
        [{"name": "asset", "type": "address"}],
        [
            {"name": "aTokenAddress", "type": "address"},
            {"name": "stableDebtTokenAddress", "type": "address"},
            {"name": "variableDebtTokenAddress", "type": "address"},
        ],
    ),
]

ERC20_ABI = [
    _view(
        "balanceOf",
        [{"name": "account", "type": "address"}],
        [{"name": "", "type": "uint256"}],
    ),
    _view("totalSupply", [], [{"name": "", "type": "uint256"}]),
]

AAVE_ORACLE_ABI = [
    _view("BASE_CURRENCY", [], [{"name": "", "type": "address"}]),
    _view("BASE_CURRENCY_UNIT", [], [{"name": "", "type": "uint256"}]),
    _view(
        "getAssetPrice",
        [{"name": "asset", "type": "address"}],
        [{"name": "", "type": "uint256"}],
    ),
    _view(
        "getAssetsPrices",
        [{"name": "assets", "type": "address[]"}],
        [{"name": "", "type": "uint256[]"}],
    ),
    _view(
        "getSourceOfAsset",
        [{"name": "asset", "type": "address"}],
        [{"name": "", "type": "address"}],
    ),
]

CHAINLINK_AGGREGATOR_ABI = [
    _view(
        "latestRoundData",
        [],
        [
            {"name": "roundId", "type": "uint80"},
            {"name": "answer", "type": "int256"},
            {"name": "startedAt", "type": "uint256"},
            {"name": "updatedAt", "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"},
        ],
    ),
]

EIP1967_IMPLEMENTATION_SLOT = int(
    "360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc", 16
)


__all__ = [
    "AAVE_ORACLE_ABI",
    "CHAINLINK_AGGREGATOR_ABI",
    "EIP1967_IMPLEMENTATION_SLOT",
    "ERC20_ABI",
    "POOL_ABI",
    "POOL_ADDRESSES_PROVIDER_ABI",
    "POOL_DATA_PROVIDER_ABI",
]
