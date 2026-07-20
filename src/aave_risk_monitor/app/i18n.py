"""Small, explicit bilingual copy table for the Streamlit presentation layer."""

from __future__ import annotations

from typing import Final

COPY: Final[dict[str, tuple[str, str]]] = {
    "title": ("Aave 巨鲸资产级风险监控", "Aave Whale Asset-Level Risk Monitor"),
    "subtitle": (
        "逐资产健康因子、清算概率、尾部风险与清算人净收益",
        "Per-asset health factor, liquidation probability, tail risk, and liquidator net P&L",
    ),
    "language": ("语言", "Language"),
    "data_mode": ("数据模式", "Data mode"),
    "demo": ("固定演示快照", "Fixed demo snapshot"),
    "live": ("以太坊实时读取", "Live Ethereum read"),
    "address": ("钱包地址", "Wallet address"),
    "rpc_missing": (
        "未配置 RPC；请在 .env 中设置 ALCHEMY_RPC_URL。",
        "RPC is not configured; set ALCHEMY_RPC_URL in .env.",
    ),
    "portfolio": ("资产组合", "Portfolio"),
    "stress": ("逐资产压力测试", "Asset stress test"),
    "simulation": ("概率模拟", "Risk simulation"),
    "liquidator": ("清算人净收益", "Liquidator net P&L"),
    "methodology": ("方法与限制", "Methodology & limits"),
    "collateral": ("抵押品价值", "Collateral value"),
    "debt": ("负债价值", "Debt value"),
    "health_factor": ("资产级健康因子", "Asset-level health factor"),
    "onchain_hf": ("链上聚合 HF", "Onchain aggregate HF"),
    "no_debt": ("无负债", "No debt"),
    "reconciled": ("同区块对账通过", "Same-block reconciliation passed"),
    "not_reconciled": (
        "资产级结果与链上聚合值未在容差内对齐",
        "Asset-level result did not reconcile within tolerance",
    ),
    "shock_help": (
        "每个价格冲击同时作用于该资产的抵押和负债估值。负数代表下跌。",
        "Each shock revalues both collateral and debt for that asset. Negative values are price drops.",
    ),
    "hypothetical": (
        "以下为假设压力情景，不代表账户当前可被链上清算。",
        "This is a hypothetical stress scenario, not a claim that the account is currently liquidatable.",
    ),
    "horizon": ("预测窗口（天）", "Horizon (days)"),
    "paths": ("模拟路径数", "Simulation paths"),
    "seed": ("随机种子", "Random seed"),
    "run_simulation": ("运行可复现模拟", "Run reproducible simulation"),
    "missing_history": (
        "固定历史数据缺少当前仓位资产，已拒绝用猜测波动率补齐。",
        "Cached history does not cover every active asset; guessed volatility is intentionally not used.",
    ),
    "liq_probability": ("所选期限内清算概率", "Liquidation probability within horizon"),
    "ttl": ("条件中位首次触线时间", "Conditional median time to liquidation"),
    "current_safe": (
        "账户当前 HF 不低于 1，不能执行清算；这里只显示成本模型说明。",
        "Current HF is not below 1, so no liquidation is executable; only the cost-model explanation is shown.",
    ),
    "research_only": (
        "研究与监控用途：不签名、不发送交易，也不将任何结果称为“无风险套利”。",
        "Research and monitoring only: no transaction signing/submission and no result is described as risk-free arbitrage.",
    ),
}


def translate(key: str, language: str) -> str:
    """Return Chinese for ``中文`` and English otherwise."""

    chinese, english = COPY[key]
    return chinese if language == "中文" else english


__all__ = ["COPY", "translate"]
