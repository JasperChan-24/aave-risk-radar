import os
import streamlit as st
from web3 import Web3
from dotenv import load_dotenv
import numpy as np
import pandas as pd

# --- 1. 页面基本设置 (必须在第一行) ---
st.set_page_config(page_title="DeFi Risk Radar", page_icon="🌊", layout="centered")

# --- 2. 🌐 双语引擎设置 ---
lang = st.sidebar.radio("🌐 Language / 语言", ["English", "中文"])

def t(zh, en):
    return zh if lang == "中文" else en

# --- 3. 页面头部 ---
st.title(t("🌊 Aave 巨鲸风险监控与压力测试仪", "🌊 Aave Whale Risk Monitor & Stress Tester"))
st.markdown(t("**作者：Yanyu.Chen** | 实时监控链上杠杆与清算边界", "**Author: Jasper Chan** | Real-time On-chain Leverage & Liquidation Monitor"))

# --- 4. 侧边栏：用户输入 ---
st.sidebar.header(t("⚙️ 参数设置", "⚙️ Parameters Setup"))
target_address = st.sidebar.text_input(
    t("输入 Ethereum 钱包地址", "Enter Ethereum Wallet Address"), 
    "0xed0c6079229e2d407672a117c22b62064f4a4312"
)
price_drop = st.sidebar.slider(
    t("📉 模拟抵押品价格下跌 (%)", "📉 Simulate Collateral Price Drop (%)"), 
    min_value=0, max_value=80, value=0, step=5
)

# 加载环境变量
load_dotenv() 

# --- 5. 核心逻辑：节点连接与缓存 ---
@st.cache_resource 
def connect_web3():
    rpc_url = os.getenv("ALCHEMY_RPC_URL") 
    return Web3(Web3.HTTPProvider(rpc_url))

w3 = connect_web3()

@st.cache_data(ttl=60)
def fetch_onchain_data(target_addr):
    AAVE_V3_POOL = w3.to_checksum_address("0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2")
    ABI = [{"inputs":[{"internalType":"address","name":"user","type":"address"}],"name":"getUserAccountData","outputs":[{"internalType":"uint256","name":"totalCollateralBase","type":"uint256"},{"internalType":"uint256","name":"totalDebtBase","type":"uint256"},{"internalType":"uint256","name":"availableBorrowsBase","type":"uint256"},{"internalType":"uint256","name":"currentLiquidationThreshold","type":"uint256"},{"internalType":"uint256","name":"ltv","type":"uint256"},{"internalType":"uint256","name":"healthFactor","type":"uint256"}],"stateMutability":"view","type":"function"}]
    contract = w3.eth.contract(address=AAVE_V3_POOL, abi=ABI)
    user_address = w3.to_checksum_address(target_addr)
    data = contract.functions.getUserAccountData(user_address).call()
    
    chainlink_eth_usd = w3.to_checksum_address("0x5f4eC3Df9cbd43714FE2740f5E3616155c5b8419")
    chainlink_abi = [{"inputs":[],"name":"latestRoundData","outputs":[{"internalType":"uint80","name":"roundId","type":"uint80"},{"internalType":"int256","name":"answer","type":"int256"},{"internalType":"uint256","name":"startedAt","type":"uint256"},{"internalType":"uint256","name":"updatedAt","type":"uint256"},{"internalType":"uint80","name":"answeredInRound","type":"uint80"}],"stateMutability":"view","type":"function"}]
    price_contract = w3.eth.contract(address=chainlink_eth_usd, abi=chainlink_abi)
    eth_price_usd = price_contract.functions.latestRoundData().call()[1] / 10**8
    
    return data, eth_price_usd

# --- 6. 主程序 ---
if not w3.is_connected():
    st.error(t("网络连接失败，请检查 RPC。", "Network connection failed, please check RPC URL."))
else:
    st.success(t(f"🟢 链上节点已连接 (最新区块: {w3.eth.block_number})", f"🟢 On-chain Node Connected (Latest Block: {w3.eth.block_number})"))
    
    try:
        with st.spinner(t('正在从以太坊主网抓取数据 (带 60 秒 API 缓存)...', 'Fetching data from Ethereum Mainnet (with 60s cache)...')):
            data, eth_price_usd = fetch_onchain_data(target_address)
            
        raw_usd_collateral = data[0] / 10**8
        usd_debt = data[1] / 10**8
        raw_hf = data[5] / 10**18
        
        drop_multiplier = 1 - (price_drop / 100.0)
        usd_collateral = raw_usd_collateral * drop_multiplier
        hf = raw_hf * drop_multiplier
        
        # --- 数据展示区 ---
        st.subheader(t("📊 账户实时资产快照 (基于 Aave V3)", "📊 Real-time Account Snapshot (Aave V3)"))
        
        col1, col2, col3 = st.columns(3)
        col1.metric(t("总抵押品 (USD)", "Total Collateral (USD)"), f"${usd_collateral:,.2f}")
        col2.metric(t("总负债 (USD)", "Total Debt (USD)"), f"${usd_debt:,.2f}")
        
        if hf > 10000:
            col3.metric(t("健康因子 (HF)", "Health Factor (HF)"), t("无借款 (MAX)", "No Debt (MAX)"))
            st.info(t("💡 该账户目前没有任何负债，非常安全。请在左侧换一个有借款的地址！", "💡 This account has no debt and is completely safe. Please enter an address with active borrows!"))
        else:
            col3.metric(t("当前健康因子", "Current Health Factor"), f"{hf:.4f}")
            
            # --- 金融数学风控区 ---
            st.divider()
            st.subheader(t("🌲 量化风控引擎：基于二叉树的动态清算推演", "🌲 Quant Risk Engine: Binomial Tree Liquidation Simulation"))
            
            # 原理折叠面板
            with st.expander(t("📖 展开阅读：一分钟看懂模型背后的金数原理 (Methodology)", "📖 Expand: Financial Math Methodology Behind the Model")):
                if lang == "中文":
                    st.markdown("""
                    本项目跳出传统的“静态健康因子”，引入了金融工程中经典的**二叉树模型（Binomial Tree）**。
                    我们将巨鲸的借贷仓位视为一个**向下敲出障碍期权 (Down-and-Out Barrier Option)**，推演未来市场的“平行宇宙”：
                    * **模拟未来波动**：假设未来每步抵押品资产可能上涨 **10%** ($u=1.1$) 或下跌 **-5%** ($d=0.95$)。
                    * **风险中性概率**：基于无套利定价理论，向上跳跃的风险中性概率为 **1/3**。
                    * **生存即价值**：多步预测中，若 HF 保持在 **1.0** 以上则仓位存活（Payoff = 5）；若跌破 1.0 即触发无情清算（Payoff = 0）。
                    *结论：通过动态规划反向推导，计算出在未来 N 步波动中的**动态存活概率**。*
                    """)
                else:
                    st.markdown("""
                    This project goes beyond the traditional "static health factor" by introducing the classic **Binomial Tree Model** from quantitative finance.
                    We price the whale's debt position as a **Down-and-Out Barrier Option** to simulate "parallel universes" of future market conditions:
                    * **Volatility Simulation**: Assuming the collateral value can jump up by **10%** ($u=1.1$) or drop by **-5%** ($d=0.95$) in each step.
                    * **Risk-Neutral Probability**: Based on the no-arbitrage pricing theory, the probability of an upward jump is **1/3**.
                    * **Survival Value**: If the Health Factor (HF) stays above **1.0**, the position survives (Payoff = 5). If it breaches 1.0 at any node, forced liquidation is triggered (Payoff = 0).
                    *Conclusion: We use backward induction to calculate the expected **Dynamic Survival Probability** over N future steps.*
                    """)

            steps = st.slider(
                t("⏱️ 设置未来预测的时间步数 (N-Steps)", "⏱️ Set Future Prediction Steps (N-Steps)"), 
                min_value=1, max_value=10, value=3
            )

            def calculate_binomial_expectation(current_hf, n_steps):
                u, d, p = 1.1, 0.95, 1/3
                survival_payoff, liquidation_payoff, barrier = 5, 0, 1.0
                hf_tree = [current_hf * (u ** (n_steps - i)) * (d ** i) for i in range(n_steps + 1)]
                payoffs = [survival_payoff if h > barrier else liquidation_payoff for h in hf_tree]
                for step in range(n_steps - 1, -1, -1):
                    for i in range(step + 1):
                        current_node_hf = current_hf * (u ** (step - i)) * (d ** i)
                        if current_node_hf <= barrier:
                            payoffs[i] = liquidation_payoff
                        else:
                            payoffs[i] = p * payoffs[i] + (1 - p) * payoffs[i + 1]
                return payoffs[0]

            expected_payoff = calculate_binomial_expectation(hf, steps)
            survival_score = (expected_payoff / 5) * 100 
            
            st.markdown(t(f"**📍 当前健康因子起点：** `{hf:.4f}`", f"**📍 Starting HF Node:** `{hf:.4f}`"))
            
            res_col1, res_col2 = st.columns(2)
            res_col1.metric(t("⚖️ 期权期望价值 (满分5)", "⚖️ Option Expected Value (Max 5)"), f"{expected_payoff:.4f}")
            res_col2.metric(t("🛡️ 动态存活概率", "🛡️ Dynamic Survival Probability"), f"{survival_score:.1f}%")
            
            if expected_payoff == 0:
                st.error(t(f"🚨 **极度危险 (必定爆仓)**：在 {steps} 步二叉树推演下，所有未来的价格路径均触及 1.0 清算线，系统判定存活概率为 **0%**！", f"🚨 **CRITICAL DANGER (Guaranteed Liquidation)**: Under a {steps}-step simulation, all price paths breach the 1.0 threshold. Survival probability is **0%**!"))
                st.progress(0)
            elif expected_payoff < 2.5:
                st.warning(t(f"⚠️ **高危预警 (走钢丝)**：巨鲸仓位随时可能在后续波动中被击穿，当前存活概率仅为 **{survival_score:.1f}%**。", f"⚠️ **HIGH RISK (Walking on a Tightrope)**: The position is highly vulnerable to upcoming volatility. Survival probability is only **{survival_score:.1f}%**."))
                st.progress(int(survival_score))
            else:
                st.success(t(f"✅ **资产安全 (护城河深厚)**：距离清算边界有充足的缓冲垫，推演存活概率高达 **{survival_score:.1f}%**。", f"✅ **SAFE (Deep Moat)**: Sufficient buffer against liquidation boundaries. Survival probability is high at **{survival_score:.1f}%**."))
                st.progress(int(survival_score))

            # --- 套利雷达 ---
            if hf < 1.0:
                st.divider()
                st.subheader(t("🦈 清算人视角：无风险套利雷达", "🦈 Liquidator POV: Risk-free Arbitrage Radar"))
                max_liquidatable_debt = usd_debt * 0.5
                liquidation_bonus = max_liquidatable_debt * 0.05
                
                st.success(t(
                    f"**发现清算猎物！**\n若通过智能合约发起闪电贷 (Flashloan) 介入清算，当前最高可代还 **${max_liquidatable_debt:,.2f}** 负债。\n扣除成本前，预估可获得的清算套利毛利润为：**${liquidation_bonus:,.2f}**",
                    f"**Liquidation Target Detected!**\nBy executing a Flashloan smart contract, you can liquidate up to **${max_liquidatable_debt:,.2f}** of the debt.\nEstimated gross arbitrage profit (before gas) is: **${liquidation_bonus:,.2f}**"
                ))

            # --- 蒙特卡洛模拟 ---
            st.divider()
            st.subheader(t("🌌 蒙特卡洛模拟 (GBM)：平行宇宙的连续价格路径", "🌌 Monte Carlo Simulation (GBM): Parallel Universe Price Paths"))
            st.markdown(t(f"基于 **几何布朗运动 (Geometric Brownian Motion)** 与正态分布随机漫步，为您展开 **100 条**未来 {steps} 步的独立演变路径：", f"Based on **Geometric Brownian Motion (GBM)** and normal distribution random walk, displaying **100** independent simulated paths for the next {steps} steps:"))
            
            n_paths, mu, sigma = 100, 0.0, 0.07
            Z = np.random.standard_normal((n_paths, steps))
            step_multipliers = np.exp((mu - 0.5 * sigma**2) + sigma * Z)
            path_multipliers = np.cumprod(step_multipliers, axis=1)
            starting_hf_array = np.full((n_paths, 1), hf)
            simulated_paths = np.hstack((starting_hf_array, hf * path_multipliers))
            
            df_paths = pd.DataFrame(simulated_paths.T)
            st.line_chart(df_paths)
            
            st.caption(t(
                "💡 **量化分析说明**：上图使用随机偏微分方程 (SDE) 生成了独立的 GBM 路径。在真实清算逻辑中，只要任意一条线在任意时间点触碰并跌穿 **1.0 (清算红线)**，该条路径即宣告“死亡”并触发强制清算。", 
                "💡 **Quant Analysis Note**: The chart above uses Stochastic Differential Equations (SDE) to generate independent GBM paths. In real-world mechanisms, if any path breaches the **1.0 Liquidation Threshold (Red Line)** at any time, that universe path 'dies' and triggers forced liquidation."
            ))
    
    except Exception as e:
        st.error(t(f"查询失败，请检查地址格式。详细错误: {e}", f"Query failed. Please check the address format. Error: {e}"))