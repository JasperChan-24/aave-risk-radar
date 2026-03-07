# 🌊 Aave V3 Whale Risk Radar & Quantitative Liquidation Simulator

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Web3.py](https://img.shields.io/badge/Web3-Ethereum-8A2BE2)
![Quantitative Finance](https://img.shields.io/badge/FinTech-Quantitative_Risk-success)

## 👨‍💻 Author
**Chen Yanyu**

## 📌 Project Overview
An advanced on-chain risk monitoring and stress-testing engine designed for the **Aave V3** protocol. 
Unlike traditional static health factor trackers, this project introduces quantitative finance methodologies—including **Binomial Tree Option Pricing** and **Geometric Brownian Motion (GBM) Monte Carlo Simulations**—to dynamically evaluate the survival probability of whale debt positions under extreme market volatility.

## ✨ Core Features
- **🌐 Real-Time Web3 Integration**: Connects to the Ethereum Mainnet via Alchemy RPC to fetch live Aave V3 lending data.
- **🔗 Decentralized Oracle Pricing**: Utilizes **Chainlink** smart contracts to retrieve highly accurate, tamper-proof ETH/USD exchange rates.
- **🌲 Down-and-Out Barrier Option Pricing**: Models whale debt positions as barrier options using backward induction to calculate mathematical expected values and dynamic survival probabilities.
- **🌌 GBM Monte Carlo Simulation**: Employs Stochastic Differential Equations (SDE) to generate 100 independent parallel price paths, offering visual insights into liquidation thresholds.
- **🦈 Flashloan Arbitrage Radar**: Automatically calculates potential gross profit for liquidators when a position's Health Factor drops below 1.0.
- **🌍 Bilingual Interface**: Seamlessly switches between English and Chinese UI for international accessibility.

## 📐 Methodology (Financial Mathematics)
To evaluate the liquidation risk, the application applies the **Binomial Option Pricing Model**:
- **Upward Jump Multiplier ($u$)**: 1.1 (+10%)
- **Downward Jump Multiplier ($d$)**: 0.95 (-5%)
- **Risk-Neutral Probability ($p$)**: 1/3
- **Barrier Condition**: Health Factor (HF) $\le$ 1.0 triggers a Knock-out (Payoff = 0). Survival yields a Payoff of 5.

Furthermore, the Monte Carlo engine generates continuous price paths assuming collateral returns follow a normal distribution $N(0, 1)$, mapped through the **Geometric Brownian Motion**:
$$S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma Z \sqrt{\Delta t} \right)$$

## 🚀 Run Locally
```bash
# Clone the repository
git clone [https://github.com/JasperChan-24/aave-risk-radar.git](https://github.com/JasperChan-24/aave-risk-radar.git)

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file and add your Alchemy RPC URL:
# ALCHEMY_RPC_URL=[https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY](https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY)

# Launch the app
streamlit run app.py