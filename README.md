# 📊 Heston Model Dashboard Tool

A user-friendly interactive dashboard for the implementation, calibration, and visualization of the **Heston stochastic volatility model** used in option pricing.

## 🧠 Overview

This tool enables users—quant analysts, researchers, and financial engineers—to:

- Simulate and visualize stochastic volatility paths under the Heston model
- Calibrate model parameters to real market option prices
- Price European call and put options using Heston's semi-analytical solution
- Analyze implied volatility surfaces
- Interact with model inputs and outputs via a dynamic Python-based web dashboard

---

## ⚙️ Features

- 📈 **Volatility Path Simulation**: Simulate variance and price paths using the Euler–Maruyama method.
- 🎯 **Parameter Calibration**: Calibrate parameters (θ, κ, σ, ρ, v₀) using historical or market data.
- 📉 **Option Pricing Engine**: Compute European option prices under Heston dynamics.
- 🧩 **Implied Vol Surface**: Visualize model-based implied volatilities across strikes and maturities.
- 📊 **Dashboard Interface**: Built using `Dash`/`Plotly` or `Streamlit` for interactive visualization.

---

## 🧪 Heston Model Equation

The Heston model assumes the following dynamics under the risk-neutral measure:

\[
\begin{aligned}
dS_t &= \mu S_t dt + \sqrt{v_t} S_t dW_t^S \\
dv_t &= \kappa(\theta - v_t) dt + \sigma \sqrt{v_t} dW_t^v \\
\end{aligned}
\]

Where:
- \( S_t \) = asset price
- \( v_t \) = variance
- \( \kappa \) = mean reversion speed
- \( \theta \) = long-term variance
- \( \sigma \) = volatility of variance
- \( \rho \) = correlation between asset and variance Brownian motions

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- Recommended Libraries:
  - `numpy`
  - `pandas`
  - `scipy`
  - `plotly`
  - `dash` or `streamlit` *(choose one for your GUI framework)*
  - `yfinance` *(optional for live data)*

### Installation

```bash
git clone https://github.com/yourusername/heston-dashboard.git
cd heston-dashboard
pip install -r requirements.txt
