# Modern Portfolio Theory & Multi-Asset Analytics

> *"Diversification is the only free lunch in finance."* — Harry Markowitz, 1952

---

## Demo

![Dashboard Demo](demo/demo.gif)

> **To record your own demo:**
> 1. Launch the app: `streamlit run app.py`
> 2. Use a screen-capture tool (e.g. [ScreenToGif](https://www.screentogif.com/) on Windows, or [LICEcap](https://www.cockos.com/licecap/)) to record the browser window.
> 3. Save the output as `demo/demo.gif` (create the `demo/` folder if it does not exist).
> 4. The image will automatically render on GitHub once the file is in place.

---

## Table of Contents

1. [Demo](#demo)
2. [Introduction](#introduction)
3. [Mathematical Methodology](#mathematical-methodology)
4. [The Optimization Problem](#the-optimization-problem)
5. [Key Insights & Results](#key-insights--results)
6. [Dashboard — Tab-by-Tab Insights](#dashboard--tab-by-tab-insights)
7. [Engineering Implementation](#engineering-implementation)
8. [Project Structure](#project-structure)
9. [Quick Start](#quick-start)
10. [Future Extensions](#future-extensions)

---

## Introduction

Modern portfolio theory (MPT), introduced by Harry Markowitz in his 1952 paper *"Portfolio Selection"*, marks a fundamental shift in investment thinking: from analysing individual securities in isolation to analysing the *system* of assets as a whole. The insight is that the risk of a portfolio is not a simple weighted average of individual asset risks — it is governed by the **covariance structure** of returns.

This project implements the full MPT pipeline across three interconnected modules:

| Module | File | Focus |
|---|---|---|
| Data Layer | `data_loader.py` | Data ingestion, cleaning, Parquet storage |
| Technical Analysis | `indicators.py` | Bollinger Bands, Drawdown |
| Optimizer | `optimizer.py` | Mean-Variance Optimization, Efficient Frontier |
| Dashboard | `app.py` | Interactive Streamlit analytics suite |

### Why Diversification is the "Free Lunch"

A rational, risk-averse investor always prefers *more return* for the same risk, or *less risk* for the same return. Diversification achieves exactly this — without any cost — by exploiting the fact that imperfectly correlated assets partially cancel each other's fluctuations:

$$\sigma_p^2 = \sum_i w_i^2 \sigma_i^2 + \sum_i \sum_{j \neq i} w_i w_j \sigma_{ij}$$

When assets are negatively correlated ($\sigma_{ij} < 0$), the cross terms *reduce* total portfolio variance below even the lowest individual variance — a genuine free reduction in risk. Even weakly positive correlations provide meaningful diversification benefits.

The asset basket chosen for this project spans five distinct asset classes (equities, broad market ETF, long-duration bonds, gold, and crypto) precisely to maximise the diversification surface.

---

## Mathematical Methodology

### Portfolio Return

The expected return of a portfolio with weight vector $\mathbf{w} \in \mathbb{R}^N$ is the linearly weighted sum of individual expected returns:

$$R_p = \sum_{i=1}^{N} w_i R_i = \mathbf{w}^T \boldsymbol{\mu}$$

where $\boldsymbol{\mu} = [\mu_1, \mu_2, \dots, \mu_N]^T$ is the vector of annualised expected returns, estimated from historical log returns scaled by 252 trading days.

### Portfolio Variance

$$\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$$

where $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$ is the annualised covariance matrix of log returns. The quadratic form captures both individual variances (on the diagonal) and all pairwise covariances (off-diagonal).

### Log Returns vs. Simple Returns

Log returns $r_t = \ln(P_t / P_{t-1})$ are used for the optimizer because:

- They are **time-additive**: the multi-period log return equals the sum of single-period log returns, consistent with the normality assumption underlying MVO.
- They are **more symmetric** around zero, reducing the impact of large price moves on distributional estimates.

Simple returns $r_t = (P_t - P_{t-1}) / P_{t-1}$ are used for equity curves because compound wealth grows **multiplicatively**: $W_T = W_0 \prod_{t=1}^{T}(1 + r_t)$.

### Sharpe Ratio

The Sharpe ratio measures excess return per unit of total risk:

$$S = \frac{E[R_p] - R_f}{\sigma_p} = \frac{\mathbf{w}^T \boldsymbol{\mu} - R_f}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}}$$

where $R_f$ is the annualised risk-free rate (default: 4 %). Maximising $S$ identifies the portfolio on the tangency point of the Capital Market Line (CML) with the efficient frontier — the single optimal risky portfolio for a mean-variance investor.

### Bollinger Bands

Bollinger Bands are a volatility-adaptive envelope around a rolling mean, providing a measure of **mean reversion** potential:

$$\text{SMA}_t = \frac{1}{n} \sum_{k=0}^{n-1} P_{t-k}$$

$$\text{Upper}_t = \text{SMA}_t + 2\sigma_t, \qquad \text{Lower}_t = \text{SMA}_t - 2\sigma_t$$

where $\sigma_t$ is the rolling standard deviation over the same $n$-day window (default $n = 20$). Under a normal distribution, approximately 95 % of price observations will fall within the bands. Persistent excursions beyond the bands signal regime changes or momentum, rather than simple mean reversion.

The **bandwidth** metric $(\text{Upper} - \text{Lower}) / \text{SMA}$ normalises the spread, making it comparable across assets at different price levels and useful for low-volatility squeeze detection.

---

## The Optimization Problem

### Problem Formulation

The efficient frontier is the set of portfolios that minimise variance for each achievable level of expected return. Each point on the frontier solves:

$$\min_{\mathbf{w}} \quad \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$$

$$\text{subject to} \quad \mathbf{w}^T \boldsymbol{\mu} = \mu_{\text{target}}, \quad \sum_{i=1}^{N} w_i = 1, \quad w_i \geq 0 \; \forall i$$

The constraint $w_i \geq 0$ enforces a **long-only** portfolio, excluding short positions. The constraint $\sum w_i = 1$ ensures full investment (no cash allocation). Together they define a convex feasible set; the objective is a convex quadratic form — so the problem is a **Quadratic Program (QP)** with a unique global minimum.

### Maximum Sharpe via Charnes-Cooper Transformation

Maximising the Sharpe ratio is a **fractional program** (ratio of linear to quadratic terms), which is non-convex in the original variables. The Charnes-Cooper variable substitution converts it into a standard convex QP solvable by cvxpy without resorting to scipy or non-convex solvers:

Let $\kappa > 0$ and $\mathbf{y} = \kappa \mathbf{w}$. Then:

$$\min_{\mathbf{y}} \quad \mathbf{y}^T \boldsymbol{\Sigma} \mathbf{y}$$

$$\text{subject to} \quad (\boldsymbol{\mu} - R_f)^T \mathbf{y} = 1, \quad y_i \geq 0$$

The normalisation constraint fixes $\kappa$ implicitly. Optimal weights are recovered as $\mathbf{w}^* = \mathbf{y}^* / \sum_i y_i^*$. This approach is **globally optimal** and eliminates the need for nonlinear solvers.

### Covariance Regularisation: Ledoit-Wolf Shrinkage

The sample covariance matrix $\hat{\boldsymbol{\Sigma}}$ is rank-deficient when $T \lesssim N$ (number of observations close to number of assets), making the QP numerically unstable. Ledoit-Wolf shrinkage regularises $\hat{\boldsymbol{\Sigma}}$ toward a structured estimator $\mathbf{F}$ (scaled identity):

$$\boldsymbol{\Sigma}_{\text{LW}} = (1 - \alpha) \hat{\boldsymbol{\Sigma}} + \alpha \mathbf{F}$$

where the shrinkage intensity $\alpha$ is estimated analytically to minimise the expected squared Frobenius norm of the estimation error. The result is guaranteed **positive definite**, enabling stable Cholesky decomposition and reliable QP solutions.

---

## Key Insights & Results

### Effect of Adding Crypto to a Traditional Portfolio

Incorporating BTC-USD and ETH-USD into a stock/bond portfolio has two competing effects:

1. **Positive**: Very low long-run correlation with equities and bonds (~0.1–0.3) expands the **feasible set** — the efficient frontier shifts upward and to the left, offering higher return for the same risk.
2. **Negative**: Crypto's extreme **positive skewness** and **fat tails** violate the Gaussian normality assumption of MVO, causing the optimizer to over-allocate to crypto based on historical Sharpe ratios that may not persist.

In practice, the optimizer typically places a 5–15 % allocation to BTC-USD in the Max Sharpe portfolio, reflective of its historically high returns during the 2017–2024 bull cycles. Constrained variants (max weight per asset ≤ 10 %) are a recommended extension.

### Max Sharpe vs Equal-Weight: Maximum Drawdown Comparison

| Portfolio | Expected Return | Volatility | Max Drawdown |
|---|---|---|---|
| Max Sharpe (MVO) | Higher | Lower | *Typically smaller* |
| Equal-Weight (1/N) | Lower | Higher | *Typically larger* |

The Max Sharpe portfolio concentrates weight in low-volatility, high-return assets (GLD, TLT) while reducing allocation to high-volatility assets, producing a shallower drawdown profile. However, the equal-weight portfolio is more robust to **estimation error** in $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ — a well-documented finding in empirical finance (DeMiguel et al., 2009).

### Bollinger Band Regime Insights

During the 2022 equity bear market and 2022 crypto winter, Bollinger Band **bandwidth** for all assets simultaneously expanded — a systemic signal that cross-asset correlations had spiked toward 1 (diversification collapse). This is precisely when Ledoit-Wolf regularisation becomes most important.

---

## Dashboard — Tab-by-Tab Insights

The Streamlit dashboard (`streamlit run app.py`) is organised into three tabs, each targeting a different analytical question.

---

### Tab 1 — 📊 Portfolio Lab

**Question answered:** *"What is the best way to combine these assets, and what does the optimal portfolio look like?"*

#### Efficient Frontier Scatter Plot

**Definition:** The Efficient Frontier is the set of portfolios that achieve the *maximum possible return for each level of risk*. Any portfolio below or to the right of the curve is sub-optimal — you could get more return, less risk, or both, by rebalancing.

Every dot on the chart is a solved portfolio. The colour encodes its Sharpe Ratio (yellow = high, purple = low). Two special markers are overlaid:
- ⭐ **Max Sharpe** (gold star) — the single portfolio offering the best risk-adjusted return.
- ◆ **Min Variance** (cyan diamond) — the safest possible allocation.
- A dashed **Capital Market Line (CML)** shows the linear trade-off available when combining the Max Sharpe portfolio with the risk-free asset.

> **Example insight:** If NVDA and GLD are both selected, the frontier will shift upward compared to either asset alone — the negative/low correlation between a tech stock and gold creates a genuinely better risk-return surface.

#### Optimal Weights Pie Chart

**Definition:** The pie chart shows the exact percentage of capital the optimiser allocates to each asset in the Max Sharpe portfolio. Weights below 0.5 % are hidden to keep the chart readable.

> **Example insight:** Gold (GLD) often receives a disproportionately *large* weight compared to its market cap, because its low correlation with equities contributes more to risk reduction than its raw return suggests.

#### Correlation Heatmap

**Definition:** The heatmap shows the **Pearson correlation coefficient** $\rho_{ij} \in [-1, 1]$ between each pair of asset's daily log returns. Red = negative correlation (assets move opposite), blue = positive correlation (assets move together).

| Value | Meaning |
|---|---|
| $\rho \approx +1$ | Assets are nearly identical in movement — no diversification benefit |
| $\rho \approx 0$ | Uncorrelated — maximum diversification benefit |
| $\rho \approx -1$ | Perfect hedge — one rises when the other falls |

> **Example insight:** SPY and AAPL will typically show $\rho \approx 0.75$–$0.85$, meaning they move together most of the time. TLT (long bonds) often shows negative or near-zero correlation with equities — this is why a stock/bond mix has historically reduced drawdown.

---

### Tab 2 — 🔍 Asset Deep-Dive

**Question answered:** *"How does each individual asset behave over time, and how does the optimal portfolio compare to just holding SPY?"*

#### Bollinger Bands Chart

**Definition:** Bollinger Bands place a volatility envelope around a rolling 20-day SMA. The upper and lower bands sit $2\sigma$ above and below the SMA. Under a normal distribution, ~95 % of prices should stay *inside* the bands.

| Signal | Interpretation |
|---|---|
| Price touches **Upper Band** | Asset is trading at an elevated volatility excursion — potential mean-reversion zone, or the start of a breakout |
| Price touches **Lower Band** | Asset is trading at a depressed excursion — potential bounce zone, or the start of a downtrend |
| **Narrow bandwidth** | Volatility squeeze — often precedes a large directional move |
| **Wide bandwidth** | Elevated volatility regime (e.g., earnings, macro shock) |

> **Example insight:** During March 2020 (COVID crash), AAPL's price broke far below its lower Bollinger Band and the bandwidth exploded — a clear signal of a high-volatility regime, not a normal buying opportunity.

#### Cumulative Returns: Portfolio vs SPY Benchmark

**Definition:** Starting from a base of 0 %, this chart shows the total percentage gain (or loss) of $1 invested in the Max Sharpe portfolio versus $1 invested in SPY (the S&P 500 proxy). It answers: *"Did active optimisation actually beat buying-and-holding the market index?"*

> **Example insight:** The Max Sharpe portfolio, by weighting GLD heavily, tends to outperform SPY in bear markets (2022) but may lag during strong equity bull runs (2023–2024 AI rally), because gold's absolute return is lower than NVDA's during those regimes.

#### Maximum Drawdown (MDD) Metric Cards

**Definition:** MDD is the single largest peak-to-trough loss an investor would have experienced if they bought at the worst possible peak and sold at the worst subsequent trough:

$$\text{MDD} = \min_t \left( \frac{V_t}{\max_{s \leq t} V_s} - 1 \right)$$

A lower (less negative) MDD means the portfolio holds up better during crashes — a critical concern for investors who cannot afford to wait for recovery.

> **Example insight:** A typical 8-year backtest may show MDD ≈ −30 % for the Max Sharpe portfolio vs ≈ −34 % for SPY — the optimiser's diversification into bonds and gold meaningfully cushions the worst drawdowns.

#### Rolling Drawdown Chart

**Definition:** Unlike MDD (a single number), the rolling drawdown chart shows the drawdown level at *every point in time*. Deep troughs in the chart correspond to major market dislocations.

> **Example insight:** The rolling drawdown chart will show sharp dips in March 2020 and 2022 for both series. The depth and recovery speed of the portfolio vs SPY reveals whether the optimiser's asset mix truly shields investors during those specific crises.

---

### Tab 3 — 🧪 Robustness Analysis

**Question answered:** *"Can I trust the optimiser's output? Is it actually better than chance, and which assets are secretly driving the risk?"*

#### Monte Carlo Simulation vs. Efficient Frontier

**Definition:** 5 000 random portfolios are generated by sampling weights uniformly from the allocation simplex (Dirichlet distribution). Each random portfolio is plotted as a grey dot. The Efficient Frontier is overlaid as a coloured line.

**Why it matters:** The frontier should form the *upper-left boundary* of the grey cloud. If random portfolios reach or exceed the frontier, the optimiser's solution is not genuinely optimal and the problem is suspect.

> **Example insight:** The Max Sharpe portfolio (red star) will have a higher Sharpe ratio than ~95 %+ of all 5 000 random portfolios — this is a live numerical proof that the Charnes-Cooper QP found the true optimal solution, not just a local approximation.

#### 30-Day Rolling Volatility

**Definition:** Annualised volatility calculated over a rolling 30-day window of daily log returns, plotted through time for both the optimal portfolio and SPY. It shows *when* the portfolio was risky, not just its long-run average risk.

$$\sigma_t^{\text{rolling}} = \text{std}(r_{t-29}, \ldots, r_t) \times \sqrt{252}$$

> **Example insight:** During the 2020 COVID crash, SPY's rolling volatility spiked to ~80 % annualised. If the Max Sharpe portfolio held a large GLD allocation, its rolling vol spike would be significantly smaller — demonstrating that the diversification benefit was *realised* in the exact moment it was most needed.

#### Asset Contribution to Risk (Bar Chart)

**Definition:** This chart decomposes total portfolio variance into each asset's individual contribution using the **Marginal Risk Contribution (MRC)** formula:

$$RC_i = \frac{w_i \cdot (\boldsymbol{\Sigma} \mathbf{w})_i}{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}$$

The tomato bars show **risk %** (share of total variance); the blue bars show **weight %**. They are not the same.

| Scenario | What it Reveals |
|---|---|
| Risk bar >> Weight bar | This asset is a "risk amplifier" — highly correlated with others, punching above its weight |
| Risk bar << Weight bar | This asset is a "risk reducer" — low correlation, it actually lowers portfolio variance per dollar allocated |

> **Example insight:** NVDA may hold only 24 % of the portfolio weight but contribute 53 % of total variance — its high volatility and positive correlation with other tech exposure makes it the dominant risk driver. GLD, by contrast, often contributes *less* risk than its weight would suggest because of its low correlations.

---

## Engineering Implementation

### Why `cvxpy`?

`cvxpy` is a domain-specific language for convex optimisation embedded in Python. Key advantages for this project:

- **Disciplined Convex Programming (DCP)**: The library verifies at construction time that the problem is convex, preventing silent errors from non-convex reformulations.
- **Solver-agnostic**: The same `Problem` definition can route to CLARABEL, OSQP, ECOS, or SCS without code changes — crucial for numerical robustness.
- **Readable modelling**: The code reads like mathematical notation (`cp.quad_form(w, Σ)`) rather than low-level matrix operations, reducing implementation bugs.

### Why Streamlit?

Streamlit converts a Python script into an interactive web dashboard without requiring HTML/CSS/JS. `@st.cache_data` decorators memoize the expensive data download and optimisation steps by hashing function arguments — subsequent interactions (selecting a different ticker, adjusting the risk-free rate) reuse cached results in milliseconds.

### Importance of Adjusted Close Prices

Using **Adjusted Close** (rather than raw Close) is essential for valid backtesting because it accounts for:

- **Dividends**: Without adjustment, a dividend payment appears as a price drop, artificially inflating negative return signal.
- **Stock splits**: A 4:1 split halves the price; unadjusted data shows a -75 % "return" on that day.

Adjusted Close ensures that the return time-series $r_t = \ln(P_t^{\text{adj}} / P_{t-1}^{\text{adj}})$ reflects the true economic return to a buy-and-hold investor.

### Parquet over CSV for AI Engineering

The data pipeline stores processed data in the **Apache Parquet** columnar format:

- **Schema preservation**: Data types (float64, DatetimeTZDtype) are stored in file metadata, eliminating silent type coercions on reload.
- **Columnar compression**: Snappy encoding yields 3–10× smaller files than CSV for financial time-series.
- **Ecosystem integration**: Native zero-copy interchange with pandas, polars, Apache Arrow, DuckDB, and Spark — critical for scaling to production ML pipelines.

---

## Project Structure

```
Portfolio_Mean_Variance_Optimization/
├── data_loader.py       # Data ingestion & cleaning
├── indicators.py        # Bollinger Bands, Drawdown
├── optimizer.py         # MVO, Max Sharpe, Frontier
├── app.py               # Streamlit dashboard (all three projects)
├── data/
│   ├── prices.parquet       # Cleaned Adj Close 
│   └── log_returns.parquet  # Log returns 
└── README.md
```

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install yfinance pandas numpy pyarrow cvxpy scikit-learn streamlit plotly

# 3. Verify individual modules
python data_loader.py    # Downloads data, prints summary, saves Parquet
python optimizer.py      # Runs MVO, prints Max Sharpe + Min Var weights
python indicators.py     # Computes Bollinger Bands, prints MDD

# 4. Launch the dashboard
streamlit run app.py
```

---

## Future Extensions

### Black-Litterman Model

The Black-Litterman (BL) model addresses the most significant weakness of classical MVO: sensitivity to $\boldsymbol{\mu}$ estimation errors. BL starts from **market equilibrium returns** (implied by market-cap weights via reverse optimisation) and blends them with **investor views** expressed as probabilistic statements:

$$\boldsymbol{\mu}_{\text{BL}} = \left[(\tau \boldsymbol{\Sigma})^{-1} + \mathbf{P}^T \boldsymbol{\Omega}^{-1} \mathbf{P}\right]^{-1} \left[(\tau \boldsymbol{\Sigma})^{-1} \boldsymbol{\Pi} + \mathbf{P}^T \boldsymbol{\Omega}^{-1} \mathbf{Q}\right]$$

This produces far more stable and economically interpretable weight allocations, particularly for large asset universes.

### LSTM-Based Return Predictions

Instead of using the historical sample mean $\hat{\boldsymbol{\mu}}$, a Long Short-Term Memory (LSTM) network trained on macro features (VIX, yield curve, sentiment scores) can generate forward-looking conditional return forecasts $\hat{\boldsymbol{\mu}}_t = f_\theta(\mathbf{x}_t)$. Plugging these into the MVO objective creates an **AI-driven frontier** that adapts to regime changes rather than relying on stationary historical averages.

### Reinforcement Learning Portfolio Agents

Deep reinforcement learning (e.g., Proximal Policy Optimization) can be framed as a sequential portfolio rebalancing problem where the agent observes the state (prices, economic indicators) and outputs continuous weight allocations, optimising a reward function tied to risk-adjusted return. This removes the Gaussian return assumption entirely and handles transaction costs naturally within the reward signal.

### Additional Extensions

- **Risk parity**: Allocate equal **risk contribution** per asset rather than equal weight.
- **CVaR optimisation**: Replace variance with Conditional Value-at-Risk as the risk measure for fat-tailed asset distributions.
- **Multi-period rebalancing**: Rolling window re-optimisation with transaction cost penalties.
- **Macro-factor overlays**: Integrate Fama-French factors or yield curve signals as return predictors.

---

## References

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance, 7(1), 77–91.
- Sharpe, W. F. (1966). *Mutual Fund Performance*. Journal of Business, 39(1), 119–138.
- Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis, 88(2), 365–411.
- Charnes, A., & Cooper, W. W. (1962). *Programming with linear fractional functionals*. Naval Research Logistics Quarterly, 9(3–4), 181–186.
- DeMiguel, V., Garlappi, L., & Uppal, R. (2009). *Optimal versus naive diversification: How inefficient is the 1/N portfolio strategy?* Review of Financial Studies, 22(5), 1915–1953.
- Bollinger, J. (2002). *Bollinger on Bollinger Bands*. McGraw-Hill.