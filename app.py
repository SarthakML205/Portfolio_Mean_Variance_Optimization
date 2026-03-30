"""
app.py
======
Portfolio Analytics Suite Dashboard
----------------------------------------------------
Two-tab Streamlit application:

  Tab 1 — Portfolio Lab
      * Efficient Frontier scatter plot (Plotly)
      * Max Sharpe portfolio weight pie chart
      * Asset correlation heatmap

  Tab 2 — Asset Deep-Dive
      * Bollinger Bands overlay chart for any selected ticker
      * Cumulative Returns: Max Sharpe portfolio vs SPY benchmark
      * Maximum Drawdown metrics (st.metric cards)

Run
---
    streamlit run app.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import DEFAULT_TICKERS, PortfolioData
from indicators import (
    bollinger_bands,
    cumulative_returns,
    max_drawdown,
    portfolio_equity_curve,
    rolling_drawdown,
    rolling_volatility,
)
from optimizer import (
    OptimResult,
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    monte_carlo_portfolios,
    risk_contribution,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Analytics Suite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Downloading market data …")
def load_data(
    tickers: tuple[str, ...],
    period_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and cache Adj Close prices + log returns via PortfolioData.

    Parameters are immutable (tuple) so Streamlit's hash-based cache
    correctly invalidates when tickers or period change.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (prices, log_returns)
    """
    portfolio = PortfolioData(tickers=list(tickers), period_years=period_years).load()
    return portfolio.get_prices(), portfolio.get_returns()


@st.cache_data(show_spinner="Running portfolio optimisation …")
def run_optimization(
    # Streamlit requires hashable args for cache; pass returns as JSON-serialisable key
    log_returns_json: str,
    rf: float,
) -> tuple[OptimResult, OptimResult, list[OptimResult]]:
    """
    Run Max Sharpe, Min Variance, and Efficient Frontier optimisations.

    ``log_returns_json`` is a stable string key derived from the DataFrame;
    the actual DataFrame is reconstructed inside the function.

    Returns
    -------
    tuple[OptimResult, OptimResult, list[OptimResult]]
        (max_sharpe, min_var, frontier)
    """
    log_returns = pd.read_json(log_returns_json)
    ms = max_sharpe_portfolio(log_returns, risk_free_rate=rf)
    mv = min_variance_portfolio(log_returns, risk_free_rate=rf)
    frontier = efficient_frontier(log_returns, n_points=50, risk_free_rate=rf)
    return ms, mv, frontier


@st.cache_data(show_spinner="Running Monte Carlo simulation (5 000 portfolios) …")
def run_monte_carlo(
    log_returns_json: str,
    rf: float,
    n_sim: int = 5000,
) -> pd.DataFrame:
    """
    Run the Monte Carlo portfolio simulation and cache the result.

    Uses the same ``log_returns_json`` string key as ``run_optimization`` so
    the cache invalidates whenever tickers or period change.  With 5 000
    simulations the vectorised implementation runs in ~0.5 s; subsequent
    renders are instant from cache.

    Returns
    -------
    pd.DataFrame
        Columns: Return, Volatility, Sharpe (+ w_{ticker} weight columns).
    """
    log_returns = pd.read_json(log_returns_json)
    return monte_carlo_portfolios(log_returns, n_sim=n_sim, risk_free_rate=rf)


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")

    selected_tickers: list[str] = st.multiselect(
        label="Select Assets",
        options=DEFAULT_TICKERS,
        default=DEFAULT_TICKERS,
        help="Choose 2 or more assets for the portfolio analysis.",
    )

    period_years: int = st.slider(
        label="History (years)",
        min_value=2,
        max_value=10,
        value=8,
        step=1,
    )

    rf_pct: float = st.number_input(
        label="Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.1,
        format="%.1f",
    )
    rf: float = rf_pct / 100.0

    bb_ticker: str = st.selectbox(
        label="Bollinger Band Ticker",
        options=selected_tickers if selected_tickers else DEFAULT_TICKERS,
        index=0,
    )

    bb_window: int = st.slider("BB Window (days)", 10, 50, 20, 5)

    st.markdown("---")
    st.caption("Data: Yahoo Finance via yfinance · Optimiser: cvxpy (CLARABEL)")

# ---------------------------------------------------------------------------
# Guard: need at least 2 tickers
# ---------------------------------------------------------------------------

if len(selected_tickers) < 2:
    st.warning("Please select at least **2 assets** in the sidebar to continue.")
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

prices, log_returns = load_data(tuple(selected_tickers), period_years)

# Derive a stable JSON key for the optimisation cache
log_returns_json: str = log_returns.to_json(date_format="iso")

# ---------------------------------------------------------------------------
# Run optimisation
# ---------------------------------------------------------------------------

try:
    ms_result, mv_result, frontier = run_optimization(log_returns_json, rf)
    optim_ok = True
except Exception as exc:
    st.error(f"Optimisation failed: {exc}")
    optim_ok = False

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Portfolio Lab", "🔍 Asset Deep-Dive", "🧪 Robustness Analysis"])

# ============================================================
# TAB 1 — Portfolio Lab
# ============================================================

with tab1:
    st.header("Portfolio Lab")

    if not optim_ok:
        st.error("Optimisation unavailable — see error above.")
        st.stop()

    col_left, col_right = st.columns([3, 2], gap="large")

    # ----------------------------------------------------------
    # Efficient Frontier scatter
    # ----------------------------------------------------------
    with col_left:
        st.subheader("Efficient Frontier")

        frontier_df = pd.DataFrame(
            {
                "Volatility": [p.volatility for p in frontier],
                "Return": [p.expected_return for p in frontier],
                "Sharpe": [p.sharpe_ratio for p in frontier],
            }
        )

        fig_ef = px.scatter(
            frontier_df,
            x="Volatility",
            y="Return",
            color="Sharpe",
            color_continuous_scale="Viridis",
            labels={
                "Volatility": "Annualised Volatility (σ)",
                "Return": "Annualised Expected Return (μ)",
                "Sharpe": "Sharpe Ratio",
            },
            title="Efficient Frontier — Risk / Return Trade-off",
        )

        # Overlay Max Sharpe marker
        fig_ef.add_trace(
            go.Scatter(
                x=[ms_result.volatility],
                y=[ms_result.expected_return],
                mode="markers",
                marker=dict(symbol="star", size=18, color="gold", line=dict(width=1, color="black")),
                name=f"Max Sharpe ({ms_result.sharpe_ratio:.2f})",
            )
        )

        # Overlay Min Variance marker
        fig_ef.add_trace(
            go.Scatter(
                x=[mv_result.volatility],
                y=[mv_result.expected_return],
                mode="markers",
                marker=dict(symbol="diamond", size=14, color="cyan", line=dict(width=1, color="black")),
                name="Min Variance",
            )
        )

        # Capital Market Line (CML) from risk-free rate through Max Sharpe
        cml_x = np.linspace(0, ms_result.volatility * 1.5, 50)
        cml_y = rf + ms_result.sharpe_ratio * cml_x
        fig_ef.add_trace(
            go.Scatter(
                x=cml_x,
                y=cml_y,
                mode="lines",
                line=dict(dash="dash", color="orange", width=1.5),
                name="Capital Market Line",
            )
        )

        fig_ef.update_layout(
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            coloraxis_colorbar=dict(title="Sharpe"),
        )
        st.plotly_chart(fig_ef, use_container_width=True)

    # ----------------------------------------------------------
    # Max Sharpe weights pie + summary metrics
    # ----------------------------------------------------------
    with col_right:
        st.subheader("Max Sharpe Weights")

        # Filter out negligible weights for cleaner pie
        w_filtered = ms_result.weights[ms_result.weights >= 0.005]
        fig_pie = px.pie(
            values=w_filtered.values,
            names=w_filtered.index,
            title="Optimal Allocation — Max Sharpe",
            hole=0.35,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=True, height=340, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("**Portfolio Metrics**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Exp. Return", f"{ms_result.expected_return:.2%}")
        m2.metric("Volatility", f"{ms_result.volatility:.2%}")
        m3.metric("Sharpe Ratio", f"{ms_result.sharpe_ratio:.2f}")

        st.markdown("**Min Variance Metrics**")
        n1, n2, n3 = st.columns(3)
        n1.metric("Exp. Return", f"{mv_result.expected_return:.2%}")
        n2.metric("Volatility", f"{mv_result.volatility:.2%}")
        n3.metric("Sharpe Ratio", f"{mv_result.sharpe_ratio:.2f}")

    st.markdown("---")

    # ----------------------------------------------------------
    # Correlation heatmap
    # ----------------------------------------------------------
    st.subheader("Asset Correlation Heatmap")

    corr: pd.DataFrame = log_returns.corr().round(3)
    fig_hm = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pairwise Log-Return Correlations",
        aspect="auto",
    )
    fig_hm.update_layout(height=420)
    st.plotly_chart(fig_hm, use_container_width=True)

    # ----------------------------------------------------------
    # Weights table
    # ----------------------------------------------------------
    with st.expander("Full Weight Tables"):
        wt_df = pd.DataFrame(
            {
                "Max Sharpe": ms_result.weights.map("{:.2%}".format),
                "Min Variance": mv_result.weights.map("{:.2%}".format),
            }
        )
        st.dataframe(wt_df, use_container_width=True)


# ============================================================
# TAB 2 — Asset Deep-Dive
# ============================================================

with tab2:
    st.header("Asset Deep-Dive")

    col_bb, col_perf = st.columns([3, 2], gap="large")

    # ----------------------------------------------------------
    # Bollinger Bands
    # ----------------------------------------------------------
    with col_bb:
        st.subheader(f"Bollinger Bands — {bb_ticker}")

        bb_df = bollinger_bands(prices, bb_ticker, window=bb_window)
        # Show only the last 2 years for readability
        bb_display = bb_df.iloc[-504:].dropna()

        fig_bb = go.Figure()

        # Shaded band region
        fig_bb.add_trace(
            go.Scatter(
                x=bb_display.index,
                y=bb_display["upper"],
                mode="lines",
                line=dict(color="rgba(100,149,237,0.4)", width=1),
                name="Upper Band",
            )
        )
        fig_bb.add_trace(
            go.Scatter(
                x=bb_display.index,
                y=bb_display["lower"],
                mode="lines",
                line=dict(color="rgba(100,149,237,0.4)", width=1),
                fill="tonexty",
                fillcolor="rgba(100,149,237,0.12)",
                name="Lower Band",
            )
        )

        # SMA
        fig_bb.add_trace(
            go.Scatter(
                x=bb_display.index,
                y=bb_display["sma"],
                mode="lines",
                line=dict(color="orange", width=1.5, dash="dot"),
                name=f"{bb_window}d SMA",
            )
        )

        # Price
        fig_bb.add_trace(
            go.Scatter(
                x=bb_display.index,
                y=bb_display["price"],
                mode="lines",
                line=dict(color="white", width=1.8),
                name=bb_ticker,
            )
        )

        fig_bb.update_layout(
            title=f"{bb_ticker} — {bb_window}-day Bollinger Bands (last 2 years)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=480,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig_bb, use_container_width=True)

    # ----------------------------------------------------------
    # Performance comparison: Max Sharpe vs SPY
    # ----------------------------------------------------------
    with col_perf:
        st.subheader("Performance vs SPY Benchmark")

        if optim_ok:
            port_equity = portfolio_equity_curve(prices, ms_result.weights)
        else:
            # Fall back to equal-weight if optimisation failed
            eq_weights = pd.Series(
                [1 / len(selected_tickers)] * len(selected_tickers),
                index=selected_tickers,
            )
            port_equity = portfolio_equity_curve(prices, eq_weights)

        # SPY benchmark — may not be in selected tickers, load separately
        spy_in_prices = "SPY" in prices.columns
        if spy_in_prices:
            spy_prices_series = prices[["SPY"]]
        else:
            @st.cache_data(show_spinner=False)
            def _load_spy(period_years: int) -> pd.DataFrame:
                p = PortfolioData(tickers=["SPY"], period_years=period_years).load()
                return p.get_prices()

            spy_prices_series = _load_spy(period_years)

        spy_equity = portfolio_equity_curve(spy_prices_series, pd.Series({"SPY": 1.0}))

        # Align date ranges
        common_idx = port_equity.index.intersection(spy_equity.index)
        port_aligned = (port_equity.loc[common_idx] - 1) * 100   # → % return
        spy_aligned = (spy_equity.loc[common_idx] - 1) * 100

        perf_df = pd.DataFrame(
            {"Max Sharpe Portfolio": port_aligned, "SPY Benchmark": spy_aligned}
        )

        fig_perf = px.line(
            perf_df,
            labels={"value": "Cumulative Return (%)", "variable": ""},
            title="Cumulative Return: Portfolio vs SPY",
            color_discrete_map={
                "Max Sharpe Portfolio": "gold",
                "SPY Benchmark": "deepskyblue",
            },
        )
        fig_perf.update_layout(
            height=300,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        # MDD metrics
        port_mdd = max_drawdown(port_equity)
        spy_mdd = max_drawdown(spy_equity)

        st.markdown("**Maximum Drawdown**")
        d1, d2 = st.columns(2)
        d1.metric(
            "Max Sharpe",
            f"{port_mdd:.2%}",
            delta=f"{port_mdd - spy_mdd:.2%} vs SPY",
            delta_color="inverse",
        )
        d2.metric("SPY", f"{spy_mdd:.2%}")

    st.markdown("---")

    # ----------------------------------------------------------
    # Rolling Drawdown
    # ----------------------------------------------------------
    st.subheader("Rolling Drawdown — Portfolio vs SPY")

    port_dd = rolling_drawdown(port_equity.loc[common_idx]) * 100
    spy_dd = rolling_drawdown(spy_equity.loc[common_idx]) * 100

    dd_df = pd.DataFrame(
        {"Max Sharpe Portfolio": port_dd, "SPY Benchmark": spy_dd}
    )

    fig_dd = px.line(
        dd_df,
        labels={"value": "Drawdown (%)", "variable": ""},
        title="Rolling Drawdown (%)",
        color_discrete_map={
            "Max Sharpe Portfolio": "gold",
            "SPY Benchmark": "deepskyblue",
        },
    )
    fig_dd.update_layout(height=280, template="plotly_dark", legend=dict(orientation="h", y=-0.3))
    fig_dd.add_hline(y=0, line_dash="dot", line_color="grey", line_width=0.8)
    st.plotly_chart(fig_dd, use_container_width=True)

    # ----------------------------------------------------------
    # Individual asset cumulative returns table
    # ----------------------------------------------------------
    with st.expander("Individual Asset Cumulative Returns"):
        cum = cumulative_returns(prices)
        # Format last value as % for each ticker
        final_row = cum.iloc[-1].map("{:.2%}".format).rename("Total Return")
        st.dataframe(final_row.to_frame(), use_container_width=True)


# ============================================================
# TAB 3 — Robustness Analysis
# ============================================================

with tab3:
    st.header("Robustness Analysis")
    st.caption(
        "Three charts that answer: *Did the optimiser find the edge of possibility? "
        "How does the portfolio behave during crashes? Which asset truly drives the risk?*"
    )

    if not optim_ok:
        st.error("Optimisation unavailable — see error above.")
        st.stop()

    # ----------------------------------------------------------
    # 1. Monte Carlo vs Efficient Frontier
    # ----------------------------------------------------------
    st.subheader("Monte Carlo Simulation vs. Efficient Frontier")
    st.markdown(
        "**5 000 random long-only portfolios** (grey cloud) are overlaid on the "
        "**Efficient Frontier** (coloured line) and the **Max Sharpe** star. "
        "The frontier traces the upper-left boundary of the cloud — proof the "
        "optimiser found the mathematically optimal edge of achievable risk-return space."
    )

    mc_df = run_monte_carlo(log_returns_json, rf, n_sim=5000)

    fig_mc = go.Figure()

    # Grey Monte Carlo cloud
    fig_mc.add_trace(
        go.Scatter(
            x=mc_df["Volatility"],
            y=mc_df["Return"],
            mode="markers",
            marker=dict(
                size=3,
                color=mc_df["Sharpe"],
                colorscale="Greys",
                opacity=0.35,
                showscale=False,
            ),
            name="Random Portfolios (MC)",
            hovertemplate="Vol: %{x:.3f}<br>Return: %{y:.3f}<br>Sharpe: %{customdata:.2f}",
            customdata=mc_df["Sharpe"].values,
        )
    )

    # Efficient Frontier — sorted by volatility for a smooth line
    frontier_df3 = pd.DataFrame(
        {
            "Volatility": [p.volatility for p in frontier],
            "Return": [p.expected_return for p in frontier],
            "Sharpe": [p.sharpe_ratio for p in frontier],
        }
    ).sort_values("Volatility")

    fig_mc.add_trace(
        go.Scatter(
            x=frontier_df3["Volatility"],
            y=frontier_df3["Return"],
            mode="lines+markers",
            marker=dict(
                size=6,
                color=frontier_df3["Sharpe"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe", x=1.02),
            ),
            line=dict(color="gold", width=2.5),
            name="Efficient Frontier",
            hovertemplate="Vol: %{x:.3f}<br>Return: %{y:.3f}<br>Sharpe: %{customdata:.2f}",
            customdata=frontier_df3["Sharpe"].values,
        )
    )

    # Max Sharpe star
    fig_mc.add_trace(
        go.Scatter(
            x=[ms_result.volatility],
            y=[ms_result.expected_return],
            mode="markers",
            marker=dict(symbol="star", size=20, color="red", line=dict(width=1, color="white")),
            name=f"Max Sharpe ({ms_result.sharpe_ratio:.2f})",
        )
    )

    fig_mc.update_layout(
        height=520,
        template="plotly_dark",
        xaxis_title="Annualised Volatility (σ)",
        yaxis_title="Annualised Expected Return (μ)",
        title="5 000 Random Portfolios vs. Efficient Frontier",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    pct_beaten = (mc_df["Sharpe"] < ms_result.sharpe_ratio).mean()
    st.markdown(
        f"> The Max Sharpe portfolio beats **{pct_beaten:.1%}** of all random "
        f"portfolios in Sharpe ratio — confirming the optimiser's superiority over naive allocation."
    )

    st.markdown("---")

    # ----------------------------------------------------------
    # 2. Rolling Vol  +  3. Risk Contribution  (side by side)
    # ----------------------------------------------------------
    col_rvol, col_rc = st.columns([3, 2], gap="large")

    with col_rvol:
        st.subheader("30-Day Rolling Volatility")
        st.markdown(
            "Annualised rolling volatility shows *when* the portfolio was risky, not just "
            "the long-run average. Spikes correspond to market dislocations (COVID crash Mar 2020, "
            "rate hike cycle 2022, crypto winter 2022)."
        )

        rvol_df = rolling_volatility(
            log_returns, ms_result.weights, benchmark_col="SPY", window=30
        ).dropna() * 100  # → percentage

        fig_rvol = go.Figure()
        fig_rvol.add_trace(
            go.Scatter(
                x=rvol_df.index,
                y=rvol_df["Portfolio"],
                mode="lines",
                line=dict(color="gold", width=1.8),
                name="Max Sharpe Portfolio",
                fill=None,
            )
        )
        fig_rvol.add_trace(
            go.Scatter(
                x=rvol_df.index,
                y=rvol_df["Benchmark"],
                mode="lines",
                line=dict(color="deepskyblue", width=1.8),
                name="SPY Benchmark",
                fill="tonexty",
                fillcolor="rgba(255,215,0,0.07)",  # shaded divergence region
            )
        )
        fig_rvol.update_layout(
            height=400,
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Annualised Volatility (%)",
            title="30-Day Rolling Volatility: Portfolio vs SPY",
            legend=dict(orientation="h", yanchor="bottom", y=-0.28),
        )
        st.plotly_chart(fig_rvol, use_container_width=True)

        r1, r2, r3 = st.columns(3)
        avg_p = rvol_df["Portfolio"].mean()
        avg_b = rvol_df["Benchmark"].mean()
        r1.metric("Avg Portfolio Vol", f"{avg_p:.1f}%")
        r2.metric("Avg SPY Vol", f"{avg_b:.1f}%")
        r3.metric(
            "Vol Reduction vs SPY",
            f"{avg_b - avg_p:.1f}pp",
            delta=f"{avg_b - avg_p:.1f}pp",
            delta_color="normal",
        )

    with col_rc:
        st.subheader("Asset Contribution to Risk")
        st.markdown(
            "**Weight ≠ Risk.** Tomato bars show each asset's share of total portfolio "
            "variance; blue bars show its weight. High-correlation assets punch above their weight."
        )

        rc = risk_contribution(
            ms_result.weights[ms_result.weights > 0], log_returns
        ).sort_values(ascending=False)
        w_pct = (ms_result.weights * 100).reindex(rc.index).fillna(0)
        rc_pct = rc * 100

        fig_rc = go.Figure()
        fig_rc.add_trace(
            go.Bar(
                x=rc.index,
                y=rc_pct.values,
                name="Risk Contribution (%)",
                marker_color="tomato",
                text=rc_pct.map("{:.1f}%".format),
                textposition="outside",
            )
        )
        fig_rc.add_trace(
            go.Bar(
                x=rc.index,
                y=w_pct.values,
                name="Portfolio Weight (%)",
                marker_color="steelblue",
                opacity=0.65,
                text=w_pct.map("{:.1f}%".format),
                textposition="outside",
            )
        )
        fig_rc.update_layout(
            height=400,
            template="plotly_dark",
            barmode="group",
            xaxis_title="Asset",
            yaxis_title="Percentage (%)",
            title="Risk Contribution vs. Portfolio Weight",
            yaxis=dict(range=[0, max(rc_pct.max(), w_pct.max()) * 1.3]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.28),
        )
        st.plotly_chart(fig_rc, use_container_width=True)

        top_ticker = rc_pct.idxmax()
        st.info(
            f"**{top_ticker}** owns **{rc_pct[top_ticker]:.1f}%** of total variance "
            f"while holding only **{w_pct[top_ticker]:.1f}%** of the portfolio weight."
        )
