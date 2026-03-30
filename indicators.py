"""
indicators.py
=============
Technical Analysis Engine
--------------------------------------------
Provides stateless helper functions for:

* **Bollinger Bands** — 20-day SMA ± 2σ volatility envelope
* **Cumulative Returns** — compound wealth index from price series
* **Maximum Drawdown (MDD)** — worst peak-to-trough loss
* **Rolling Drawdown** — full drawdown time-series for plotting
* **Portfolio Equity Curve** — weighted blend of individual asset returns

All functions are pure (no side effects) and return ``pd.DataFrame`` or
``pd.Series`` objects, making them directly usable in Streamlit / Plotly.

Usage
-----
    from data_loader  import PortfolioData
    from optimizer    import max_sharpe_portfolio
    from indicators   import (
        bollinger_bands, cumulative_returns,
        max_drawdown,    portfolio_equity_curve,
    )

    portfolio   = PortfolioData().load()
    prices      = portfolio.get_prices()
    log_returns = portfolio.get_returns()

    bb      = bollinger_bands(prices, "AAPL")
    cum_ret = cumulative_returns(prices)
    result  = max_sharpe_portfolio(log_returns)
    equity  = portfolio_equity_curve(prices, result.weights)
    mdd     = max_drawdown(equity)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    prices: pd.DataFrame,
    ticker: str,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """
    Compute Bollinger Bands for a single ticker.

    Bollinger Bands are a volatility-based envelope around a moving average:

        SMA    = rolling mean over ``window`` days
        Upper  = SMA + n_std × rolling σ
        Lower  = SMA − n_std × rolling σ

    A narrow bandwidth (upper − lower)/SMA signals low volatility / consolidation;
    a wide bandwidth signals elevated volatility or trending conditions.
    Price touching the upper band is not inherently a sell signal — in a strong
    trend, prices can "walk the band" for extended periods.

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned Adj Close price matrix (rows = dates, columns = tickers).
    ticker : str
        Column label to compute bands for.
    window : int
        Rolling window size in trading days (default 20).
    n_std : float
        Number of standard deviations for the band width (default 2.0).

    Returns
    -------
    pd.DataFrame
        Columns: ``price``, ``sma``, ``upper``, ``lower``, ``bandwidth``.
        First ``window - 1`` rows will contain NaN (insufficient history).
    """
    if ticker not in prices.columns:
        raise KeyError(f"Ticker '{ticker}' not found in prices DataFrame.")

    price_series: pd.Series = prices[ticker]
    sma: pd.Series = price_series.rolling(window=window, min_periods=window).mean()
    std: pd.Series = price_series.rolling(window=window, min_periods=window).std(ddof=1)

    upper: pd.Series = sma + n_std * std
    lower: pd.Series = sma - n_std * std

    # bandwidth: normalised spread — useful for volatility regime detection
    bandwidth: pd.Series = (upper - lower) / sma

    return pd.DataFrame(
        {
            "price": price_series,
            "sma": sma,
            "upper": upper,
            "lower": lower,
            "bandwidth": bandwidth,
        }
    )


# ---------------------------------------------------------------------------
# Cumulative Returns
# ---------------------------------------------------------------------------

def cumulative_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative simple returns for every ticker.

    Uses **simple returns** (not log returns) for the cumulative chart because
    compound wealth grows multiplicatively:

        Cumulative Rₜ = ∏(1 + rᵢ) − 1

    where rᵢ = (Pᵢ − Pᵢ₋₁) / Pᵢ₋₁  (simple daily return).

    Note: log returns are used for the MVO (additive, Gaussian-friendly), but
    simple returns are the correct choice for plotting wealth curves.

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned Adj Close price matrix.

    Returns
    -------
    pd.DataFrame
        Cumulative return for each ticker, starting at 0.0 on the first date.
        Shape matches ``prices`` (first row is 0.0).
    """
    simple_ret: pd.DataFrame = prices.pct_change()
    cum_ret: pd.DataFrame = (1 + simple_ret).cumprod() - 1
    # Set the first row to 0 for a clean start-at-zero chart
    cum_ret.iloc[0] = 0.0
    return cum_ret


# ---------------------------------------------------------------------------
# Maximum Drawdown
# ---------------------------------------------------------------------------

def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute the Maximum Drawdown (MDD) of an equity curve.

    MDD measures the largest peak-to-trough decline in portfolio value,
    expressed as a negative fraction:

        MDD = min( Vₜ / max(V₀…Vₜ) − 1 )

    An MDD of −0.35 means the portfolio lost 35 % from its all-time high
    at some point during the measurement period.

    Parameters
    ----------
    equity_curve : pd.Series
        Series of portfolio values (e.g. from :func:`portfolio_equity_curve`).
        Must be strictly positive.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal (e.g. −0.35 for −35 %).
    """
    rolling_peak: pd.Series = equity_curve.cummax()
    drawdown: pd.Series = equity_curve / rolling_peak - 1
    return float(drawdown.min())


def rolling_drawdown(equity_curve: pd.Series) -> pd.Series:
    """
    Compute the full rolling drawdown time-series.

    Useful for plotting the drawdown trough over time rather than just
    reporting the single worst point.

    Parameters
    ----------
    equity_curve : pd.Series
        Series of portfolio values.

    Returns
    -------
    pd.Series
        Drawdown at each point in time (non-positive values).
    """
    rolling_peak: pd.Series = equity_curve.cummax()
    return equity_curve / rolling_peak - 1


# ---------------------------------------------------------------------------
# Portfolio Equity Curve
# ---------------------------------------------------------------------------

def portfolio_equity_curve(
    prices: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Compute a weighted portfolio equity curve (starting value = 1.0).

    Steps:

    1. Align ``prices`` columns to ``weights`` index — only tickers present
       in both are used; missing tickers are silently ignored and weights
       are re-normalised.
    2. Compute daily simple returns for each asset.
    3. Compute the daily portfolio return as the weighted average.
    4. Compound into a wealth index starting at 1.0.

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned Adj Close price matrix.
    weights : pd.Series
        Asset weights indexed by ticker symbol (will be re-normalised to sum
        to 1 after alignment).

    Returns
    -------
    pd.Series
        Portfolio equity curve (wealth index), indexed by date.
        First value is 1.0.
    """
    # Align tickers
    common: list[str] = [t for t in weights.index if t in prices.columns]
    if not common:
        raise ValueError("No overlap between weights index and prices columns.")

    w: pd.Series = weights[common]
    w_total: float = w.sum()
    if w_total <= 0:
        raise ValueError("Weights sum to zero after alignment.")
    w = w / w_total  # re-normalise

    daily_returns: pd.DataFrame = prices[common].pct_change().fillna(0.0)
    portfolio_daily: pd.Series = daily_returns @ w
    equity: pd.Series = (1 + portfolio_daily).cumprod()
    equity.name = "Portfolio"
    return equity


# ---------------------------------------------------------------------------
# Rolling Volatility
# ---------------------------------------------------------------------------

def rolling_volatility(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    benchmark_col: str = "SPY",
    window: int = 30,
) -> pd.DataFrame:
    """
    Compute annualised rolling volatility for a weighted portfolio and a benchmark.

    Rolling volatility is the standard deviation of daily log returns over a
    trailing ``window``-day window, annualised by multiplying by sqrt(252).
    It reveals how the portfolio's *realised* risk changes through time —
    spiking during crashes and compressing during low-vol regimes — allowing
    direct comparison with the benchmark to quantify risk reduction.

    Uses log returns (consistent with the optimiser) rather than simple
    returns, giving the correct Gaussian-regime volatility estimate.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log-return matrix.
    weights : pd.Series
        Asset weights indexed by ticker (aligned to ``log_returns`` columns).
    benchmark_col : str
        Column in ``log_returns`` to use as the benchmark (default ``"SPY"``).
    window : int
        Rolling window in trading days (default 30).

    Returns
    -------
    pd.DataFrame
        Columns: ``Portfolio``, ``Benchmark`` (annualised volatility).
        First ``window - 1`` rows are NaN (insufficient history).
    """
    common: list[str] = [t for t in weights.index if t in log_returns.columns]
    w = weights[common]
    w = w / w.sum()

    # Daily portfolio log returns via dot product
    port_daily: pd.Series = log_returns[common] @ w
    port_vol: pd.Series = (
        port_daily.rolling(window=window, min_periods=window).std() * np.sqrt(252)
    )
    port_vol.name = "Portfolio"

    if benchmark_col in log_returns.columns:
        bench_vol: pd.Series = (
            log_returns[benchmark_col]
            .rolling(window=window, min_periods=window)
            .std() * np.sqrt(252)
        )
        bench_vol.name = "Benchmark"
    else:
        bench_vol = pd.Series(np.nan, index=log_returns.index, name="Benchmark")

    return pd.DataFrame({"Portfolio": port_vol, "Benchmark": bench_vol})


# ---------------------------------------------------------------------------
# Standalone verification entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from data_loader import PortfolioData
    from optimizer import max_sharpe_portfolio

    print("Loading data …")
    portfolio = PortfolioData().load()
    prices = portfolio.get_prices()
    log_returns = portfolio.get_returns()

    # --- Bollinger Bands ---
    print("\n--- Bollinger Bands (AAPL, last 5 rows) ---")
    bb = bollinger_bands(prices, "AAPL")
    print(bb.tail(5).round(2).to_string())

    # --- Cumulative Returns ---
    print("\n--- Cumulative Returns (last 3 rows) ---")
    cum = cumulative_returns(prices)
    print(cum.tail(3).round(4).to_string())

    # --- Max Sharpe Equity Curve & MDD ---
    print("\n--- Portfolio Equity Curve (Max Sharpe, last 3 rows) ---")
    result = max_sharpe_portfolio(log_returns)
    equity = portfolio_equity_curve(prices, result.weights)
    print(equity.tail(3).round(4).to_string())

    mdd = max_drawdown(equity)
    print(f"\nMax Drawdown (Max Sharpe portfolio) : {mdd:.4%}")

    # --- SPY MDD for comparison ---
    spy_equity = portfolio_equity_curve(prices, pd.Series({"SPY": 1.0}))
    spy_mdd = max_drawdown(spy_equity)
    print(f"Max Drawdown (SPY benchmark)        : {spy_mdd:.4%}")

    # Sanity checks
    assert bb.shape[1] == 5, "Bollinger Bands should have 5 columns."
    assert cum.shape == prices.shape, "Cumulative returns shape mismatch."
    assert mdd < 0, "MDD must be negative."
    assert abs(equity.iloc[0] - 1.0) < 1e-6, "Equity curve must start at 1.0."
    print("\nAll sanity checks passed.")
