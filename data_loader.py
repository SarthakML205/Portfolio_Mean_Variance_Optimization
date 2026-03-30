"""
Usage
-----
    # standalone
    python data_loader.py

    # importable (e.g., Streamlit dashboard)
    from data_loader import PortfolioData

    portfolio = PortfolioData().load()
    returns   = portfolio.get_returns()   # pd.DataFrame of log returns
    prices    = portfolio.get_prices()    # pd.DataFrame of cleaned Adj Close
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Suppress noisy yfinance / pandas FutureWarnings that pollute stdout
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — override via PortfolioData constructor
# ---------------------------------------------------------------------------
DEFAULT_TICKERS: list[str] = [
    # Equities (mega-cap tech + broad market)
    "AAPL",
    "MSFT",
    "NVDA",
    # Broad market ETF
    "SPY",
    # Fixed income — 20+ yr Treasury duration
    "TLT",
    # Commodities — gold
    "GLD",
    # Crypto — late-start assets; handled gracefully
    "BTC-USD",
    "ETH-USD",
]

DEFAULT_PERIOD_YEARS: int = 10
DEFAULT_DATA_DIR: str = "data"


# ---------------------------------------------------------------------------
# PortfolioData
# ---------------------------------------------------------------------------
class PortfolioData:
    """
    Download, clean, and expose multi-asset historical price data.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to download.  Crypto tickers (e.g. ``BTC-USD``) that
        have a shorter history than the requested window are handled
        gracefully — their pre-listing NaN rows are dropped during cleaning
        so they never crash the pipeline.
    period_years : int
        Number of calendar years of history to request.  Actual data may be
        shorter for assets with limited history.
    data_dir : str | Path
        Directory where ``.parquet`` output files are written.

    Attributes (populated after calling ``load()``)
    -------------------------------------------------
    _raw_prices : pd.DataFrame
        Un-synchronised Adj Close prices straight from yfinance.
        Contains NaN rows where a ticker has not yet listed.
        Used only for :meth:`print_summary`.
    prices : pd.DataFrame
        Cleaned, synchronised Adj Close prices on the **common timeline**
        (forward-filled then rows with any NaN dropped).
        — use for Bollinger Bands and cumulative return plots.
    log_returns : pd.DataFrame
        Daily log returns  ln(Pₜ / Pₜ₋₁)  on the common timeline.
        — feed directly into the covariance matrix and the Mean-Variance Optimiser.
    """

    def __init__(
        self,
        tickers: list[str] = DEFAULT_TICKERS,
        period_years: int = DEFAULT_PERIOD_YEARS,
        data_dir: str | Path = DEFAULT_DATA_DIR,
    ) -> None:
        self.tickers: list[str] = tickers
        self.period_years: int = period_years
        self.data_dir: Path = Path(data_dir)

        self.end_date: datetime = datetime.today()
        self.start_date: datetime = self.end_date - timedelta(days=365 * period_years)

        # Populated by load()
        self._raw_prices: pd.DataFrame = pd.DataFrame()
        self.prices: pd.DataFrame = pd.DataFrame()
        self.log_returns: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "PortfolioData":
        """
        Execute the full pipeline: download → clean → compute log returns.

        Returns
        -------
        PortfolioData
            Returns ``self`` to support fluent method chaining, e.g.::

                portfolio = PortfolioData().load()
        """
        logger.info(
            "Downloading %d tickers from %s to %s …",
            len(self.tickers),
            self.start_date.strftime("%Y-%m-%d"),
            self.end_date.strftime("%Y-%m-%d"),
        )
        self._download()
        self._clean()
        logger.info(
            "Pipeline complete. Common timeline: %s → %s  (%d trading days)",
            self.prices.index[0].strftime("%Y-%m-%d"),
            self.prices.index[-1].strftime("%Y-%m-%d"),
            len(self.prices),
        )
        return self

    def get_prices(self) -> pd.DataFrame:
        """
        Return cleaned, synchronised Adjusted Close prices.

        Returns
        -------
        pd.DataFrame
            Shape (trading_days, n_tickers).
            — use for Bollinger Bands and cumulative return plots.
        """
        if self.prices.empty:
            raise RuntimeError("Call load() before accessing prices.")
        return self.prices

    def get_returns(self) -> pd.DataFrame:
        """
        Return daily log returns  ln(Pₜ / Pₜ₋₁).

        Log returns are preferred over simple returns for portfolio optimisation
        because they are time-additive: the multi-period log return equals the
        sum of single-period log returns, making them consistent with the
        Gaussian assumptions in Mean-Variance theory.

        Returns
        -------
        pd.DataFrame
            Shape (trading_days - 1, n_tickers).
            — feed directly into the covariance matrix and the Mean-Variance Optimiser.
        """
        if self.log_returns.empty:
            raise RuntimeError("Call load() before accessing log_returns.")
        return self.log_returns

    def print_summary(self) -> None:
        """
        Print a per-ticker summary table to stdout.

        Columns: Ticker | Start Date | End Date | Total Data Points

        Uses ``_raw_prices`` (pre-sync) so that each ticker shows its own
        actual start date, demonstrating the graceful late-start handling
        for crypto assets.
        """
        if self._raw_prices.empty:
            raise RuntimeError("Call load() before printing summary.")

        _SEP = "-" * 62
        _HDR = f"{'Ticker':<12}{'Start Date':<15}{'End Date':<15}{'Data Points':>12}"

        print("\n" + _SEP)
        print("  PORTFOLIO DATA SUMMARY")
        print(_SEP)
        print(_HDR)
        print(_SEP)

        for ticker in self._raw_prices.columns:
            col = self._raw_prices[ticker]
            valid = col.dropna()
            if valid.empty:
                start_str = end_str = "N/A"
                n_points = 0
            else:
                start_str = valid.index[0].strftime("%Y-%m-%d")
                end_str = valid.index[-1].strftime("%Y-%m-%d")
                n_points = len(valid)

            print(f"  {ticker:<10}  {start_str:<13}  {end_str:<13}  {n_points:>10,}")

        print(_SEP)
        print(
            f"  Common timeline after sync: "
            f"{self.prices.index[0].strftime('%Y-%m-%d')} → "
            f"{self.prices.index[-1].strftime('%Y-%m-%d')}  "
            f"({len(self.prices):,} trading days)"
        )
        print(_SEP + "\n")

    def save_parquet(self, subdir: str | Path | None = None) -> None:
        """
        Persist cleaned prices and log returns as ``.parquet`` files.

        Why Parquet over CSV for AI Engineering
        ----------------------------------------
        1. **Schema preservation** — column dtypes (float64, DatetimeTZDtype,
           CategoricalDtype, …) are stored in the file metadata, eliminating
           silent date→string or float→object conversions that plague CSV round-trips.
        2. **Columnar compression** — Snappy/Zstd compression applied per column
           typically yields 3–10× smaller files than CSV for financial time-series.
        3. **Ecosystem integration** — Parquet is the native interchange format
           for pandas, polars, Apache Arrow, DuckDB, and Spark; zero-copy reads
           with PyArrow make it dramatically faster than CSV parsing in ML pipelines.
        4. **Partial reads** — columnar layout allows reading only the requested
           tickers without loading the full file — critical at scale.

        Parameters
        ----------
        subdir : str | Path | None
            Output directory.  Defaults to ``self.data_dir`` if ``None``.
        """
        if self.prices.empty or self.log_returns.empty:
            raise RuntimeError("Call load() before saving data.")

        out_dir = Path(subdir) if subdir is not None else self.data_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        prices_path = out_dir / "prices.parquet"
        returns_path = out_dir / "log_returns.parquet"

        self.prices.to_parquet(prices_path)
        self.log_returns.to_parquet(returns_path)

        logger.info("Saved → %s", prices_path.resolve())
        logger.info("Saved → %s", returns_path.resolve())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download(self) -> None:
        """
        Download Adj Close prices for all tickers via yfinance.

        ``auto_adjust=False`` is set explicitly so that the ``Adj Close``
        column is always available regardless of the yfinance version default.
        A single multi-ticker call is used for efficiency; the resulting
        MultiIndex columns are sliced to ``["Adj Close"]``.

        Crypto assets (BTC-USD, ETH-USD) only have data from their respective
        listing dates.  Rows before those dates remain as ``NaN`` in
        ``_raw_prices``; this is intentional and handled in :meth:`_clean`.
        """
        raw: pd.DataFrame = yf.download(
            tickers=self.tickers,
            start=self.start_date.strftime("%Y-%m-%d"),
            end=self.end_date.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        # yfinance returns MultiIndex columns (Price, Ticker) for multi-ticker
        # downloads.  Extract the "Adj Close" layer.
        if isinstance(raw.columns, pd.MultiIndex):
            adj_close: pd.DataFrame = raw["Adj Close"].copy()
        else:
            # Single-ticker edge case (not expected given DEFAULT_TICKERS)
            adj_close = raw[["Adj Close"]].copy()
            adj_close.columns = self.tickers

        # Ensure column order matches the tickers list for deterministic output
        adj_close = adj_close[[t for t in self.tickers if t in adj_close.columns]]

        self._raw_prices = adj_close

    def _clean(self) -> None:
        """
        Produce a synchronised, NaN-free price matrix and compute log returns.

        Steps
        -----
        1. **Forward-fill** (``ffill``): propagates the last known price into
           any stale/holiday gaps within a ticker's trading history.
        2. **Drop remaining NaN rows** (``dropna``): removes dates where at
           least one ticker has no data yet (i.e. pre-listing rows for crypto).
           This produces the **common timeline** required for the covariance matrix.
        3. **Log returns**: ``ln(Pₜ / Pₜ₋₁)`` computed via ``np.log``.
           The first row (all NaN after shifting) is dropped with ``.dropna()``.
        """
        cleaned: pd.DataFrame = (
            self._raw_prices
            .ffill()          # Step 1 — fill intra-series gaps
            .dropna()         # Step 2 — align to common timeline
        )

        self.prices = cleaned

        # Step 3 — log returns: ln(Pt / Pt-1)
        self.log_returns = (
            np.log(self.prices / self.prices.shift(1))
            .dropna()
        )


# ---------------------------------------------------------------------------
# Standalone entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    portfolio = (
        PortfolioData(
            tickers=DEFAULT_TICKERS,
            period_years=DEFAULT_PERIOD_YEARS,
            data_dir=DEFAULT_DATA_DIR,
        )
        .load()
    )

    portfolio.print_summary()
    portfolio.save_parquet()

    # Quick sanity assertions
    assert portfolio.log_returns.isna().sum().sum() == 0, (
        "log_returns contains NaN — cleaning pipeline failed."
    )
    assert portfolio.prices.isna().sum().sum() == 0, (
        "prices contains NaN — cleaning pipeline failed."
    )
    logger.info(
        "Sanity checks passed. prices shape=%s  log_returns shape=%s",
        portfolio.prices.shape,
        portfolio.log_returns.shape,
    )
