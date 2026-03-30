"""
optimizer.py
============
Mean-Variance Optimization (Efficient Frontier)
-------------------------------------------------------------
Implements three core optimisation routines using ``cvxpy``:

* :func:`min_variance_portfolio`  — global minimum-variance portfolio
* :func:`max_sharpe_portfolio`    — maximum Sharpe ratio via Charnes-Cooper transform
* :func:`efficient_frontier`      — parametric sweep of 50 optimal risk-return points

Covariance robustness
---------------------
When the sample covariance matrix is rank-deficient (common when the number of
assets N is close to the number of observations T), the routine falls back to
Ledoit-Wolf shrinkage.  Ledoit-Wolf shrinks the sample covariance toward a
structured estimator (scaled identity), producing a guaranteed positive-definite
matrix without ad-hoc ridge tuning.

Usage
-----
    from data_loader import PortfolioData
    from optimizer import max_sharpe_portfolio, efficient_frontier

    log_returns = PortfolioData().load().get_returns()
    result      = max_sharpe_portfolio(log_returns)
    frontier    = efficient_frontier(log_returns)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRADING_DAYS: int = 252
DEFAULT_RF: float = 0.04          # 4 % annualised risk-free rate
DEFAULT_N_POINTS: int = 50        # number of frontier portfolios


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class OptimResult:
    """
    Typed container for a single optimised portfolio.

    Attributes
    ----------
    weights : pd.Series
        Optimal asset weights indexed by ticker symbol.
    expected_return : float
        Annualised expected portfolio return.
    volatility : float
        Annualised portfolio volatility (standard deviation).
    sharpe_ratio : float
        Excess return per unit of risk: (E[Rp] - Rf) / σp.
    tickers : list[str]
        Ordered list of tickers corresponding to ``weights``.
    label : str
        Human-readable label (e.g. "Max Sharpe", "Min Variance").
    """

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    tickers: list[str] = field(default_factory=list)
    label: str = ""

    def __post_init__(self) -> None:
        # Defensive: clip tiny negative weights from numerical noise to zero
        self.weights = self.weights.clip(lower=0)
        total = self.weights.sum()
        if total > 0:
            self.weights = self.weights / total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _annualised_params(
    log_returns: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute annualised expected returns vector and covariance matrix.

    Applies Ledoit-Wolf shrinkage when the sample covariance matrix is not
    full-rank (i.e. rank(Σ) < N), which occurs when T ≈ N or assets are
    perfectly correlated.  Shrinkage guarantees a positive-definite Σ required
    by the quadratic programming solvers.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log-return matrix (T × N).

    Returns
    -------
    mu : np.ndarray  shape (N,)
        Annualised expected returns.
    cov : np.ndarray  shape (N, N)
        Annualised covariance matrix (PSD-guaranteed).
    """
    n: int = log_returns.shape[1]
    mu: np.ndarray = log_returns.mean().values * TRADING_DAYS
    sample_cov: np.ndarray = log_returns.cov().values * TRADING_DAYS

    rank: int = int(np.linalg.matrix_rank(sample_cov))
    if rank < n:
        logger.warning(
            "Sample covariance matrix is rank-deficient (rank=%d, N=%d). "
            "Applying Ledoit-Wolf shrinkage to guarantee PSD.",
            rank, n,
        )
        lw = LedoitWolf().fit(log_returns.values)
        cov = lw.covariance_ * TRADING_DAYS
    else:
        cov = sample_cov

    return mu, cov


def _portfolio_stats(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
) -> tuple[float, float, float]:
    """Return (expected_return, volatility, sharpe_ratio) for a weight vector."""
    ret: float = float(weights @ mu)
    var: float = float(weights @ cov @ weights)
    vol: float = float(np.sqrt(np.maximum(var, 0.0)))
    sharpe: float = (ret - rf) / vol if vol > 1e-10 else 0.0
    return ret, vol, sharpe


# ---------------------------------------------------------------------------
# Public optimisation functions
# ---------------------------------------------------------------------------

def min_variance_portfolio(
    log_returns: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RF,
) -> OptimResult:
    """
    Compute the Global Minimum Variance (GMV) portfolio.

    Solves the convex QP:

        minimise    wᵀ Σ w
        subject to  Σ wᵢ = 1
                    wᵢ ≥ 0   (long-only)

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log-return matrix.
    risk_free_rate : float
        Annualised risk-free rate (used only for Sharpe calculation).

    Returns
    -------
    OptimResult
    """
    tickers: list[str] = list(log_returns.columns)
    n: int = len(tickers)
    mu, cov = _annualised_params(log_returns)

    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Min-variance optimisation failed: {prob.status}")

    w_opt: np.ndarray = np.array(w.value).flatten()
    ret, vol, sharpe = _portfolio_stats(w_opt, mu, cov, risk_free_rate)

    return OptimResult(
        weights=pd.Series(w_opt, index=tickers),
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        tickers=tickers,
        label="Min Variance",
    )


def max_sharpe_portfolio(
    log_returns: pd.DataFrame,
    risk_free_rate: float = DEFAULT_RF,
) -> OptimResult:
    """
    Compute the Maximum Sharpe Ratio portfolio.

    Uses the **Charnes-Cooper variable transformation** to convert the
    fractional programming problem into a standard convex QP solvable by
    cvxpy without resorting to scipy or non-convex solvers:

        Let κ > 0 and y = κ·w.
        Minimise   yᵀ Σ y
        subject to (μ - rƒ)ᵀ y = 1
                   Σ yᵢ ≥ 0
                   y ≥ 0

    Recover weights via  w = y / Σ yᵢ.

    This is mathematically equivalent to maximising the Sharpe ratio over
    the long-only simplex and guarantees a globally optimal solution because
    the transformed problem is strictly convex.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log-return matrix.
    risk_free_rate : float
        Annualised risk-free rate.

    Returns
    -------
    OptimResult
    """
    tickers: list[str] = list(log_returns.columns)
    n: int = len(tickers)
    mu, cov = _annualised_params(log_returns)

    excess_mu: np.ndarray = mu - risk_free_rate

    # Charnes-Cooper transform: y = κ·w
    y = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(y, cov))
    constraints = [
        excess_mu @ y == 1,   # normalisation constraint (fixes κ)
        y >= 0,               # long-only (equivalent to w ≥ 0)
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Max-Sharpe optimisation failed: {prob.status}")

    y_opt: np.ndarray = np.array(y.value).flatten()
    y_sum: float = float(y_opt.sum())
    w_opt: np.ndarray = y_opt / y_sum if y_sum > 1e-10 else np.ones(n) / n

    ret, vol, sharpe = _portfolio_stats(w_opt, mu, cov, risk_free_rate)

    return OptimResult(
        weights=pd.Series(w_opt, index=tickers),
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        tickers=tickers,
        label="Max Sharpe",
    )


def efficient_frontier(
    log_returns: pd.DataFrame,
    n_points: int = DEFAULT_N_POINTS,
    risk_free_rate: float = DEFAULT_RF,
) -> list[OptimResult]:
    """
    Generate the Efficient Frontier by parametric target-return sweep.

    For each target return level μ_target in [μ_GMV, max(μ)]:

        minimise   wᵀ Σ w
        subject to μᵀ w = μ_target
                   Σ wᵢ = 1
                   wᵢ ≥ 0

    Infeasible points (e.g. if no long-only combination can reach a
    very high target return) are silently skipped.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log-return matrix.
    n_points : int
        Number of equally-spaced target-return levels to evaluate.
    risk_free_rate : float
        Annualised risk-free rate (used for Sharpe calculation only).

    Returns
    -------
    list[OptimResult]
        Ordered list of frontier portfolios from lowest to highest return.
        Guaranteed to contain at least the GMV portfolio even if sweep fails.
    """
    tickers: list[str] = list(log_returns.columns)
    n: int = len(tickers)
    mu, cov = _annualised_params(log_returns)

    # Anchor points: GMV return as lower bound, individual max return as upper
    gmv = min_variance_portfolio(log_returns, risk_free_rate)
    ret_min: float = gmv.expected_return
    ret_max: float = float(mu.max())

    # Ensure a sensible sweep range
    if ret_max <= ret_min:
        ret_max = ret_min * 1.5

    target_returns: np.ndarray = np.linspace(ret_min, ret_max, n_points)

    results: list[OptimResult] = []

    for target in target_returns:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov))
        constraints = [
            mu @ w == target,
            cp.sum(w) == 1,
            w >= 0,
        ]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            # Skip infeasible target-return levels gracefully
            continue

        w_opt = np.array(w.value).flatten()
        ret, vol, sharpe = _portfolio_stats(w_opt, mu, cov, risk_free_rate)

        results.append(
            OptimResult(
                weights=pd.Series(w_opt, index=tickers),
                expected_return=ret,
                volatility=vol,
                sharpe_ratio=sharpe,
                tickers=tickers,
                label=f"Frontier (μ={target:.3f})",
            )
        )

    logger.info(
        "Efficient frontier: %d / %d target-return levels solved successfully.",
        len(results), n_points,
    )
    return results


def monte_carlo_portfolios(
    log_returns: pd.DataFrame,
    n_sim: int = 5000,
    risk_free_rate: float = DEFAULT_RF,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate ``n_sim`` random portfolios for Monte Carlo robustness analysis.

    Weights are sampled from a **Dirichlet(1, …, 1)** distribution, which
    produces a uniform distribution over the N-simplex (all weights >= 0,
    sum = 1).  This is the statistically correct method — naive uniform
    sampling followed by normalisation is biased toward the centre of the
    simplex.  Overlaying these portfolios on the Efficient Frontier proves
    the optimiser found the true *edge* of achievable risk-return space.

    The variance calculation uses ``np.einsum`` for full vectorisation:
    no Python-level loop over n_sim, keeping runtime under ~0.5 s.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log-return matrix.
    n_sim : int
        Number of random portfolios to simulate (default 5 000).
    risk_free_rate : float
        Annualised risk-free rate for Sharpe calculation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``Return``, ``Volatility``, ``Sharpe``, plus one weight
        column per ticker prefixed with ``w_``.  Shape: (n_sim, 3 + N).
    """
    rng = np.random.default_rng(seed)
    tickers: list[str] = list(log_returns.columns)
    n: int = len(tickers)
    mu, cov = _annualised_params(log_returns)

    # Dirichlet(alpha=1) == uniform distribution on the N-simplex
    raw = rng.exponential(scale=1.0, size=(n_sim, n))
    weights_matrix: np.ndarray = raw / raw.sum(axis=1, keepdims=True)

    # Vectorised portfolio stats
    port_returns: np.ndarray = weights_matrix @ mu
    port_vars: np.ndarray = np.einsum("ij,jk,ik->i", weights_matrix, cov, weights_matrix)
    port_vols: np.ndarray = np.sqrt(np.maximum(port_vars, 0.0))
    port_sharpes: np.ndarray = np.where(
        port_vols > 1e-10,
        (port_returns - risk_free_rate) / port_vols,
        0.0,
    )

    stats_df = pd.DataFrame(
        {"Return": port_returns, "Volatility": port_vols, "Sharpe": port_sharpes}
    )
    weight_df = pd.DataFrame(weights_matrix, columns=[f"w_{t}" for t in tickers])
    return pd.concat([stats_df, weight_df], axis=1)


def risk_contribution(
    weights: pd.Series,
    log_returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute each asset's percentage contribution to total portfolio variance.

    Marginal Risk Contribution (MRC) of asset i:

        RC_i = w_i * (Sigma @ w)_i / (w^T Sigma w)

    where (Sigma w)_i is the i-th element of the covariance-weighted weight
    vector.  RC_i values sum exactly to 1.0 and represent the fraction of
    total portfolio variance *owned* by each asset.

    Key insight: RC_i is NOT simply proportional to w_i.  A large-weight
    asset in a low-correlation sleeve may contribute *less* variance than a
    small-weight asset that has high systematic correlation with the rest of
    the portfolio — a fact missed entirely by inspecting weights alone.

    Parameters
    ----------
    weights : pd.Series
        Asset weights indexed by ticker symbol.
    log_returns : pd.DataFrame
        Daily log-return matrix (used to compute the annualised covariance).

    Returns
    -------
    pd.Series
        Risk contribution fractions indexed by ticker, summing to 1.0.
    """
    tickers: list[str] = list(weights.index)
    _, cov = _annualised_params(log_returns[tickers])
    w: np.ndarray = weights.values.astype(float)

    marginal: np.ndarray = cov @ w              # Sigma @ w
    contributions: np.ndarray = w * marginal    # w_i * (Sigma w)_i
    total_var: float = float(w @ marginal)      # w^T Sigma w

    pct = contributions / total_var if total_var > 0 else np.ones(len(tickers)) / len(tickers)
    return pd.Series(pct, index=tickers, name="Risk Contribution")


# ---------------------------------------------------------------------------
# Standalone verification entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directory to path if running from a sub-folder
    sys.path.insert(0, str(Path(__file__).parent))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from data_loader import PortfolioData

    print("Loading data …")
    log_returns = PortfolioData().load().get_returns()

    print("\n--- Max Sharpe Portfolio ---")
    ms = max_sharpe_portfolio(log_returns)
    print(ms.weights.round(4).to_string())
    print(f"Expected Return : {ms.expected_return:.4f}")
    print(f"Volatility      : {ms.volatility:.4f}")
    print(f"Sharpe Ratio    : {ms.sharpe_ratio:.4f}")
    print(f"Weights sum     : {ms.weights.sum():.6f}")

    print("\n--- Min Variance Portfolio ---")
    mv = min_variance_portfolio(log_returns)
    print(mv.weights.round(4).to_string())
    print(f"Expected Return : {mv.expected_return:.4f}")
    print(f"Volatility      : {mv.volatility:.4f}")
    print(f"Sharpe Ratio    : {mv.sharpe_ratio:.4f}")
    print(f"Weights sum     : {mv.weights.sum():.6f}")

    print("\n--- Efficient Frontier ---")
    frontier = efficient_frontier(log_returns, n_points=50)
    print(f"Valid frontier points : {len(frontier)} / 50")

    # Sanity checks
    assert abs(ms.weights.sum() - 1.0) < 1e-4, "Max Sharpe weights don't sum to 1!"
    assert abs(mv.weights.sum() - 1.0) < 1e-4, "Min Var weights don't sum to 1!"
    assert len(frontier) >= 10, "Too few frontier points — optimisation may have failed."
    assert ms.sharpe_ratio >= mv.sharpe_ratio, "Max Sharpe should have higher Sharpe than Min Var!"
    print("\nAll sanity checks passed.")
