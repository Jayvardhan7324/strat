"""
Monte Carlo Analysis for All Polymarket AI Trading Models

Reads trade CSV output from each strategy and runs bootstrap
resampling to estimate confidence intervals for returns,
drawdowns, Sharpe ratio, and ruin probability.

Models analyzed with individual trade data:
- chop_direction_predictor
- buy1_cent
- chop_scalper_v1
- chop_scalper_v2
- buy97_sell99
- prev10_momentum
- live_guarded_v1

Usage:
    python monte_carlo_analysis.py
    python monte_carlo_analysis.py --n_sims 50000
    python monte_carlo_analysis.py --plot
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

STARTING_CAPITAL = 10_000.0


# ========================================================================
# Monte Carlo Engine
# ========================================================================

def monte_carlo_bootstrap(
    pnls: np.ndarray,
    n_sims: int = 10_000,
    confidence: float = 0.95,
    starting_capital: float = STARTING_CAPITAL,
) -> dict:
    """
    Bootstrap resample a PnL series to build confidence intervals.
    """
    n = len(pnls)
    if n == 0:
        return {"error": "No trades"}

    equity = np.cumsum(pnls) + starting_capital
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    obs_dd = float(np.max(dd))

    obs_return = float(np.sum(pnls) / starting_capital)
    obs_wr = float(np.mean(pnls > 0))
    obs_mean = float(np.mean(pnls))
    obs_std = float(np.std(pnls, ddof=1)) if n > 1 else 1e-9
    obs_sharpe = (obs_mean / obs_std) * np.sqrt(n) if obs_std > 0 else 0.0

    pos_sum = np.sum(pnls[pnls > 0])
    neg_sum = np.sum(pnls[pnls < 0])
    obs_pf = abs(pos_sum / neg_sum) if neg_sum != 0 else float("inf")

    # Tail-risk: average win vs loss
    avg_win = float(np.mean(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    avg_loss = float(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else 0.0
    worst_loss = float(np.min(pnls)) if np.any(pnls < 0) else 0.0

    mc_returns = np.empty(n_sims)
    mc_dds = np.empty(n_sims)
    mc_sharpes = np.empty(n_sims)
    mc_wrs = np.empty(n_sims)
    mc_pfs = np.empty(n_sims)
    mc_ruins = np.empty(n_sims)  # fraction of sims that hit 0 capital

    rng = np.random.default_rng(42)
    for i in range(n_sims):
        sample = rng.choice(pnls, size=n, replace=True)
        mc_returns[i] = np.sum(sample) / starting_capital

        eq = np.cumsum(sample) + starting_capital
        p = np.maximum.accumulate(eq)
        mc_dds[i] = np.max((p - eq) / p)
        mc_ruins[i] = float(np.any(eq <= 0))

        m = np.mean(sample)
        s = np.std(sample, ddof=1) if n > 1 else 1e-9
        mc_sharpes[i] = (m / s) * np.sqrt(n) if s > 0 else 0.0
        mc_wrs[i] = np.mean(sample > 0)

        ps = np.sum(sample[sample > 0])
        ns = np.sum(sample[sample < 0])
        mc_pfs[i] = abs(ps / ns) if ns != 0 else float("inf")

    alpha = (1 - confidence) / 2
    low_p = alpha * 100
    high_p = (1 - alpha) * 100

    def ci(arr):
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0,
                    "p_low": 0.0, "p_high": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "median": float(np.median(arr)),
            "p_low": float(np.percentile(arr, low_p)),
            "p_high": float(np.percentile(arr, high_p)),
        }

    def ci_finite(arr):
        finite = np.array([v for v in arr if np.isfinite(v)])
        if len(finite) == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0,
                    "p_low": 0.0, "p_high": 0.0}
        return {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=1)),
            "median": float(np.median(finite)),
            "p_low": float(np.percentile(finite, low_p)),
            "p_high": float(np.percentile(finite, high_p)),
        }

    return {
        "n_trades": n,
        "n_sims": n_sims,
        "confidence_level": confidence,
        "observed": {
            "total_return_pct": float(obs_return * 100),
            "win_rate": float(obs_wr * 100),
            "pnl_per_trade": float(obs_mean),
            "max_drawdown_pct": float(obs_dd * 100),
            "sharpe": float(obs_sharpe),
            "profit_factor": float(obs_pf) if np.isfinite(obs_pf) else None,
        },
        "monte_carlo": {
            "total_return_pct": ci(mc_returns * 100),
            "max_drawdown_pct": ci(mc_dds * 100),
            "sharpe": ci(mc_sharpes),
            "win_rate": ci(mc_wrs * 100),
            "profit_factor": ci_finite(mc_pfs),
        },
        "prob_return_gt_0pct": float(np.mean(mc_returns > 0)),
        "prob_return_gt_50pct": float(np.mean(mc_returns > 0.5)),
        "prob_return_gt_100pct": float(np.mean(mc_returns > 1.0)),
        "prob_drawdown_gt_20pct": float(np.mean(mc_dds > 0.20)),
        "prob_drawdown_gt_30pct": float(np.mean(mc_dds > 0.30)),
        "prob_ruin": float(np.mean(mc_ruins)),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "worst_loss": worst_loss,
    }


# ========================================================================
# CSV Loading helpers
# ========================================================================

def load_trades_adaptive(csv_path: Path) -> list[dict]:
    """Load trades from CSV, trying common column names for PnL."""
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    # Find pnl column
    pnl_col = None
    for col in ["pnl", "PnL", "profit", "profit_loss", "net_pnl", "trade_pnl"]:
        if col in df.columns:
            pnl_col = col
            break
    if pnl_col is None and len(df.columns) > 0:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                pnl_col = col
                break

    if pnl_col is None:
        return []

    trades = []
    for _, row in df.iterrows():
        try:
            val = float(row[pnl_col])
            trades.append({
                "pnl": val,
                "side": str(row.get("side", "")),
                "outcome": int(row.get("outcome", 0)),
            })
        except Exception:
            continue
    return trades


# ========================================================================
# Strategy mapping - trade-level CSVs only
# ========================================================================

STRATEGY_FILES = {
    "chop_direction_predictor": [
        "chop_dir_trades_all_patterns_test.csv",
        "chop_dir_trades_all_patterns_train.csv",
    ],
    "buy1_cent": [
        "buy1_cent_trades.csv",
    ],
    "chop_scalper_v1": [
        "chop_scalper_trades_test.csv",
        "chop_scalper_trades_train.csv",
    ],
    "chop_scalper_v2": [
        "chop_v2_trades_chop_test.csv",
        "chop_v2_trades_chop_train.csv",
    ],
    "buy97_sell99": [
        "buy97_sell99_trades_test.csv",
        "buy97_sell99_trades_train.csv",
    ],
    "prev10_momentum": [
        "prev10_momentum_next_best_trades.csv",
    ],
    "live_guarded_v1": [
        "live_guarded_metrics_slip_0.00c.csv",
    ],
}


def find_trade_files() -> dict[str, list[Path]]:
    """Find available trade CSV files in the project."""
    found = {}
    for strategy_name, filenames in STRATEGY_FILES.items():
        paths = [Path(fn) for fn in filenames if Path(fn).exists()]
        if paths:
            found[strategy_name] = paths
    return found


def print_ci(obs: dict, mc: dict, ci_type: str, label: str, fmt: str = ".2f"):
    """Pretty-print a confidence interval."""
    ci = mc.get(ci_type, {})
    p_low = ci.get("p_low", None)
    p_high = ci.get("p_high", None)
    mean = ci.get("mean", 0.0)
    if p_low is not None and p_high is not None:
        print(f"    {label:<18} [{p_low:+{fmt}}, {p_high:+{fmt}}]  (mean: {mean:+{fmt}})")
    else:
        print(f"    {label:<18} {mean:+{fmt}}")


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Analysis for Trading Models")
    parser.add_argument("--n_sims", type=int, default=10_000, help="Number of Monte Carlo simulations per model")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level for intervals")
    parser.add_argument("--output", type=str, default="monte_carlo_results.json", help="Output JSON file")
    parser.add_argument("--plot", action="store_true", help="Generate distribution plots (requires matplotlib)")
    args = parser.parse_args()

    print("=" * 70)
    print("MONTE CARLO ANALYSIS FOR ALL AI TRADING MODELS")
    print("=" * 70)
    print(f"Simulations per model: {args.n_sims:,}")
    print(f"Confidence level:      {args.confidence}")
    print()

    strategies = find_trade_files()

    if not strategies:
        print("ERROR: No trade CSV files found. Please run backtests first.")
        return

    results = {}
    all_pnls = {}

    for strategy_name, paths in strategies.items():
        print(f"\n{'='*70}")
        print(f"STRATEGY: {strategy_name}")
        print(f"{'='*70}")

        trades = []
        for p in paths:
            t = load_trades_adaptive(p)
            if t:
                print(f"  Loaded {len(t)} trades from {p.name}")
                trades.extend(t)

        if not trades:
            print(f"  [SKIP] No trades found for {strategy_name}")
            continue

        pnls = np.array([t["pnl"] for t in trades])
        all_pnls[strategy_name] = pnls

        print(f"\n  Running {args.n_sims:,} Monte Carlo simulations on {len(pnls)} trades...")
        result = monte_carlo_bootstrap(pnls, n_sims=args.n_sims, confidence=args.confidence)
        results[strategy_name] = result

        obs = result.get("observed", {})
        mc = result.get("monte_carlo", {})

        print(f"\n  Observed Statistics:")
        print(f"    Total Return:      {obs.get('total_return_pct', 0):+.2f}%")
        print(f"    Win Rate:          {obs.get('win_rate', 0):.1f}%")
        print(f"    Max Drawdown:      {obs.get('max_drawdown_pct', 0):.2f}%")
        print(f"    Sharpe Ratio:      {obs.get('sharpe', 0):.3f}")
        pf = obs.get('profit_factor')
        pf_str = f"{pf:.2f}" if pf is not None else "N/A"
        print(f"    Profit Factor:     {pf_str}")

        print(f"\n  Monte Carlo {args.confidence*100:.0f}% Confidence Intervals:")
        print_ci(obs, mc, "total_return_pct", "Total Return", ".2f")
        print_ci(obs, mc, "max_drawdown_pct", "Max Drawdown", ".2f")
        print_ci(obs, mc, "sharpe", "Sharpe Ratio", ".3f")
        print_ci(obs, mc, "win_rate", "Win Rate", ".1f")

        print(f"\n  Risk-of-Ruin (Tail Risk):")
        print(f"    Avg Win:              ${result.get('avg_win', 0):.2f}")
        print(f"    Avg Loss:             ${result.get('avg_loss', 0):.2f}")
        print(f"    Worst Loss:           ${result.get('worst_loss', 0):.2f}")
        print(f"    Loss / Win Ratio:     {abs(result.get('avg_loss', 0) / result.get('avg_win', 1)):.2f}x")
        print(f"    Prob(Ruin) on $10k:   {result.get('prob_ruin', 0)*100:.1f}%")
        print(f"\n  Standard Risk Estimates:")
        print(f"    Prob(Return > 0):     {result.get('prob_return_gt_0pct', 0)*100:.1f}%")
        print(f"    Prob(Drawdown > 20%): {result.get('prob_drawdown_gt_20pct', 0)*100:.1f}%")
        print(f"    Prob(Drawdown > 30%): {result.get('prob_drawdown_gt_30pct', 0)*100:.1f}%")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to {args.output}")

    # Optional plotting
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle("Monte Carlo: Return & Drawdown Distributions", fontsize=16)

            for name, pnls_item in all_pnls.items():
                sim_returns = []
                sim_dds = []
                rng = np.random.default_rng(42)
                for _ in range(args.n_sims):
                    sample = rng.choice(pnls_item, size=len(pnls_item), replace=True)
                    sim_returns.append(np.sum(sample) / STARTING_CAPITAL * 100)
                    eq = np.cumsum(sample) + STARTING_CAPITAL
                    p = np.maximum.accumulate(eq)
                    dd = np.max((p - eq) / p)
                    sim_dds.append(dd * 100)

                axes[0, 0].hist(sim_returns, bins=50, alpha=0.5, label=name, density=True)
                axes[0, 1].hist(sim_dds, bins=50, alpha=0.5, label=name, density=True)

            axes[0, 0].set_title("Total Return Distribution (%)")
            axes[0, 0].set_xlabel("Return (%)")
            axes[0, 0].axvline(x=0, color="red", linestyle="--")
            axes[0, 0].legend(fontsize=6)

            axes[0, 1].set_title("Max Drawdown Distribution (%)")
            axes[0, 1].set_xlabel("Max Drawdown (%)")

            plt.tight_layout()
            plot_path = "monte_carlo_distributions.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {plot_path}")
        except ImportError:
            print("matplotlib not installed, skipping plots")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
