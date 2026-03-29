"""
Visualiser for the Python IMC Prosperity 4 backtester.

Produces:
  1. PnL chart (total + per-product)
  2. Max drawdown chart
  3. Mid-price + inventory overlay (dual axis)
  4. Return distribution histogram
  5. Full metrics table

Designed for both terminal and Google Colab (inline matplotlib).
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.gridspec import GridSpec

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from backtester import BacktestResult, SHORT_PRODUCT_LABELS


def _save_and_show(fig, save_path: str, label: str = "") -> None:
    """Save chart to file, then show inline (if interactive) or print path."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"  [SAVED] {save_path}")
    plt.show()  # Shows inline in Colab/Jupyter (%matplotlib inline), no-op in Agg
    plt.close(fig)



# ─── Metrics Computation ─────────────────────────────────────────────────────


def compute_metrics(result: BacktestResult) -> Dict[str, Any]:
    """Compute all backtest performance metrics from a BacktestResult."""

    pnl_values = [entry["total"] for entry in result.pnl_series]
    timestamps = [entry["timestamp"] for entry in result.pnl_series]

    if len(pnl_values) < 2:
        return _empty_metrics(result)

    # ── Drawdown calculations ──
    cummax = []
    running_max = -math.inf
    for v in pnl_values:
        running_max = max(running_max, v)
        cummax.append(running_max)

    drawdowns = [pnl_values[i] - cummax[i] for i in range(len(pnl_values))]
    max_drawdown = min(drawdowns) if drawdowns else 0.0

    # Max % drawdown
    pct_drawdowns = []
    for i in range(len(pnl_values)):
        if cummax[i] != 0:
            pct_drawdowns.append(drawdowns[i] / abs(cummax[i]) * 100.0)
        else:
            pct_drawdowns.append(0.0)
    max_pct_drawdown = min(pct_drawdowns) if pct_drawdowns else 0.0

    # Average % drawdown (only during drawdown periods)
    dd_periods = [d for d in pct_drawdowns if d < 0]
    avg_pct_drawdown = sum(dd_periods) / len(dd_periods) if dd_periods else 0.0

    # ── Returns ──
    returns = []
    for i in range(1, len(pnl_values)):
        returns.append(pnl_values[i] - pnl_values[i - 1])
    returns_arr = np.array(returns) if returns else np.array([0.0])

    # ── Sharpe ratio ──
    # Assuming 100ms ticks, ~10000 ticks per day, annualized over ~252 trading days
    ticks_per_day = 10000
    annualization_factor = math.sqrt(ticks_per_day * 252)
    mean_return = float(np.mean(returns_arr))
    std_return = float(np.std(returns_arr, ddof=1)) if len(returns_arr) > 1 else 1e-10
    sharpe_ratio = (mean_return / std_return * annualization_factor) if std_return > 1e-10 else 0.0

    # ── Calmar ratio ──
    total_return = pnl_values[-1] - pnl_values[0]
    calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # ── Trade statistics ──
    total_trades = result.metrics.own_trade_count
    fills = [t.quantity for t in result.own_trades_all]
    avg_fill = sum(fills) / len(fills) if fills else 0.0

    # ── Recovery ──
    # Time from max-drawdown trough to recovery (new high)
    if max_drawdown < 0:
        trough_idx = drawdowns.index(max_drawdown)
        recovery_idx = None
        for i in range(trough_idx + 1, len(pnl_values)):
            if pnl_values[i] >= cummax[trough_idx]:
                recovery_idx = i
                break
        if recovery_idx is not None:
            recovery_ticks = recovery_idx - trough_idx
            recovery_str = f"{recovery_ticks} ticks"
        else:
            recovery_str = "Not recovered"
    else:
        recovery_str = "No drawdown"

    return {
        "max_pct_drawdown": max_pct_drawdown,
        "max_drawdown": max_drawdown,
        "avg_pct_drawdown": avg_pct_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": calmar_ratio,
        "total_trades": total_trades,
        "avg_fill": avg_fill,
        "recovery": recovery_str,
        "final_pnl": pnl_values[-1],
        "tick_count": result.metrics.tick_count,
        "drawdowns": drawdowns,
        "pct_drawdowns": pct_drawdowns,
        "returns": returns,
        "pnl_values": pnl_values,
        "timestamps": timestamps,
        "cummax": cummax,
    }


def _empty_metrics(result: BacktestResult) -> Dict[str, Any]:
    return {
        "max_pct_drawdown": 0.0,
        "max_drawdown": 0.0,
        "avg_pct_drawdown": 0.0,
        "sharpe_ratio": 0.0,
        "calmar_ratio": 0.0,
        "total_trades": result.metrics.own_trade_count,
        "avg_fill": 0.0,
        "recovery": "N/A",
        "final_pnl": 0.0,
        "tick_count": result.metrics.tick_count,
        "drawdowns": [],
        "pct_drawdowns": [],
        "returns": [],
        "pnl_values": [],
        "timestamps": [],
        "cummax": [],
    }


# ─── Product-wise Metrics ────────────────────────────────────────────────────


def compute_product_metrics(
    result: BacktestResult,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-product PnL metrics."""
    product_metrics: Dict[str, Dict[str, Any]] = {}

    for product in result.products:
        pnl_vals = [entry.get(product, 0.0) for entry in result.pnl_series]
        if len(pnl_vals) < 2:
            product_metrics[product] = {
                "final_pnl": pnl_vals[-1] if pnl_vals else 0.0,
                "max_drawdown": 0.0,
                "trades": 0,
            }
            continue

        cummax = []
        running_max = -math.inf
        for v in pnl_vals:
            running_max = max(running_max, v)
            cummax.append(running_max)
        drawdowns = [pnl_vals[i] - cummax[i] for i in range(len(pnl_vals))]
        max_dd = min(drawdowns) if drawdowns else 0.0

        trade_count = sum(
            1 for t in result.own_trades_all if t.symbol == product
        )

        product_metrics[product] = {
            "final_pnl": pnl_vals[-1],
            "max_drawdown": max_dd,
            "trades": trade_count,
        }

    return product_metrics


# ─── Metrics Printing ────────────────────────────────────────────────────────


def print_metrics(metrics: Dict[str, Any], label: str = "") -> None:
    """Print formatted metrics to stdout."""
    header = f"═══ METRICS{f' — {label}' if label else ''} ═══"
    print(f"\n{'═' * len(header)}")
    print(header)
    print(f"{'═' * len(header)}")
    print(f"  Final PnL:          {metrics['final_pnl']:>12.2f}")
    print(f"  Tick Count:         {metrics['tick_count']:>12d}")
    print(f"  Total Trades:       {metrics['total_trades']:>12d}")
    print(f"  Avg Fill:           {metrics['avg_fill']:>12.2f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown']:>12.2f}")
    print(f"  Max % Drawdown:     {metrics['max_pct_drawdown']:>11.2f}%")
    print(f"  Avg % Drawdown:     {metrics['avg_pct_drawdown']:>11.2f}%")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>12.4f}")
    print(f"  Calmar Ratio:       {metrics['calmar_ratio']:>12.4f}")
    print(f"  Recovery:           {metrics['recovery']:>12s}")
    print(f"{'═' * len(header)}")


def print_product_table(
    product_metrics: Dict[str, Dict[str, Any]], label: str = ""
) -> None:
    """Print per-product PnL table."""
    header = f"─── Product Analysis{f' — {label}' if label else ''} ───"
    print(f"\n{header}")
    print(f"  {'PRODUCT':<25s} {'FINAL_PNL':>12s} {'MAX_DD':>12s} {'TRADES':>8s}")
    for product, pm in product_metrics.items():
        short = SHORT_PRODUCT_LABELS.get(product, product)
        print(
            f"  {short:<25s} {pm['final_pnl']:>12.2f} "
            f"{pm['max_drawdown']:>12.2f} {pm['trades']:>8d}"
        )
    print(f"{'─' * len(header)}")


# ─── Visualisations ──────────────────────────────────────────────────────────


def visualise(
    result: BacktestResult,
    label: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Generate all visualisations for a backtest result:
      1. PnL timeseries (total + per-product)
      2. Drawdown chart
      3. Mid-price + Inventory overlay
      4. Return distribution
    """
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not installed — skipping visualisations.")
        print("       Install with: pip install matplotlib numpy")
        return

    metrics = compute_metrics(result)
    product_metrics = compute_product_metrics(result)

    if not metrics["pnl_values"]:
        print("[WARN] No PnL data to visualise.")
        return

    # ── Style setup ──
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0d1117")

    title = f"Backtest Analysis{f' — {label}' if label else ''}"
    fig.suptitle(title, fontsize=18, fontweight="bold", color="#58a6ff", y=0.98)

    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    timestamps = metrics["timestamps"]
    pnl_values = metrics["pnl_values"]

    # Color palette
    colors = [
        "#58a6ff", "#3fb950", "#f0883e", "#d2a8ff",
        "#ff7b72", "#79c0ff", "#7ee787", "#ffa657",
    ]

    # ═══ Panel 1: PnL Chart ═══
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161b22")
    ax1.plot(
        timestamps, pnl_values, color="#58a6ff", linewidth=1.8,
        label="Total PnL", zorder=5,
    )
    ax1.fill_between(
        timestamps, 0, pnl_values, alpha=0.15, color="#58a6ff",
    )
    # Per-product lines
    for i, product in enumerate(result.products):
        prod_pnl = [entry.get(product, 0.0) for entry in result.pnl_series]
        color = colors[(i + 1) % len(colors)]
        short = SHORT_PRODUCT_LABELS.get(product, product)
        ax1.plot(
            timestamps, prod_pnl, color=color, linewidth=0.9,
            alpha=0.7, label=short,
        )
    ax1.axhline(y=0, color="#484f58", linewidth=0.5, linestyle="--")
    ax1.set_title("Profit & Loss", fontsize=13, color="#c9d1d9", pad=10)
    ax1.set_xlabel("Timestamp", fontsize=10, color="#8b949e")
    ax1.set_ylabel("PnL", fontsize=10, color="#8b949e")
    ax1.legend(fontsize=7, loc="upper left", framealpha=0.3)
    ax1.grid(True, alpha=0.1, color="#30363d")
    ax1.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color("#30363d")

    # ═══ Panel 2: Drawdown Chart ═══
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22")
    drawdowns = metrics["drawdowns"]
    ax2.fill_between(
        timestamps, 0, drawdowns, color="#f85149", alpha=0.4,
    )
    ax2.plot(
        timestamps, drawdowns, color="#f85149", linewidth=1.2,
    )
    ax2.axhline(y=0, color="#484f58", linewidth=0.5, linestyle="--")
    max_dd = metrics["max_drawdown"]
    if max_dd < 0:
        dd_idx = drawdowns.index(max_dd)
        ax2.annotate(
            f"Max DD: {max_dd:.2f}",
            xy=(timestamps[dd_idx], max_dd),
            xytext=(timestamps[dd_idx], max_dd * 0.6),
            fontsize=9, color="#f85149", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#f85149", lw=1.2),
        )
    ax2.set_title("Drawdown", fontsize=13, color="#c9d1d9", pad=10)
    ax2.set_xlabel("Timestamp", fontsize=10, color="#8b949e")
    ax2.set_ylabel("Drawdown", fontsize=10, color="#8b949e")
    ax2.grid(True, alpha=0.1, color="#30363d")
    ax2.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color("#30363d")

    # ═══ Panel 3: Mid-Price + Inventory Overlay ═══
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor("#161b22")

    # Plot mid-prices (left axis)
    plotted_any_mid = False
    for i, product in enumerate(result.products):
        mid_vals = [entry.get(product) for entry in result.mid_price_series]
        ts = [entry["timestamp"] for entry in result.mid_price_series]
        # Filter out None values
        valid = [(t, m) for t, m in zip(ts, mid_vals) if m is not None]
        if valid:
            t_valid, m_valid = zip(*valid)
            short = SHORT_PRODUCT_LABELS.get(product, product)
            color = colors[i % len(colors)]
            ax3.plot(
                t_valid, m_valid, color=color, linewidth=1.0,
                alpha=0.8, label=f"{short} mid",
            )
            plotted_any_mid = True

    ax3.set_ylabel("Mid Price", fontsize=10, color="#8b949e")
    ax3.tick_params(axis="y", colors="#8b949e", labelsize=8)

    # Inventory (right axis)
    ax3_inv = ax3.twinx()
    for i, product in enumerate(result.products):
        inv_vals = [entry.get(product, 0) for entry in result.position_series]
        ts = [entry["timestamp"] for entry in result.position_series]
        short = SHORT_PRODUCT_LABELS.get(product, product)
        color = colors[i % len(colors)]
        ax3_inv.plot(
            ts, inv_vals, color=color, linewidth=0.8,
            alpha=0.5, linestyle="--", label=f"{short} inv",
        )
    ax3_inv.set_ylabel("Inventory", fontsize=10, color="#8b949e")
    ax3_inv.tick_params(axis="y", colors="#8b949e", labelsize=8)

    ax3.set_title(
        "Mid-Price & Inventory Over Time",
        fontsize=13, color="#c9d1d9", pad=10,
    )
    ax3.set_xlabel("Timestamp", fontsize=10, color="#8b949e")
    ax3.grid(True, alpha=0.1, color="#30363d")

    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_inv.get_legend_handles_labels()
    ax3.legend(
        lines1 + lines2, labels1 + labels2,
        fontsize=7, loc="upper left", framealpha=0.3, ncol=4,
    )
    ax3.tick_params(axis="x", colors="#8b949e", labelsize=8)
    for spine in ax3.spines.values():
        spine.set_color("#30363d")
    for spine in ax3_inv.spines.values():
        spine.set_color("#30363d")

    # ═══ Panel 4: Return Distribution ═══
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#161b22")
    returns = metrics["returns"]
    if returns:
        ax4.hist(
            returns, bins=min(80, max(20, len(returns) // 20)),
            color="#58a6ff", alpha=0.7, edgecolor="#1f6feb",
        )
        mean_ret = np.mean(returns)
        ax4.axvline(mean_ret, color="#3fb950", linewidth=1.5, linestyle="--",
                     label=f"Mean: {mean_ret:.4f}")
    ax4.set_title("Return Distribution", fontsize=13, color="#c9d1d9", pad=10)
    ax4.set_xlabel("Return (tick-to-tick PnL change)", fontsize=10, color="#8b949e")
    ax4.set_ylabel("Frequency", fontsize=10, color="#8b949e")
    ax4.legend(fontsize=9, framealpha=0.3)
    ax4.grid(True, alpha=0.1, color="#30363d")
    ax4.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax4.spines.values():
        spine.set_color("#30363d")

    # ═══ Panel 5: Metrics Summary Table ═══
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#161b22")
    ax5.axis("off")

    table_data = [
        ["Metric", "Value"],
        ["Final PnL", f"{metrics['final_pnl']:.2f}"],
        ["Tick Count", f"{metrics['tick_count']:,}"],
        ["Total Trades", f"{metrics['total_trades']:,}"],
        ["Avg Fill", f"{metrics['avg_fill']:.2f}"],
        ["Max Drawdown", f"{metrics['max_drawdown']:.2f}"],
        ["Max % Drawdown", f"{metrics['max_pct_drawdown']:.2f}%"],
        ["Avg % Drawdown", f"{metrics['avg_pct_drawdown']:.2f}%"],
        ["Sharpe Ratio", f"{metrics['sharpe_ratio']:.4f}"],
        ["Calmar Ratio", f"{metrics['calmar_ratio']:.4f}"],
        ["Recovery", metrics["recovery"]],
    ]

    table = ax5.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        colWidths=[0.5, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Style the table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#30363d")
        if row == 0:
            cell.set_facecolor("#1f6feb")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#0d1117" if row % 2 == 0 else "#161b22")
            cell.set_text_props(color="#c9d1d9")

    ax5.set_title("Performance Metrics", fontsize=13, color="#c9d1d9", pad=10)

    # Always save to file
    if not save_path:
        os.makedirs("runs", exist_ok=True)
        safe_label = label.replace(" ", "_").replace("/", "-") if label else "backtest"
        save_path = os.path.join("runs", f"chart_{safe_label}.png")

    _save_and_show(fig, save_path, label)


def visualise_product_comparison(
    results: Dict[str, BacktestResult],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualise product-wise PnL comparison across multiple runs (e.g. Day -1 vs Day -2).
    """
    if not HAS_MATPLOTLIB:
        print("[WARN] matplotlib not installed — skipping visualisations.")
        return

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 6))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        "Product-wise PnL Comparison",
        fontsize=16, fontweight="bold", color="#58a6ff", y=1.02,
    )

    if len(results) == 1:
        axes = [axes]

    colors = [
        "#58a6ff", "#3fb950", "#f0883e", "#d2a8ff",
        "#ff7b72", "#79c0ff", "#7ee787", "#ffa657",
    ]

    for idx, (run_label, result) in enumerate(results.items()):
        ax = axes[idx]
        ax.set_facecolor("#161b22")
        pm = compute_product_metrics(result)

        products = list(pm.keys())
        short_labels = [SHORT_PRODUCT_LABELS.get(p, p) for p in products]
        pnl_vals = [pm[p]["final_pnl"] for p in products]
        bar_colors = [
            "#3fb950" if v >= 0 else "#f85149" for v in pnl_vals
        ]

        bars = ax.barh(short_labels, pnl_vals, color=bar_colors, alpha=0.8, height=0.6)
        ax.axvline(0, color="#484f58", linewidth=0.5, linestyle="--")
        ax.set_title(run_label, fontsize=13, color="#c9d1d9", pad=10)
        ax.set_xlabel("PnL", fontsize=10, color="#8b949e")
        ax.tick_params(colors="#8b949e", labelsize=9)
        ax.grid(True, alpha=0.1, color="#30363d", axis="x")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

        # Add value labels
        for bar, val in zip(bars, pnl_vals):
            ax.text(
                bar.get_width() + (abs(max(pnl_vals, default=1)) * 0.02 if val >= 0 else -abs(max(pnl_vals, default=1)) * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
                color="#c9d1d9",
            )

    plt.tight_layout()

    # Always save to file
    if not save_path:
        os.makedirs("runs", exist_ok=True)
        save_path = os.path.join("runs", "chart_product_comparison.png")

    _save_and_show(fig, save_path, "comparison")


# ─── All-in-one Analysis ─────────────────────────────────────────────────────


def full_analysis(
    result: BacktestResult,
    label: str = "",
    save_prefix: Optional[str] = None,
    show: bool = True,
) -> Dict[str, Any]:
    """Run full analysis: print metrics + generate all charts."""
    metrics = compute_metrics(result)
    product_metrics = compute_product_metrics(result)

    print_metrics(metrics, label)
    print_product_table(product_metrics, label)

    safe_label = label.replace(" ", "_").replace("/", "-") if label else "backtest"
    if save_prefix:
        save_path = f"{save_prefix}_{safe_label}.png"
    else:
        os.makedirs("runs", exist_ok=True)
        save_path = os.path.join("runs", f"chart_{safe_label}.png")

    visualise(result, label=label, save_path=save_path, show=show)

    return metrics


# ─── Log Saving ──────────────────────────────────────────────────────────────


def save_run_log(result: BacktestResult, label: str = "") -> str:
    """Save full backtest log (metrics + PnL series + trades) to a JSON file."""
    os.makedirs("runs", exist_ok=True)
    safe_label = label.replace(" ", "_").replace("/", "-") if label else "backtest"
    log_path = os.path.join("runs", f"log_{safe_label}.json")

    metrics = compute_metrics(result)
    product_metrics = compute_product_metrics(result)

    log_data = {
        "label": label,
        "final_pnl": metrics["final_pnl"],
        "tick_count": metrics["tick_count"],
        "total_trades": metrics["total_trades"],
        "avg_fill": metrics["avg_fill"],
        "max_drawdown": metrics["max_drawdown"],
        "max_pct_drawdown": metrics["max_pct_drawdown"],
        "avg_pct_drawdown": metrics["avg_pct_drawdown"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "calmar_ratio": metrics["calmar_ratio"],
        "recovery": metrics["recovery"],
        "pnl_by_product": {
            k: v for k, v in result.metrics.final_pnl_by_product.items()
        },
        "product_metrics": product_metrics,
        "pnl_series": result.pnl_series,
        "position_series": result.position_series,
        "own_trades": [
            {
                "symbol": t.symbol,
                "price": t.price,
                "quantity": t.quantity,
                "buyer": t.buyer,
                "seller": t.seller,
                "timestamp": t.timestamp,
            }
            for t in result.own_trades_all
        ],
    }

    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"  [LOG] {log_path}")
    return log_path
