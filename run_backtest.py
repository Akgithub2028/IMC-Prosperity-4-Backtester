#!/usr/bin/env python3
"""
run_backtest.py — Main entry point for the Python backtester.

Usage:
  python run_backtest.py                          # Run with defaults
  python run_backtest.py --trader my_trader.py    # Specify trader
  python run_backtest.py --dataset datasets/tutorial  # Specify dataset

For Google Colab:
  1. git clone <this repo>
  2. Overwrite solution.py with your Trader class
  3. Run this script
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtester import BacktestResult, MatchingConfig, RunRequest, run_backtest
from runner import (
    resolve_trader,
    resolve_datasets,
    print_summary,
    _short_dataset_label,
)
from visualiser import (
    compute_metrics,
    compute_product_metrics,
    full_analysis,
    print_metrics,
    print_product_table,
    visualise,
    visualise_product_comparison,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Python IMC Prosperity 4 Backtester"
    )
    parser.add_argument("--trader", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--day", type=int, default=None)
    parser.add_argument("--trade-match-mode", type=str, default="all")
    parser.add_argument("--queue-penetration", type=float, default=1.0)
    parser.add_argument("--price-slippage-bps", type=float, default=0.0)
    parser.add_argument("--no-visualise", action="store_true")
    parser.add_argument("--save-charts", type=str, default=None,
                        help="Prefix for saving chart PNGs")

    args = parser.parse_args()

    trader_path = resolve_trader(args.trader, PROJECT_ROOT)
    matching = MatchingConfig(
        trade_match_mode=args.trade_match_mode,
        queue_penetration=args.queue_penetration,
        price_slippage_bps=args.price_slippage_bps,
    )

    # Resolve datasets
    if args.day is not None:
        all_targets = resolve_datasets(args.dataset, PROJECT_ROOT)
        targets = [(p, d) for p, d in all_targets if d == args.day]
        if not targets:
            # Maybe the day doesn't match expanded days, try forcing it
            targets = [(p, args.day) for p, _ in all_targets[:1]]
    else:
        targets = resolve_datasets(args.dataset, PROJECT_ROOT)

    if not targets:
        print("No datasets found.")
        sys.exit(1)

    # Run all backtests
    results: Dict[str, BacktestResult] = {}
    labeled: List[Tuple[str, BacktestResult]] = []

    for dataset_path, day in targets:
        label = _short_dataset_label(dataset_path)
        if day is not None:
            label = f"{label} Day {day}"

        run_id = f"backtest-{int(time.time() * 1000)}"
        request = RunRequest(
            trader_file=trader_path,
            dataset_file=dataset_path,
            day=day,
            matching=matching,
            run_id=run_id,
        )

        print(f"\n{'='*60}")
        print(f"  Running: {label}")
        print(f"  Dataset: {os.path.basename(dataset_path)}")
        print(f"  Day: {day if day is not None else 'all'}")
        print(f"{'='*60}")

        result = run_backtest(request)
        results[label] = result
        labeled.append((label, result))

    # Print summary
    print_summary(labeled, trader_path)

    # Visualise each result
    if not args.no_visualise:
        for label, result in results.items():
            full_analysis(
                result,
                label=label,
                save_prefix=args.save_charts,
                show=True,
            )

        # Product comparison across days
        if len(results) > 1:
            visualise_product_comparison(
                results,
                save_path=f"{args.save_charts}_comparison.png" if args.save_charts else None,
                show=True,
            )


if __name__ == "__main__":
    main()
