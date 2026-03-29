"""
CLI runner — port of cli.rs.

Resolves trader files, dataset files, builds run plans, prints results.
"""

from __future__ import annotations

import os
import sys
import glob
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backtester import (
    BacktestResult,
    MatchingConfig,
    RunRequest,
    SHORT_PRODUCT_LABELS,
    load_dataset,
    run_backtest,
)


def resolve_trader(trader_arg: Optional[str] = None, project_root: str = ".") -> str:
    """Find the trader file."""
    if trader_arg:
        return os.path.abspath(trader_arg)

    # Framework files that should never be picked as traders
    _EXCLUDE_FILES = {
        "runner.py", "backtester.py", "visualiser.py", "run_backtest.py",
        "colab_setup.py", "datamodel.py", "__init__.py",
    }

    # Check for solution.py first (preferred trader location)
    solution_path = os.path.join(project_root, "solution.py")
    if os.path.isfile(solution_path):
        with open(solution_path, "r") as fh:
            if "class Trader" in fh.read():
                return os.path.abspath(solution_path)

    for search_dir in ["scripts", "traders/submissions", "traders", "."]:
        search_path = os.path.join(project_root, search_dir)
        if not os.path.isdir(search_path):
            continue
        py_files = glob.glob(os.path.join(search_path, "**/*.py"), recursive=True)
        for f in py_files:
            basename = os.path.basename(f)
            if basename in _EXCLUDE_FILES:
                continue
            try:
                with open(f, "r") as fh:
                    if "class Trader" in fh.read():
                        return os.path.abspath(f)
            except Exception:
                continue

    # Fallback: return solution.py even if it doesn't have class Trader yet
    if os.path.isfile(solution_path):
        return os.path.abspath(solution_path)

    raise FileNotFoundError(
        "No trader file found. Pass --trader <file.py> or place a Trader class "
        "in solution.py or scripts/"
    )


def resolve_datasets(
    dataset_arg: Optional[str] = None, project_root: str = "."
) -> List[Tuple[str, Optional[int]]]:
    """
    Resolve dataset files and return list of (file_path, day) tuples.
    Each day in a dataset is a separate entry.
    """
    datasets_root = os.path.join(project_root, "datasets")

    if dataset_arg:
        path = dataset_arg
        if os.path.isfile(path):
            return _expand_dataset_days(path)
        if os.path.isdir(path):
            return _collect_from_dir(path)
        # Try as alias
        alias_path = os.path.join(datasets_root, path)
        if os.path.isdir(alias_path):
            return _collect_from_dir(alias_path)
        raise FileNotFoundError(f"Dataset not found: {dataset_arg}")

    # Default: latest round
    if os.path.isdir(datasets_root):
        candidates = ["tutorial"] + [f"round{i}" for i in range(1, 9)]
        for name in reversed(candidates):
            round_dir = os.path.join(datasets_root, name)
            if os.path.isdir(round_dir) and _has_datasets(round_dir):
                return _collect_from_dir(round_dir)

    raise FileNotFoundError(
        f"No datasets found. Place CSV/JSON files in {datasets_root}/tutorial/ "
        "or pass --dataset <path>"
    )


def _has_datasets(directory: str) -> bool:
    for f in os.listdir(directory):
        if f.endswith((".csv", ".json", ".log")):
            return True
    return False


def _collect_from_dir(directory: str) -> List[Tuple[str, Optional[int]]]:
    results: List[Tuple[str, Optional[int]]] = []
    files = sorted(os.listdir(directory))

    # Collect CSV day files and JSON/log submission files
    seen_keys = set()
    for f in files:
        full = os.path.join(directory, f)
        if not os.path.isfile(full):
            continue
        if f.startswith("trades_"):
            continue  # paired automatically
        if f.endswith((".csv", ".json", ".log")):
            key = _dataset_key(f)
            if key and key not in seen_keys:
                seen_keys.add(key)
                results.extend(_expand_dataset_days(full))

    if not results:
        raise FileNotFoundError(f"No supported datasets in {directory}")
    return results


def _dataset_key(filename: str) -> Optional[str]:
    lower = filename.lower()
    if lower.startswith("trades_"):
        return None
    if lower.startswith("prices_"):
        return os.path.splitext(lower)[0]
    if lower.endswith((".json", ".log")):
        return os.path.splitext(lower)[0]
    return None


def _expand_dataset_days(path: str) -> List[Tuple[str, Optional[int]]]:
    """Load dataset and return a separate entry for each day."""
    try:
        dataset = load_dataset(path)
    except Exception as e:
        print(f"[WARN] Cannot load {path}: {e}")
        return []

    days = sorted(set(t.day for t in dataset.ticks if t.day is not None))
    if not days:
        return [(path, None)]
    return [(path, day) for day in days]


def _short_dataset_label(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0].lower()
    if "day_-1" in stem:
        return "D-1"
    if "day_-2" in stem:
        return "D-2"
    if "submission" in stem or (stem.isdigit()):
        return "SUB"
    return stem[:20].replace("_", "-").upper()


def print_summary(
    results: List[Tuple[str, BacktestResult]],
    trader_path: str,
) -> None:
    """Print a formatted summary table matching Rust backtester output."""
    print(f"\ntrader: {os.path.basename(trader_path)}")
    print(f"mode: fast")
    print(
        f"{'SET':<12s} {'DAY':>6s} {'TICKS':>8s} "
        f"{'OWN_TRADES':>11s} {'FINAL_PNL':>12s}"
    )

    for label, result in results:
        m = result.metrics
        day_str = str(m.day) if m.day is not None else "all"
        print(
            f"{label:<12s} {day_str:>6s} {m.tick_count:>8d} "
            f"{m.own_trade_count:>11d} {m.final_pnl_total:>12.2f}"
        )

    # Product breakdown
    if results:
        all_products = set()
        for _, r in results:
            all_products.update(r.products)
        all_products = sorted(all_products)

        if all_products:
            product_width = max(
                len(SHORT_PRODUCT_LABELS.get(p, p)) for p in all_products
            )
            product_width = max(product_width, 7)
            col_width = max(10, max(len(l) for l, _ in results))

            print()
            header = f"{'PRODUCT':<{product_width}}"
            for label, _ in results:
                header += f" {label:>{col_width}}"
            print(header)

            for product in all_products:
                short = SHORT_PRODUCT_LABELS.get(product, product)
                row = f"{short:<{product_width}}"
                for _, r in results:
                    pnl = r.metrics.final_pnl_by_product.get(product, 0.0)
                    row += f" {pnl:>{col_width}.2f}"
                print(row)


def run_cli() -> None:
    """Simple CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Python IMC Prosperity 4 Backtester"
    )
    parser.add_argument("--trader", type=str, default=None, help="Path to trader.py")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset path or alias")
    parser.add_argument("--day", type=int, default=None, help="Run only this day")
    parser.add_argument(
        "--trade-match-mode", type=str, default="all",
        choices=["all", "worse", "none"],
    )
    parser.add_argument("--queue-penetration", type=float, default=1.0)
    parser.add_argument("--price-slippage-bps", type=float, default=0.0)

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    trader_path = resolve_trader(args.trader, project_root)
    print(f"trader: {os.path.basename(trader_path)}")

    matching = MatchingConfig(
        trade_match_mode=args.trade_match_mode,
        queue_penetration=args.queue_penetration,
        price_slippage_bps=args.price_slippage_bps,
    )

    if args.day is not None:
        # Run specific day
        if args.dataset:
            if os.path.isfile(args.dataset):
                targets = [(args.dataset, args.day)]
            elif os.path.isdir(args.dataset):
                # Find the right file for this day
                targets = [
                    (p, d) for p, d in _collect_from_dir(args.dataset)
                    if d == args.day
                ]
            else:
                ds_path = os.path.join(project_root, "datasets", args.dataset)
                targets = [
                    (p, d) for p, d in _collect_from_dir(ds_path)
                    if d == args.day
                ]
        else:
            targets = [
                (p, d) for p, d in resolve_datasets(args.dataset, project_root)
                if d == args.day
            ]
    else:
        targets = resolve_datasets(args.dataset, project_root)

    if not targets:
        print("No datasets found for the specified criteria.")
        sys.exit(1)

    labeled_results: List[Tuple[str, BacktestResult]] = []

    for dataset_path, day in targets:
        label = _short_dataset_label(dataset_path)
        if day is not None:
            run_id = f"backtest-{int(time.time() * 1000)}-{label}-day{day}"
        else:
            run_id = f"backtest-{int(time.time() * 1000)}-{label}"

        request = RunRequest(
            trader_file=trader_path,
            dataset_file=dataset_path,
            day=day,
            matching=matching,
            run_id=run_id,
        )

        result = run_backtest(request)
        labeled_results.append((label, result))

    print_summary(labeled_results, trader_path)


if __name__ == "__main__":
    run_cli()
