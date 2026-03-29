#!/usr/bin/env python3
"""
colab_setup.py — Helper for Google Colab environment.

Downloads the tutorial datasets from the original GitHub repo if they
are not already present locally.
"""

import os
import urllib.request
from pathlib import Path

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/GeyzsoN/"
    "prosperity_rust_backtester/main/datasets/tutorial"
)

DATASET_FILES = [
    "prices_round_0_day_-1.csv",
    "trades_round_0_day_-1.csv",
    "prices_round_0_day_-2.csv",
    "trades_round_0_day_-2.csv",
    "submission.log",
]


def setup_datasets(project_root: str = None) -> None:
    """Download tutorial datasets if not present."""
    if project_root is None:
        project_root = os.path.dirname(os.path.abspath(__file__))

    tutorial_dir = os.path.join(project_root, "datasets", "tutorial")
    os.makedirs(tutorial_dir, exist_ok=True)

    for filename in DATASET_FILES:
        local_path = os.path.join(tutorial_dir, filename)
        if os.path.isfile(local_path):
            print(f"  [OK] {filename} already exists")
            continue

        url = f"{GITHUB_RAW_BASE}/{filename}"
        print(f"  [DL] Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"  [OK] {filename} saved")
        except Exception as e:
            print(f"  [ERR] Failed to download {filename}: {e}")
            print(f"        Please manually download from:")
            print(f"        {url}")

    # Create round directories for future use
    for i in range(1, 9):
        os.makedirs(
            os.path.join(project_root, "datasets", f"round{i}"),
            exist_ok=True,
        )

    # Create runs directory
    os.makedirs(os.path.join(project_root, "runs"), exist_ok=True)

    print("\n  Dataset setup complete!")


if __name__ == "__main__":
    setup_datasets()
