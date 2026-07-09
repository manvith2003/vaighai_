"""EXTRACTION 2/3 — ERP purchase receipts (actual system records).
Production: read-only API on the legacy ERP. Here: raw CSV drop.
"""
import os
import shutil
import sys

from utils import BRONZE, banner

FILENAME = "purchase_data.csv"


def run(raw_dir):
    banner("EXTRACT 2/3", "ERP purchase data -> bronze")
    src, dst = os.path.join(raw_dir, FILENAME), os.path.join(BRONZE, FILENAME)
    shutil.copy(src, dst)
    n = sum(1 for _ in open(dst)) - 1
    print(f"  {FILENAME}: {n:,} rows landed")
    return n


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
