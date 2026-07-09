"""EXTRACTION 1/3 — MIR field data (market intelligence estimates).
Production: API pull from the MIR collection system. Here: raw CSV drop.
Lands the source untouched in the Bronze layer with an extraction audit line.
"""
import os
import shutil
import sys

from utils import BRONZE, banner

FILENAME = "mir_field_data.csv"


def run(raw_dir):
    banner("EXTRACT 1/3", "MIR field data -> bronze")
    src, dst = os.path.join(raw_dir, FILENAME), os.path.join(BRONZE, FILENAME)
    shutil.copy(src, dst)
    n = sum(1 for _ in open(dst)) - 1
    print(f"  {FILENAME}: {n:,} rows landed")
    return n


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
