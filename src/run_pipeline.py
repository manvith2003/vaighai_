"""Supply Radar — end-to-end pipeline orchestrator (local, no Azure needed).

    3 sources -> INGEST -> CLEAN -> VALIDATE -> SQL -> ML MODELS -> SQL -> AGENT -> DASHBOARD
     (bronze)              (silver)            (warehouse)  (gold)          (brief)  (html)

Azure mapping: Logic Apps -> Functions -> DQ gates -> Azure SQL -> Azure ML
               -> AI Foundry -> Power BI. Same code layout, different runtime.

Usage:  python3 src/run_pipeline.py <folder containing the two raw CSVs>
"""
import json
import os
import sys
import time

import agent
import clean
import ingest
import load_sql
import models
import report
import validate
from common import GOLD, ROOT


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "raw")
    t0 = time.time()
    ingest.run(raw_dir)          # 1. three sources -> bronze
    clean.run()                  # 2. clean + join -> silver
    validate.run()               # 3. quality gates (fails pipeline on hard errors)
    load_sql.load_silver()       # 4. silver -> SQL warehouse
    metrics = models.run()       # 5. ML -> gold
    load_sql.load_gold()         # 6. gold -> SQL warehouse + views
    agent.run()                  # 7. agent writes the sourcing brief
    report.run()                 # 8. dashboard build

    print(f"\n{'=' * 70}\nPIPELINE COMPLETE in {time.time()-t0:.1f}s")
    print(json.dumps(metrics, indent=2))
    print("Open:  dashboard/supply_radar_dashboard.html   docs/weekly_sourcing_brief.md")


if __name__ == "__main__":
    main()
