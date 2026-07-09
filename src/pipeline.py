"""Supply Radar — end-to-end pipeline orchestrator (fully local, no cloud).

 EXTRACTION            TRANSFORMATION      WAREHOUSE        ML          AGENT       REPORTING
 extract_mir_field_data  transform_supply_  load_warehouse  ml_train_   agent_      report_
 extract_erp_purchases   panel + dq_validate (SQLite or      score       brief       dashboard
 extract_weather_api                          Postgres)

Schedule with cron / n8n:   0 7 * * MON  python3 src/pipeline.py data/raw

Usage:  python3 src/pipeline.py <raw data folder>
"""
import json
import os
import sys
import time

import agent_brief
import dq_validate
import extract_erp_purchases
import extract_mir_field_data
import extract_weather_api
import load_warehouse
import ml_train_score
import report_dashboard
import transform_supply_panel
from utils import ROOT


def main():
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "raw")
    t0 = time.time()

    extract_mir_field_data.run(raw_dir)   # E1: MIR field estimates
    extract_erp_purchases.run(raw_dir)    # E2: ERP purchase receipts
    extract_weather_api.run()             # E3: rainfall history (API)
    transform_supply_panel.run()          # T:  clean + join -> silver
    dq_validate.run()                     # DQ: hard gates (fail-fast)
    load_warehouse.load_silver()          # L:  silver -> warehouse
    metrics = ml_train_score.run()        # ML: train + score -> gold
    load_warehouse.load_gold()            # L:  gold -> warehouse + views
    agent_brief.run()                     # A:  LLM/template sourcing brief
    report_dashboard.run()                # R:  dashboard build

    print(f"\n{'=' * 70}\nPIPELINE COMPLETE in {time.time()-t0:.1f}s")
    print(json.dumps(metrics, indent=2))
    print("Open:  dashboard/supply_radar_dashboard.html   docs/weekly_sourcing_brief.md")


if __name__ == "__main__":
    main()
