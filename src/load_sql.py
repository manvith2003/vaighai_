"""STAGE 4 & 6 — SQL WAREHOUSE. Azure equivalent: Azure SQL Database.

SQLite plays the role of Azure SQL locally — same tables, same views, same SQL.
Called twice by the orchestrator: after CLEAN (silver tables) and after MODELS
(gold tables + reporting views).
"""
import os
import shutil
import sqlite3

import pandas as pd

from common import DB_PATH, DB_SNAPSHOT, GOLD, SILVER, banner

VIEWS = {
    "vw_critical_watchlist": """
        SELECT supplier, region, latest_q_dispatch_MT, trailing_4q_avg_MT,
               ROUND(decline_risk*100) AS risk_pct, forecast_next_q_MT, expected_next_q_MT
        FROM watchlist_decline_risk WHERE risk_band='Critical'
        ORDER BY decline_risk DESC""",
    "vw_top_opportunities": """
        SELECT supplier, region, dispatched_MT, ROUND(share_pct,1) AS our_share_pct,
               ROUND(growth_pct) AS yoy_growth_pct, untapped_MT, opportunity_score
        FROM sourcing_opportunities ORDER BY opportunity_score DESC LIMIT 25""",
    "vw_region_summary": """
        SELECT region, fiscal_year, SUM(mill_dispatched_MT) AS market_MT,
               SUM(vaighai_offtake_est_MT) AS offtake_MT, SUM(vaighai_purchased_MT) AS purchased_MT
        FROM supply_panel GROUP BY region, fiscal_year""",
}


def load_silver():
    banner("SQL", f"Loading Silver tables into warehouse ({os.path.basename(DB_PATH)})")
    con = sqlite3.connect(DB_PATH)
    for name in ("supply_panel", "weather_quarterly"):
        df = pd.read_csv(os.path.join(SILVER, f"{name}.csv"))
        df.to_sql(name, con, if_exists="replace", index=False)
        print(f"  table {name}: {len(df):,} rows")
    con.close()


def load_gold():
    banner("SQL", "Loading Gold tables + reporting views")
    con = sqlite3.connect(DB_PATH)
    for name in ("watchlist_decline_risk", "sourcing_opportunities", "concentration_risk"):
        df = pd.read_csv(os.path.join(GOLD, f"{name}.csv"))
        df.to_sql(name, con, if_exists="replace", index=False)
        print(f"  table {name}: {len(df):,} rows")
    cur = con.cursor()
    for name, sql in VIEWS.items():
        cur.execute(f"DROP VIEW IF EXISTS {name}")
        cur.execute(f"CREATE VIEW {name} AS {sql}")
        n = cur.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  view  {name}: {n} rows")
    con.commit()
    con.close()
    try:  # snapshot the warehouse into the repo for inspection / git
        shutil.copy(DB_PATH, DB_SNAPSHOT)
        print(f"  warehouse snapshot -> data/{os.path.basename(DB_SNAPSHOT)}")
    except OSError as e:
        print(f"  (snapshot skipped: {e})")


if __name__ == "__main__":
    load_silver()
    load_gold()
