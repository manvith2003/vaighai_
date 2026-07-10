"""WAREHOUSE LOAD — Silver/Gold tables + reporting views.

Backend chosen by WAREHOUSE_URL env var:
  unset -> SQLite  (zero-setup local default; snapshot committed to data/)
  postgresql+psycopg2://radar:radar@localhost:5432/supplyradar -> Postgres
    (start it with: docker compose up -d postgres)

Same tables, same views, same SQL either way — this is the Azure SQL stand-in.
"""
import os
import shutil

import pandas as pd

from utils import GOLD, SILVER, SQLITE_PATH, SQLITE_SNAPSHOT, banner, wh_connect, wh_execute

# NOTE: mixed-case column names must be double-quoted — Postgres folds unquoted
# identifiers to lowercase (SQLite is case-insensitive, so it never complained).
VIEWS = {
    "vw_critical_watchlist": """
        SELECT supplier, region, "latest_q_dispatch_MT", "trailing_4q_avg_MT",
               ROUND(decline_risk*100) AS risk_pct, "forecast_next_q_MT", "expected_next_q_MT"
        FROM watchlist_decline_risk WHERE risk_band='Critical'
        ORDER BY decline_risk DESC""",
    "vw_top_opportunities": """
        SELECT supplier, region, "dispatched_MT", ROUND(CAST(share_pct AS numeric),1) AS our_share_pct,
               ROUND(CAST(growth_pct AS numeric)) AS yoy_growth_pct, "untapped_MT", opportunity_score
        FROM sourcing_opportunities ORDER BY opportunity_score DESC LIMIT 25""",
    "vw_region_summary": """
        SELECT region, fiscal_year, SUM("mill_dispatched_MT") AS market_MT,
               SUM("vaighai_offtake_est_MT") AS offtake_MT, SUM("vaighai_purchased_MT") AS purchased_MT
        FROM supply_panel GROUP BY region, fiscal_year""",
}


def load_silver():
    con, backend = wh_connect()
    banner("WAREHOUSE", f"Loading Silver tables ({backend})")
    for name in ("supply_panel", "weather_quarterly", "weather_next_quarter"):
        df = pd.read_csv(os.path.join(SILVER, f"{name}.csv"))
        df.to_sql(name, con, if_exists="replace", index=False)
        print(f"  table {name}: {len(df):,} rows")
    if backend == "sqlite":
        con.close()


def load_gold():
    con, backend = wh_connect()
    banner("WAREHOUSE", f"Loading Gold tables + views ({backend})")
    for name in ("watchlist_decline_risk", "sourcing_opportunities", "concentration_risk"):
        df = pd.read_csv(os.path.join(GOLD, f"{name}.csv"))
        df.to_sql(name, con, if_exists="replace", index=False)
        print(f"  table {name}: {len(df):,} rows")
    # MLOps audit trail: model runs history for Metabase / monitoring
    import model_registry
    runs = model_registry.runs_table()
    if len(runs):
        runs.to_sql("ml_runs", con, if_exists="replace", index=False)
        print(f"  table ml_runs: {len(runs):,} model runs tracked")
    for name, sql in VIEWS.items():
        wh_execute(con, backend, f"DROP VIEW IF EXISTS {name}")
        wh_execute(con, backend, f"CREATE VIEW {name} AS {sql}")
        print(f"  view  {name} created")
    if backend == "sqlite":
        con.close()
        try:
            shutil.copy(SQLITE_PATH, SQLITE_SNAPSHOT)
            print(f"  warehouse snapshot -> data/{os.path.basename(SQLITE_SNAPSHOT)}")
        except OSError as e:
            print(f"  (snapshot skipped: {e})")


if __name__ == "__main__":
    load_silver()
    load_gold()
