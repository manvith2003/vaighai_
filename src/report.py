"""STAGE 8 — DASHBOARD. Azure equivalent: Power BI on Azure SQL.

Builds the interactive HTML dashboard from the SQL warehouse: injects the data
as JSON into dashboard/template.html -> dashboard/supply_radar_dashboard.html.
"""
import json
import os
import sqlite3

import pandas as pd

from common import DASH, DB_PATH, GOLD, Q_NUM, banner


def run():
    banner("DASHBOARD", "Building supply_radar_dashboard.html from the warehouse")
    con = sqlite3.connect(DB_PATH)
    panel = pd.read_sql("SELECT * FROM supply_panel", con)
    watch = pd.read_sql("SELECT * FROM watchlist_decline_risk ORDER BY decline_risk DESC", con)
    opp = pd.read_sql("SELECT * FROM sourcing_opportunities ORDER BY opportunity_score DESC", con)
    conc = pd.read_sql("SELECT * FROM concentration_risk ORDER BY fiscal_year", con)
    con.close()
    m = json.loads(open(os.path.join(GOLD, "model_metrics.json")).read())

    panel["qidx"] = panel["fiscal_year"] * 4 + panel["fiscal_quarter"].map(Q_NUM)
    mir_era = panel[panel["fiscal_year"] >= 2021]
    trend = (mir_era.groupby(["fiscal_year", "fiscal_quarter"], as_index=False)
             .agg(dispatched_MT=("mill_dispatched_MT", "sum"),
                  offtake_MT=("vaighai_offtake_est_MT", "sum"),
                  purchased_MT=("vaighai_purchased_MT", "sum"))
             .sort_values(["fiscal_year", "fiscal_quarter"]))
    trend["label"] = "FY" + trend["fiscal_year"].astype(str) + " " + trend["fiscal_quarter"]

    mix = (panel[panel["fiscal_year"] == m["latest_complete_fy"]]
           .groupby("region", as_index=False)
           .agg(dispatched_MT=("mill_dispatched_MT", "sum"),
                offtake_MT=("vaighai_offtake_est_MT", "sum")))
    mix = mix[mix["dispatched_MT"] > 0].sort_values("dispatched_MT", ascending=False)

    data = {"meta": m,
            "watchlist": watch.head(40).fillna("").to_dict(orient="records"),
            "opportunities": opp.head(25).round(1).fillna("").to_dict(orient="records"),
            "trend": trend.round(0).to_dict(orient="records"),
            "region_mix": mix.round(0).to_dict(orient="records"),
            "concentration": conc.to_dict(orient="records")}

    tpl = open(os.path.join(DASH, "template.html")).read()
    out = os.path.join(DASH, "supply_radar_dashboard.html")
    with open(out, "w") as f:
        f.write(tpl.replace("/*__DATA__*/", json.dumps(data, default=str)))
    print(f"  -> dashboard/supply_radar_dashboard.html (open in any browser)")
    return out


if __name__ == "__main__":
    run()
