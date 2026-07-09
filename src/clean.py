"""STAGE 2 — CLEAN & JOIN (Silver). Azure equivalent: Azure Functions (Python).

Builds the supplier x region x fiscal-quarter panel from RAW sources:
  - MIR: parse dates, drop future-dated rows, aggregate to quarter grain
  - Purchases: exclude internal transfers (~70% of volume), aggregate
  - Outer join on supplier + region + fiscal_year + fiscal_quarter
  - Winsorize field-entry outliers (99.5th pct of positive values)
  - Weather: monthly rainfall -> fiscal-quarter monsoon index per region
"""
import os

import numpy as np
import pandas as pd

from common import BRONZE, SILVER, QTY_COLS, Q_NUM, banner, current_complete_quarter

MONTH_TO_FQ = {4: "FQ1", 5: "FQ1", 6: "FQ1", 7: "FQ2", 8: "FQ2", 9: "FQ2",
               10: "FQ3", 11: "FQ3", 12: "FQ3", 1: "FQ4", 2: "FQ4", 3: "FQ4"}


def run():
    banner("CLEAN", "Silver layer — clean, join, winsorize")
    cutoff_q = current_complete_quarter()

    # ---- MIR (source 1)
    mir = pd.read_csv(os.path.join(BRONZE, "mir_field_data.csv"), parse_dates=["date"])
    n0 = len(mir)
    mir = mir.dropna(subset=["fiscal_year", "fiscal_quarter", "supplier"])
    mir["qidx"] = mir["fiscal_year"].astype(int) * 4 + mir["fiscal_quarter"].map(Q_NUM)
    future = (mir["qidx"] > cutoff_q).sum()
    mir = mir[mir["qidx"] <= cutoff_q]
    mir_agg = mir.groupby(["supplier", "region", "fiscal_year", "fiscal_quarter"], as_index=False)[
        ["mill_produced_MT", "mill_dispatched_MT", "vaighai_offtake_est_MT"]].sum()
    print(f"  MIR: {n0:,} raw rows -> {len(mir_agg):,} supplier-quarters "
          f"(dropped {future:,} future-dated rows)")

    # ---- Purchases (source 2)
    pur = pd.read_csv(os.path.join(BRONZE, "purchase_data.csv"))
    n0, internal = len(pur), int(pur["is_internal_transfer"].sum())
    pur = pur[~pur["is_internal_transfer"]].dropna(subset=["fiscal_year", "fiscal_quarter"])
    pur["qidx"] = pur["fiscal_year"].astype(int) * 4 + pur["fiscal_quarter"].map(Q_NUM)
    pur = pur[pur["qidx"] <= cutoff_q]
    pur_agg = pur.groupby(["supplier", "region", "fiscal_year", "fiscal_quarter"], as_index=False)[
        ["vaighai_purchased_MT"]].sum()
    print(f"  Purchases: {n0:,} raw rows -> {len(pur_agg):,} supplier-quarters "
          f"(excluded {internal:,} internal-transfer rows)")

    # ---- Outer join (rule 5 of the build pack)
    keys = ["supplier", "region", "fiscal_year", "fiscal_quarter"]
    panel = mir_agg.merge(pur_agg, on=keys, how="outer", indicator=True)
    panel["in_mir"] = panel["_merge"].isin(["left_only", "both"])
    panel["in_purchase"] = panel["_merge"].isin(["right_only", "both"])
    panel = panel.drop(columns="_merge")
    for c in QTY_COLS:
        panel[c] = panel[c].fillna(0.0)
    panel["fiscal_year"] = panel["fiscal_year"].astype(int)

    # ---- Winsorize implausible field entries (e.g. 2.1 million MT in one quarter)
    caps = {}
    for c in QTY_COLS:
        pos = panel.loc[panel[c] > 0, c]
        cap = float(pos.quantile(0.995)) if len(pos) else 0.0
        caps[c] = cap
        n_capped = int((panel[c] > cap).sum())
        panel[c] = panel[c].clip(upper=cap)
        if n_capped:
            print(f"  winsorized {n_capped} outlier values in {c} (cap {cap:,.0f} MT)")
    panel["our_share_of_dispatch_pct"] = np.where(
        panel["mill_dispatched_MT"] > 0,
        (panel["vaighai_offtake_est_MT"] / panel["mill_dispatched_MT"] * 100).clip(0, 100), np.nan)

    panel.to_csv(os.path.join(SILVER, "supply_panel.csv"), index=False)
    print(f"  Silver panel: {len(panel):,} rows, {panel['supplier'].nunique():,} suppliers")

    # ---- Weather (source 3): month -> fiscal-quarter monsoon index per region
    wx = pd.read_csv(os.path.join(BRONZE, "weather_monthly.csv"))
    wx["fiscal_quarter"] = wx["month"].map(MONTH_TO_FQ)
    wq = wx.groupby(["region", "fiscal_quarter"], as_index=False)["rain_mm"].sum()
    wq["monsoon_idx"] = wq.groupby("region")["rain_mm"].transform(lambda s: s / s.mean())
    wq.to_csv(os.path.join(SILVER, "weather_quarterly.csv"), index=False)
    print(f"  Weather: quarterly monsoon index for {wq['region'].nunique()} regions")
    return caps


if __name__ == "__main__":
    run()
