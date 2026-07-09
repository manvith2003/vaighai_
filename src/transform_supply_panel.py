"""TRANSFORMATION — Bronze -> Silver.

  - MIR: parse, drop future-dated rows, aggregate to supplier x region x quarter
  - ERP purchases: exclude internal transfers, aggregate to same grain
  - Outer join into the supply panel; winsorize field-entry outliers (p99.5)
  - Weather: region x year x month rainfall -> fiscal-quarter actuals + monsoon index
"""
import os

import numpy as np
import pandas as pd

from utils import (BRONZE, SILVER, MONTH_TO_FQ, Q_NUM, QTY_COLS, banner,
                   current_complete_quarter)


def run():
    banner("TRANSFORM", "Bronze -> Silver: clean, join, winsorize")
    cutoff_q = current_complete_quarter()

    mir = pd.read_csv(os.path.join(BRONZE, "mir_field_data.csv"), parse_dates=["date"])
    n0 = len(mir)
    mir = mir.dropna(subset=["fiscal_year", "fiscal_quarter", "supplier"])
    mir["qidx"] = mir["fiscal_year"].astype(int) * 4 + mir["fiscal_quarter"].map(Q_NUM)
    future = int((mir["qidx"] > cutoff_q).sum())
    mir = mir[mir["qidx"] <= cutoff_q]
    mir_agg = mir.groupby(["supplier", "region", "fiscal_year", "fiscal_quarter"], as_index=False)[
        ["mill_produced_MT", "mill_dispatched_MT", "vaighai_offtake_est_MT"]].sum()
    print(f"  MIR: {n0:,} rows -> {len(mir_agg):,} supplier-quarters (dropped {future} future-dated)")

    pur = pd.read_csv(os.path.join(BRONZE, "purchase_data.csv"))
    n0, internal = len(pur), int(pur["is_internal_transfer"].sum())
    pur = pur[~pur["is_internal_transfer"]].dropna(subset=["fiscal_year", "fiscal_quarter"])
    pur["qidx"] = pur["fiscal_year"].astype(int) * 4 + pur["fiscal_quarter"].map(Q_NUM)
    pur = pur[pur["qidx"] <= cutoff_q]
    pur_agg = pur.groupby(["supplier", "region", "fiscal_year", "fiscal_quarter"], as_index=False)[
        ["vaighai_purchased_MT"]].sum()
    print(f"  Purchases: {n0:,} rows -> {len(pur_agg):,} supplier-quarters (excl. {internal:,} internal)")

    keys = ["supplier", "region", "fiscal_year", "fiscal_quarter"]
    panel = mir_agg.merge(pur_agg, on=keys, how="outer", indicator=True)
    panel["in_mir"] = panel["_merge"].isin(["left_only", "both"])
    panel["in_purchase"] = panel["_merge"].isin(["right_only", "both"])
    panel = panel.drop(columns="_merge")
    for c in QTY_COLS:
        panel[c] = panel[c].fillna(0.0)
    panel["fiscal_year"] = panel["fiscal_year"].astype(int)

    for c in QTY_COLS:
        pos = panel.loc[panel[c] > 0, c]
        cap = float(pos.quantile(0.995)) if len(pos) else 0.0
        n_capped = int((panel[c] > cap).sum())
        panel[c] = panel[c].clip(upper=cap)
        if n_capped:
            print(f"  winsorized {n_capped} outliers in {c} (cap {cap:,.0f} MT)")
    panel["our_share_of_dispatch_pct"] = np.where(
        panel["mill_dispatched_MT"] > 0,
        (panel["vaighai_offtake_est_MT"] / panel["mill_dispatched_MT"] * 100).clip(0, 100), np.nan)
    panel.to_csv(os.path.join(SILVER, "supply_panel.csv"), index=False)
    print(f"  silver.supply_panel: {len(panel):,} rows, {panel['supplier'].nunique():,} suppliers")

    # weather: year+month -> fiscal quarter ACTUALS per region (history for training)
    wx = pd.read_csv(os.path.join(BRONZE, "weather_monthly.csv"))
    wx["fiscal_year"] = np.where(wx["month"] >= 4, wx["year"] + 1, wx["year"])
    wx["fiscal_quarter"] = wx["month"].map(MONTH_TO_FQ)
    wq = wx.groupby(["region", "fiscal_year", "fiscal_quarter"], as_index=False)["rain_mm"].sum()
    wq["monsoon_idx"] = wq.groupby("region")["rain_mm"].transform(lambda s: s / s.mean())
    wq.to_csv(os.path.join(SILVER, "weather_quarterly.csv"), index=False)
    print(f"  silver.weather_quarterly: {len(wq):,} region-quarters of rainfall actuals")

    # TARGET-quarter rainfall estimate (the quarter the models predict):
    # completed months -> archive actuals; next 16 days -> LIVE forecast;
    # remaining days -> climatology. Used as the scoring-time value of the
    # rain_next_q feature (training uses actuals of t+1).
    from extract_weather_api import CLIMATOLOGY_MM
    fc = pd.read_csv(os.path.join(BRONZE, "weather_forecast.csv"), parse_dates=["date"])
    tq = cutoff_q + 1
    t_fy, t_qn = (tq - 1) // 4, tq - ((tq - 1) // 4) * 4
    months = {1: [4, 5, 6], 2: [7, 8, 9], 3: [10, 11, 12], 4: [1, 2, 3]}[t_qn]
    today = pd.Timestamp.today().normalize()
    rows = []
    for region in CLIMATOLOGY_MM:
        est = 0.0
        for m in months:
            yr = t_fy if m <= 3 else t_fy - 1
            mstart = pd.Timestamp(yr, m, 1)
            mend = mstart + pd.offsets.MonthEnd(0)
            ndays = mend.day
            clim_daily = CLIMATOLOGY_MM[region][m - 1] / ndays
            if mend < today:  # month fully elapsed -> archive actual
                est += float(wx[(wx["region"] == region) & (wx["year"] == yr)
                                & (wx["month"] == m)]["rain_mm"].sum())
            else:
                f = fc[(fc["region"] == region) & (fc["date"] >= max(mstart, today))
                       & (fc["date"] <= mend)]
                est += float(f["rain_mm"].sum())
                past_days = max((min(today, mend) - mstart).days, 0)
                rem_days = max(ndays - past_days - len(f), 0)
                est += clim_daily * (past_days + rem_days)
        rows.append({"region": region, "fiscal_year": t_fy, "fiscal_quarter": f"FQ{t_qn}",
                     "rain_mm_est": round(est, 1)})
    wnq = pd.DataFrame(rows)
    wnq.to_csv(os.path.join(SILVER, "weather_next_quarter.csv"), index=False)
    print(f"  silver.weather_next_quarter: FY{t_fy} FQ{t_qn} rainfall estimate per region "
          f"(actuals + live 16-day forecast + climatology fill)")


if __name__ == "__main__":
    run()
