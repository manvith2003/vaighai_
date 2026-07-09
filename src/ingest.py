"""STAGE 1 — INGEST (Bronze). Azure equivalent: Logic Apps / Data Factory.

Pulls the three sources into the Bronze layer untouched:
  1. MIR field data      (field-staff market estimates)
  2. ERP purchase data   (actual system receipts)
  3. Weather / monsoon   (regional rainfall seasonality for Tamil Nadu coir belts)

Weather: tries the Open-Meteo climate API; if offline, falls back to bundled
IMD-style monthly rainfall climatology so the pipeline always runs.
"""
import os
import shutil
import sys

import pandas as pd

from common import BRONZE, banner

# Approx coordinates of the coir-producing regions
REGION_COORDS = {
    "Pollachi": (10.66, 77.01), "Kangeyam": (11.01, 77.56), "Madurai": (9.93, 78.12),
    "Salem": (11.66, 78.15), "Peravurani": (10.29, 79.20),
}
# Fallback: monthly rainfall climatology (mm), typical Tamil Nadu interior pattern
# (SW monsoon Jun-Sep moderate, NE monsoon Oct-Dec heavy) — used when API unreachable.
CLIMATOLOGY_MM = {
    "Pollachi":   [18, 12, 25, 65, 90, 55, 60, 75, 110, 180, 140, 60],
    "Kangeyam":   [12, 10, 20, 55, 85, 45, 50, 70, 105, 175, 150, 55],
    "Madurai":    [20, 15, 22, 60, 70, 35, 45, 90, 115, 190, 160, 70],
    "Salem":      [10, 12, 28, 70, 95, 60, 75, 95, 130, 170, 120, 45],
    "Peravurani": [30, 18, 20, 50, 55, 30, 40, 80, 120, 230, 220, 110],
}


def fetch_weather():
    """Region x month rainfall table. API first, climatology fallback."""
    try:
        import urllib.request, json  # noqa
        rows = []
        for region, (lat, lon) in REGION_COORDS.items():
            url = (f"https://climate-api.open-meteo.com/v1/climate?latitude={lat}&longitude={lon}"
                   f"&start_date=2015-01-01&end_date=2024-12-31&models=ERA5&daily=precipitation_sum")
            with urllib.request.urlopen(url, timeout=15) as r:
                d = json.load(r)
            s = pd.Series(d["daily"]["precipitation_sum"],
                          index=pd.to_datetime(d["daily"]["time"]))
            monthly = s.groupby(s.index.month).mean() * 30
            for m, mm in monthly.items():
                rows.append({"region": region, "month": int(m), "rain_mm": round(float(mm), 1)})
        src = "open-meteo ERA5"
        wx = pd.DataFrame(rows)
    except Exception as e:  # offline / blocked — use bundled climatology
        rows = [{"region": r, "month": m + 1, "rain_mm": mm}
                for r, months in CLIMATOLOGY_MM.items() for m, mm in enumerate(months)]
        wx = pd.DataFrame(rows)
        src = f"bundled IMD-style climatology (API unavailable: {type(e).__name__})"
    wx["source"] = src
    return wx, src


def run(raw_dir):
    banner("INGEST", "Bronze layer — landing raw data from 3 sources")
    for f in ("mir_field_data.csv", "purchase_data.csv"):
        src = os.path.join(raw_dir, f)
        dst = os.path.join(BRONZE, f)
        shutil.copy(src, dst)
        n = sum(1 for _ in open(dst)) - 1
        print(f"  source {'1' if 'mir' in f else '2'}: {f:<25} -> bronze  ({n:,} rows)")
    wx, src = fetch_weather()
    wx.to_csv(os.path.join(BRONZE, "weather_monthly.csv"), index=False)
    print(f"  source 3: weather_monthly.csv       -> bronze  ({len(wx)} rows, {src})")
    return True


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else ".")
