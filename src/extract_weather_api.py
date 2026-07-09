"""EXTRACTION 3/3 — Weather history per coir region (monsoon signal for the models).

Pulls DAILY rainfall history from the Open-Meteo archive API (free, no key)
from WEATHER_START to today for each region, aggregated to region x year x month.
The transformation stage rolls this up to fiscal quarters so the ML features
join actual rainfall to each supplier-quarter — history is needed for training,
not just the current period.

Offline fallback: bundled IMD-style monthly climatology replicated across the
training years (clearly labelled in the `source` column) so the pipeline
always runs.
"""
import datetime as dt
import json
import os
import urllib.request

import pandas as pd

import config
from utils import BRONZE, banner

FILENAME = "weather_monthly.csv"
REGION_COORDS = {
    "Pollachi": (10.66, 77.01), "Kangeyam": (11.01, 77.56), "Madurai": (9.93, 78.12),
    "Salem": (11.66, 78.15), "Peravurani": (10.29, 79.20),
}
# Fallback monthly rainfall climatology (mm) — Tamil Nadu interior pattern
CLIMATOLOGY_MM = {
    "Pollachi":   [18, 12, 25, 65, 90, 55, 60, 75, 110, 180, 140, 60],
    "Kangeyam":   [12, 10, 20, 55, 85, 45, 50, 70, 105, 175, 150, 55],
    "Madurai":    [20, 15, 22, 60, 70, 35, 45, 90, 115, 190, 160, 70],
    "Salem":      [10, 12, 28, 70, 95, 60, 75, 95, 130, 170, 120, 45],
    "Peravurani": [30, 18, 20, 50, 55, 30, 40, 80, 120, 230, 220, 110],
}
API = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
       "&start_date={start}&end_date={end}&daily=precipitation_sum&timezone=Asia%2FKolkata")


def _from_api():
    end = (dt.date.today() - dt.timedelta(days=7)).isoformat()  # archive lags a few days
    rows = []
    for region, (lat, lon) in REGION_COORDS.items():
        url = API.format(lat=lat, lon=lon, start=config.WEATHER_START, end=end)
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.load(r)
        s = pd.Series(d["daily"]["precipitation_sum"],
                      index=pd.to_datetime(d["daily"]["time"]), dtype=float)
        monthly = s.groupby([s.index.year, s.index.month]).sum()
        for (yr, mo), mm in monthly.items():
            rows.append({"region": region, "year": int(yr), "month": int(mo),
                         "rain_mm": round(float(mm), 1)})
        print(f"  {region}: {len(monthly)} months of daily-rainfall history")
    df = pd.DataFrame(rows)
    df["source"] = "open-meteo archive (actuals)"
    return df


def _from_climatology():
    start_year = int(config.WEATHER_START[:4])
    years = range(start_year, dt.date.today().year + 1)
    rows = [{"region": r, "year": y, "month": m + 1, "rain_mm": mm}
            for r, months in CLIMATOLOGY_MM.items() for y in years
            for m, mm in enumerate(months)]
    df = pd.DataFrame(rows)
    df["source"] = "bundled climatology (API unavailable)"
    return df


def run():
    banner("EXTRACT 3/3", f"Weather history ({config.WEATHER_START} -> today) -> bronze")
    try:
        wx = _from_api()
    except Exception as e:
        print(f"  API unreachable ({type(e).__name__}) — using bundled climatology fallback")
        wx = _from_climatology()
    wx.to_csv(os.path.join(BRONZE, FILENAME), index=False)
    print(f"  {FILENAME}: {len(wx):,} region-months landed ({wx['source'].iloc[0]})")
    return len(wx)


if __name__ == "__main__":
    run()
