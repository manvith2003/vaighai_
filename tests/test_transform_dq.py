"""Import smoke test + transform/DQ integration test.

The import test is the cheap safety net: it fails fast on any syntax/name error in the
pipeline modules. The integration test builds tiny bronze fixtures, runs the transform,
and checks the join grain, the share formula, and the log-scale winsorizer — it skips
(never falsely fails) if the local environment can't be set up.
"""
import importlib

import pandas as pd
import pytest


def test_modules_import():
    for mod in ("mlx", "utils", "transform_supply_panel", "ml_train_score",
                "dq_validate", "agent_brief", "pipeline"):
        importlib.import_module(mod)


def test_transform_grain_share_and_winsorize(tmp_path):
    try:
        import transform_supply_panel as T
        # redirect the module's I/O to a temp bronze/silver
        bronze, silver = tmp_path / "bronze", tmp_path / "silver"
        bronze.mkdir(); silver.mkdir()
        T.BRONZE, T.SILVER = str(bronze), str(silver)

        # MIR: one small mill + one genuinely huge mill (winsorizer must NOT flatten it
        # to the small bulk).
        mir = pd.DataFrame({
            "supplier": ["Small Coir"] * 6 + ["Big Mill"] * 6,
            "region": ["Pollachi"] * 12,
            "subregion": ["x"] * 12,
            "date": pd.date_range("2021-04-01", periods=6, freq="QS").tolist() * 2,
            "fiscal_year": [2022] * 12,
            "fiscal_quarter": (["FQ1", "FQ2", "FQ3", "FQ4", "FQ1", "FQ2"]) * 2,
            "mill_produced_MT": [10, 12, 11, 9, 10, 13, 9000, 9100, 8900, 9200, 9000, 9100],
            "mill_dispatched_MT": [8, 10, 9, 7, 8, 11, 8000, 8100, 7900, 8200, 8000, 8100],
            "vaighai_offtake_est_MT": [4, 5, 4, 3, 4, 5, 800, 810, 790, 820, 800, 810],
        })
        mir.to_csv(bronze / "mir_field_data.csv", index=False)

        pur = pd.DataFrame({
            "supplier": ["Small Coir", "Big Mill"], "supplier_raw": ["Small Coir", "Big Mill"],
            "region": ["Pollachi", "Pollachi"],
            "date": ["2021-05-01", "2021-05-01"], "fiscal_year": [2022, 2022],
            "fiscal_quarter": ["FQ1", "FQ1"], "vaighai_purchased_MT": [4.0, 800.0],
            "is_internal_transfer": [False, False]})
        pur.to_csv(bronze / "purchase_data.csv", index=False)

        yrs = range(2020, 2024)
        wm = pd.DataFrame([{"region": "Pollachi", "year": y, "month": m, "rain_mm": 40.0}
                           for y in yrs for m in range(1, 13)])
        wm.to_csv(bronze / "weather_monthly.csv", index=False)
        wf = pd.DataFrame([{"region": "Pollachi", "date": d.date().isoformat(),
                            "rain_mm": 2.0, "source": "test"}
                           for d in pd.date_range("2021-01-01", periods=400)])
        wf.to_csv(bronze / "weather_forecast.csv", index=False)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"fixture setup unavailable: {e}")

    T.run()
    panel = pd.read_csv(silver / "supply_panel.csv")

    # 1) join grain: unique per supplier x region x fiscal_year x fiscal_quarter
    assert not panel.duplicated(["supplier", "region", "fiscal_year", "fiscal_quarter"]).any()
    # 2) share formula: Small Coir FQ1 offtake 4 / dispatch 8 = 50%
    row = panel[(panel.supplier == "Small Coir") & (panel.fiscal_quarter == "FQ1")].iloc[0]
    assert abs(row["our_share_of_dispatch_pct"] - 50.0) < 1e-6
    # 3) winsorizer keeps the genuinely large mill (not clipped to the small bulk)
    big = panel[panel.supplier == "Big Mill"]["mill_dispatched_MT"].max()
    assert big > 5000
    # 4) richer weather columns exist
    wq = pd.read_csv(silver / "weather_quarterly.csv")
    for c in ("rain_lag1", "monsoon_cum", "extreme_wet", "dry_spell"):
        assert c in wq.columns
