# Supply Radar — Use Case 6: Supplier & Sourcing Intelligence

End-to-end procurement intelligence pipeline for Vaighai Agro Products.
Runs **fully locally** today; every stage maps one-to-one to an Azure service
for migration once cloud access is provisioned.

```
3 sources ──> INGEST ──> CLEAN ──> VALIDATE ──> SQL ──> ML MODELS ──> AGENT ──> DASHBOARD
              bronze     silver    DQ gates    warehouse   gold        brief      html
```

| Local (this repo) | Azure (production) |
| --- | --- |
| `src/ingest.py` — 3 sources to bronze | Logic Apps / Data Factory |
| `src/clean.py` — clean, join, winsorize | Azure Functions (Python) |
| `src/validate.py` — hard/soft DQ gates | DQ checks + Log Analytics |
| SQLite (`data/supply_radar.db`) | Azure SQL Database |
| `src/models.py` — decline risk + forecast | Azure ML batch endpoint + registry |
| `src/agent.py` — plain-language brief | Azure AI Foundry agent (in-tenant) |
| `src/report.py` — HTML dashboard | Power BI |
| `src/run_pipeline.py` — orchestrator | Logic Apps schedule + Power Automate alerts |

## The three data sources

1. **MIR field data** (`data/raw/mir_field_data.csv`) — field-staff estimates of what each
   mill produced, dispatched, and how much Vaighai took. Human estimates.
2. **ERP purchases** (`data/raw/purchase_data.csv`) — actual system receipts.
   Internal transfers (~70% of volume) are excluded during cleaning.
3. **Weather / monsoon** — regional rainfall seasonality (Open-Meteo climate API,
   with a bundled IMD-style climatology fallback so the pipeline runs offline).

## Run it

```bash
pip install pandas numpy
python3 src/run_pipeline.py data/raw
```

~30 seconds. Then open:

- `dashboard/supply_radar_dashboard.html` — interactive dashboard (any browser)
- `docs/weekly_sourcing_brief.md` — the agent's act-here list
- `docs/validation_report.md` — data-quality report
- `data/supply_radar.db` — the SQL warehouse (tables + views, try `vw_critical_watchlist`)

## What the models do

- **Decline risk** — logistic model on lag/seasonality features; predicts the probability
  each supplier's dispatch falls >50% below its trailing 4-quarter average next quarter.
  Validated on held-out quarters (AUC ≈ 0.76).
- **Forecast** — three candidates (ridge, seasonal-naive, blend) backtested every run;
  the champion by WAPE makes the forecast. Honest and self-upgrading.
- **Opportunity score** — mill size percentile × share headroom × growth factor;
  the "call these mills" list.
- **Concentration** — HHI and top-5 supplier share per fiscal year.

## Docs

- `docs/UC6_Supply_Radar_Approach_and_Architecture.docx` — full approach & architecture
- `docs/uc6_architecture_without_databricks.png` — Azure target architecture (recommended)
- `docs/uc6_architecture_on_databricks.png` — Databricks variant if data volumes grow
