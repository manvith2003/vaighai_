# Supply Radar — Use Case 6: Supplier & Sourcing Intelligence

End-to-end procurement intelligence pipeline for Vaighai Agro Products.
Runs **fully locally** (no cloud). Each stage maps one-to-one to an Azure
service for later migration.

```
EXTRACTION                     TRANSFORMATION       WAREHOUSE           ML              AGENT           REPORTING
extract_mir_field_data.py  ->  transform_supply_ -> load_warehouse  ->  ml_train_   ->  agent_brief ->  report_dashboard
extract_erp_purchases.py       panel.py             (SQLite or          score.py        (LLM or         (HTML) or
extract_weather_api.py         dq_validate.py        Postgres)                          template)       Metabase
   bronze/                        silver/            tables+views        gold/          docs/brief      dashboard/
```

## Quick start (zero setup — SQLite)

```bash
pip install -r requirements.txt
python3 src/pipeline.py data/raw          # ~30s end to end
```

Open `dashboard/supply_radar_dashboard.html`, `docs/weekly_sourcing_brief.md`,
`docs/validation_report.md`.

## React dashboard (modern UI — glassmorphism, animated 3D background, tilt effects)

```bash
cd frontend
npm install
npm run dev            # dev server at http://localhost:5173
# or serve the committed production build without installing anything:
python3 -m http.server 8080 --directory frontend/dist   # -> http://localhost:8080
```

The pipeline writes `dashboard_data.json` into `frontend/public/` and `frontend/dist/`
on every run, so the React app always shows the latest scored data. The plain-HTML
dashboard remains as a zero-setup fallback.

## Local production stack (Postgres + Metabase + n8n)

```bash
docker compose up -d
export WAREHOUSE_URL=postgresql+psycopg2://radar:radar@localhost:5432/supplyradar
pip install sqlalchemy psycopg2-binary
python3 src/pipeline.py data/raw
```

- **Metabase** http://localhost:3000 — add the `supplyradar` Postgres database and build
  dashboards on `vw_critical_watchlist`, `vw_top_opportunities`, `vw_region_summary`.
- **n8n** http://localhost:5678 — Schedule trigger (Mon 7:00) → Execute Command
  (`python3 src/pipeline.py data/raw`) → Read File (`docs/weekly_sourcing_brief.md`)
  → Send Email / Teams / WhatsApp.

## LLM agent (optional, free tier)

```bash
export LLM_API_KEY=<your free Groq key>     # console.groq.com
python3 src/pipeline.py data/raw
```

The agent sends **only aggregated facts** (metrics, flagged suppliers, opportunities)
to the LLM — never row-level raw data. Without a key it falls back to deterministic
templates, so the pipeline always completes. Works with any OpenAI-compatible endpoint
(`LLM_BASE_URL`), including a fully in-house Ollama (`http://localhost:11434/v1`).

## The three data sources

1. **MIR field data** — field-staff estimates per mill (produced / dispatched / our offtake)
2. **ERP purchases** — actual receipts; internal transfers excluded in transformation
3. **Weather API** — daily rainfall history per coir region (Open-Meteo archive, free,
   from `WEATHER_START`), rolled up to fiscal-quarter actuals so model **training** joins
   real monsoon data to every supplier-quarter. Offline climatology fallback included.

## Models (all validated on held-out quarters)

- **Decline risk** — logistic model; P(next-quarter dispatch >50% below trailing 4-q avg). AUC ≈ 0.76
- **Forecast** — champion of {ridge, seasonal-naive, blend} picked by backtest WAPE each run
- **Opportunity score** — mill size × share headroom × growth; the "call these mills" list
- **Concentration** — HHI + top-5 supplier share per fiscal year

## Architecture

**Local production stack (what this repo runs today — no cloud):**

![Local stack](docs/uc6_architecture_local_stack.png)

**Azure target architecture (migration path, recommended):**

![Azure without Databricks](docs/uc6_architecture_without_databricks.png)

**Azure Databricks variant (only if data volumes grow):**

![Azure on Databricks](docs/uc6_architecture_on_databricks.png)

## Azure migration map

| Local (this repo) | Azure |
| --- | --- |
| cron / n8n | Logic Apps |
| extract_*.py | Data Factory / Logic Apps connectors |
| transform + dq | Azure Functions (Python) |
| SQLite / Postgres | Azure SQL Database |
| ml_train_score.py | Azure ML batch endpoint + registry |
| agent_brief.py (LLM mode) | Azure AI Foundry agent (in-tenant) |
| HTML dashboard / Metabase | Power BI |
| n8n alerts | Power Automate → Teams |

Docs: `docs/UC6_Supply_Radar_Approach_and_Architecture.docx`, architecture diagrams
(`docs/uc6_architecture_*.png`), validation report and weekly brief (regenerated each run).
