# Supply Radar

**Use Case 6 — Supplier & Sourcing Intelligence · Vaighai Agro Products Ltd**

End-to-end procurement intelligence pipeline. Fully local execution; one-to-one Azure migration path.

```
EXTRACT (3 sources) → TRANSFORM → VALIDATE → WAREHOUSE → ML → AGENT → DASHBOARD
      bronze            silver      DQ gates   SQL         gold   brief    React / Metabase
```

---

## Quick start

```bash
pip install -r requirements.txt
python3 src/pipeline.py data/raw
```

Runtime ≈ 30 s. Outputs:

| Output | Path |
| --- | --- |
| React dashboard | `frontend/dist` (serve) or `cd frontend && npm run dev` |
| HTML dashboard (fallback) | `dashboard/supply_radar_dashboard.html` |
| Weekly sourcing brief | `docs/weekly_sourcing_brief.md` |
| Data-quality report | `docs/validation_report.md` |
| SQL warehouse | `data/supply_radar.db` |

## Production stack (optional)

```bash
docker compose up -d        # PostgreSQL + Metabase + n8n
export WAREHOUSE_URL=postgresql+psycopg2://radar:radar@localhost:5432/supplyradar
pip install sqlalchemy psycopg2-binary
python3 src/pipeline.py data/raw
```

| Service | URL | Purpose |
| --- | --- | --- |
| Metabase | `localhost:3000` | BI dashboards on warehouse views |
| n8n | `localhost:5678` | Scheduling + email/Teams alert delivery |
| PostgreSQL | `localhost:5432` | Warehouse (`supplyradar`) |

## Configuration

Set via environment (see `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `WAREHOUSE_URL` | empty → SQLite | Postgres connection string |
| `LLM_API_KEY` | empty → template mode | Enables LLM-written brief (Groq free tier / any OpenAI-compatible endpoint) |
| `LLM_BASE_URL` | Groq | `http://localhost:11434/v1` for in-house Ollama |
| `WEATHER_START` | 2020-04-01 | Weather history window for training |

## Data sources

| # | Source | Nature | Handling |
| --- | --- | --- | --- |
| 1 | MIR field data | Field-staff market estimates per mill | Future-dated rows dropped; outliers winsorized (p99.5) |
| 2 | ERP purchases | System receipts | Internal transfers excluded |
| 3 | Weather API (Open-Meteo) | Daily rainfall actuals + 16-day live forecast | Quarterly aggregation; climatology fallback offline |

## Models

Validated on held-out quarters every run; no future data leaks into training.

| Model | Method | Output | Validation |
| --- | --- | --- | --- |
| Decline risk | Logistic regression, 19 features (lags, momentum, share, monsoon) | P(next-quarter dispatch >50% below 4-q avg) | AUC ≈ 0.76 |
| Volume forecast | Champion of ridge / seasonal-naive / blend | Next-quarter MT per supplier | WAPE, champion auto-selected |
| Opportunity score | Size percentile × share headroom × growth | Ranked sourcing shortlist | — |
| Concentration | HHI, top-5 share per FY | Dependency trend | — |

Weather feature `rain_next_q`: training rows use rainfall **actuals** of the target
quarter; scoring rows use elapsed actuals + 16-day live forecast + climatology fill.

## MLOps

Every pipeline run executes the full loop (`src/model_registry.py`):

| Step | Mechanism |
| --- | --- |
| Retraining | All history through the latest complete quarter, every run |
| Hyperparameter search | Grid over `l2 × lr` (classifier) and `l2` (ridge), selected on held-out quarters |
| Champion selection | Best forecaster of {ridge, seasonal-naive, blend} by backtest WAPE |
| Promotion gate | Tuned challenger replaces registered champion only if validation AUC does not regress |
| Versioning | Weights per run in `data/models/model_v{N}.json`; audit trail in `registry.jsonl` |
| Monitoring | `ml_runs` warehouse table — chart model quality over time in Metabase |

Azure mapping: registry → Azure ML model registry; tuning → Azure ML sweep jobs;
`ml_runs` monitoring → Azure ML job metrics.

## Architecture

**Local production stack (current):**

![Local stack](docs/uc6_architecture_local_stack.png)

**Azure target (recommended migration):**

![Azure](docs/uc6_architecture_without_databricks.png)

**Azure Databricks variant (high-volume scenario):**

![Azure Databricks](docs/uc6_architecture_on_databricks.png)

## Azure migration map

| Local | Azure |
| --- | --- |
| n8n / cron | Logic Apps |
| `extract_*.py` | Data Factory |
| `transform_supply_panel.py`, `dq_validate.py` | Azure Functions |
| SQLite / PostgreSQL | Azure SQL Database |
| `ml_train_score.py` | Azure ML (batch endpoint + registry) |
| `agent_brief.py` | Azure AI Foundry (in-tenant) |
| React / Metabase | Power BI |
| n8n alerts | Power Automate → Teams |

## Repository layout

```
src/            pipeline modules (extract → transform → dq → load → ml → agent → report)
data/raw        input CSVs          data/bronze|silver|gold  pipeline layers
frontend/       React dashboard (Vite + recharts)
dashboard/      HTML fallback dashboard
docs/           architecture diagrams, approach document, generated reports
```

## Scheduling

```
cron:  0 7 * * MON  python3 src/pipeline.py data/raw
n8n:   Schedule trigger → Execute Command → Read File (brief) → Send Email / Teams
```

---

*Internal — contains supplier-identifying data. Keep repository private.*
