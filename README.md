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
| Standalone analysis dashboard | `dashboard/supply_radar_analysis.html` (open, load `supply_radar_joined.csv`; includes a live weather-impact panel) |
| Weekly sourcing brief | `docs/weekly_sourcing_brief.md` |
| Data-quality report | `docs/validation_report.md` |
| ML diagnostics (feature importance + reliability) | `docs/ml_diagnostics_report.md` (run `python3 src/ml_diagnostics.py`) |
| SQL warehouse | `data/supply_radar.db` |

### Tests

```bash
pip install pytest
pytest -q        # helper unit tests + transform integration + import smoke test
```

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
| 1 | MIR field data | Field-staff market estimates per mill | Future-dated rows dropped; outliers capped log-scale (median + 5·MAD, keeps large real mills) |
| 2 | ERP purchases | System receipts | Internal transfers excluded |
| 3 | Weather API (Open-Meteo) | Daily rainfall actuals + 16-day live forecast | Quarterly aggregation → `rain_mm`, `monsoon_idx`, `rain_lag1/2`, `monsoon_cum`, `extreme_wet`, `dry_spell`; climatology fallback offline |

## Models

Validated by **rolling-origin cross-validation** (train on the past, test the next quarter,
repeated over the last ~6 quarters); seasonal indices and weather climatology fallbacks are
fit on the **training slice only**, so no future data leaks into training or validation.

| Model | Method | Output | Validation |
| --- | --- | --- | --- |
| Decline risk | L2 logistic regression (25 features: lags, momentum, share, monsoon), **Platt-calibrated**; **optional LightGBM challenger** used only if it beats logistic on CV | P(next-quarter dispatch >50% below 4-q avg) | Pooled CV AUC + precision/recall/F1/Brier |
| Volume forecast | Champion of ridge / seasonal-naive / blend / **Croston-TSB**, with **P10/P50/P90** intervals | Next-quarter MT per supplier (+ downside/upside) | CV WAPE, champion auto-selected |
| Opportunity score | Size percentile × share headroom × growth | Ranked sourcing shortlist | — |
| Concentration | HHI, top-5 share per FY | Dependency trend | — |

Weather feature `rain_next_q`: training rows use rainfall **actuals** of the target
quarter; scoring rows use elapsed actuals + 16-day live forecast + climatology fill.
Croston-TSB is included because many mills dispatch intermittently (zero-heavy quarters).
`src/mlx.py` holds the calibration, Croston, quantile-interval and LightGBM helpers;
`src/ml_diagnostics.py` writes a feature-importance + reliability report to `docs/`.

## MLOps

Every pipeline run executes the full loop (`src/model_registry.py`):

| Step | Mechanism |
| --- | --- |
| Retraining | All history through the latest complete quarter, every run |
| Hyperparameter search | Grid over `l2 × lr` (classifier) and `l2` (ridge), selected on pooled rolling-CV AUC/WAPE |
| Champion selection | Best forecaster of {ridge, seasonal-naive, blend, croston-TSB} by CV WAPE |
| Calibration | Platt scaling on out-of-fold predictions; precision/recall/F1/Brier reported in `model_metrics.json` |
| Registration | Serializable logistic model registered every run (the promotion-gate check is currently simplified — see note) |
| Versioning | Weights per run in `data/models/model_v{N}.json`; audit trail in `registry.jsonl` |
| Monitoring | `ml_runs` warehouse table — chart model quality over time in Metabase |

> **Note:** the champion-vs-previous **promotion gate** was simplified during the model
> upgrade — every run currently registers. Re-adding the AUC-regression gate around the new
> CV score is a small follow-up.

Azure mapping: registry → Azure ML model registry; tuning → Azure ML sweep jobs;
`ml_runs` monitoring → Azure ML job metrics.

### CI/CD (GitHub Actions — `.github/workflows/mlops.yml`)

| Trigger | Behaviour |
| --- | --- |
| Push to `src/**` | Full pipeline runs as a test gate; DQ or pipeline failure fails the build |
| Mon 07:00 IST (cron) | Scheduled retrain: tune → gate promotion → commit refreshed registry, gold outputs, brief and dashboard data |
| Manual dispatch | Same as scheduled; brief uploaded as a build artifact |

Note: scheduled runs execute on GitHub-hosted runners — keep the repository
**private**; for strict in-house processing use n8n/cron instead and keep
Actions for CI only.

## Architecture

**Local production stack — detailed pipeline (current):**

```mermaid
flowchart LR
  subgraph EXTRACT["Extract → bronze"]
    E1[MIR field data]
    E2[ERP purchases]
    E3[Open-Meteo rainfall<br/>+ 16-day forecast]
  end
  subgraph TRANSFORM["Transform → silver"]
    T1[clean · join · log-MAD winsorize]
    T2[weather features:<br/>lag / monsoon_cum / extreme flags]
  end
  DQ[DQ gates<br/>fail-fast]
  WH[(SQLite / Postgres<br/>warehouse)]
  subgraph ML["ML → gold  (src/ml_train_score.py + mlx.py)"]
    M1[rolling-origin CV<br/>+ hyperparam search]
    M2[decline risk:<br/>logistic → Platt calib<br/>· optional LightGBM]
    M3[forecast champion:<br/>ridge / naive / blend / Croston<br/>+ P10/P50/P90]
    M4[opportunity score · HHI]
  end
  REG[model registry<br/>versions + ml_runs]
  DIAG[ml_diagnostics.py<br/>importance + reliability]
  AG[agent_brief.py<br/>weekly sourcing brief]
  DASH[React / HTML dashboard<br/>+ analysis dashboard]

  E1 & E2 & E3 --> T1 --> T2 --> DQ --> WH
  WH --> M1 --> M2 & M3
  WH --> M4
  M2 & M3 & M4 --> REG
  WH --> AG --> DASH
  WH --> DIAG
```

**Local production stack (original high-level render):**

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
  mlx.py          numpy-only ML helpers: calibration, Croston-TSB, quantile intervals,
                  rolling-CV, robust cap, optional LightGBM challenger
  ml_diagnostics.py  read-only feature-importance + reliability report
tests/          pytest: mlx helpers, transform integration, import smoke test
data/raw        input CSVs          data/bronze|silver|gold  pipeline layers
frontend/       React dashboard (Vite + recharts)
dashboard/      HTML fallback dashboard + standalone analysis dashboard
docs/           architecture diagrams, approach document, generated reports
```

## Scheduling

```
cron:  0 7 * * MON  python3 src/pipeline.py data/raw
n8n:   Schedule trigger → Execute Command → Read File (brief) → Send Email / Teams
```

---

*Internal — contains supplier-identifying data. Keep repository private.*
