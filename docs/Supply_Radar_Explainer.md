# Supply Radar — How the Solution Works (Use Case 6)

*A plain-English walkthrough you can present. Covers what problem we solved, how we arrived at the approach, how the local pipeline works stage by stage, what each model does, and what the graphs show.*

---

## 1. The problem in one line

Vaighai (coco-coir manufacturer, Tamil Nadu) buys coir from many small mills. The sales side has a clear market view of *customers*; the **buying side had no equivalent view of *suppliers***. Supply Radar is that mirror image: for every mill, **is it growing or shrinking, how much of its output do we actually capture, and where should we source more?**

## 2. The core insight (why the data is set up the way it is)

We have two very different records of the same mills, and the whole solution comes from overlaying them:

- **PURCHASE (ERP)** — what we *actually received*. Machine-recorded, reliable.
- **MIR (Market Intelligence Report)** — field staff's *estimates* per mill: how much it **produced**, how much it **dispatched to everyone**, and how much **we took**.

PURCHASE alone tells us what we bought but nothing about the mill's total size. MIR tells us the mill's size and our slice of it. Put together:

> **Our share of a mill = our offtake ÷ the mill's total dispatch.**
> A big mill where our share is small = a sourcing opportunity. A mill whose dispatch is collapsing = a supply risk.

That single ratio is the spine of the entire product. We deliberately **keep the estimate-vs-actual gap** rather than "fixing" it — the gap is a useful reconciliation signal, not an error.

## 3. How we arrived at the solution (the reasoning)

1. **Fix the grain.** Both sources were messy and named the same mill differently ("A J Coir, Kedimedu" vs "AJ Coirs"). We reconciled names into one clean `supplier`, then aggregated everything to a single grain: **supplier × region × fiscal quarter**. This is the atom every metric is built on.
2. **Decide what "market" means.** ~70% of raw purchase volume is Vaighai moving its *own* stock between units (internal transfers). Those aren't market purchases, so they are **excluded** from all market analysis.
3. **Turn the ratio into decisions.** Once we had share-of-mill per quarter, four questions naturally followed — *who's about to shrink?*, *how much will they dispatch next quarter?*, *where's the untapped volume?*, *how dependent are we on a few mills?* — which became the four models.
4. **Make it explainable and repeatable.** A weekly brief translates the numbers into "call this mill, visit that one." The whole thing runs as one command locally, with a clean one-to-one path to Azure for later.

## 4. The architecture — local stack (the "first architecture")

One command, `python3 src/pipeline.py data/raw`, runs seven stages in order (this is `pipeline.py`). Data flows through **bronze → silver → gold** layers:

```
EXTRACT (3 sources) → TRANSFORM → VALIDATE → WAREHOUSE → ML → AGENT → DASHBOARD
     bronze             silver     DQ gates    SQLite     gold  brief    React / HTML
```

**Extract (→ bronze).** Three extractors pull raw inputs unchanged: MIR field data, ERP purchases, and daily rainfall from the Open-Meteo weather API (with an offline climatology fallback). Weather matters because monsoon rain drives coir drying and mill output.

**Transform (→ silver)** — `transform_supply_panel.py`. This is where the insight becomes a table:
- MIR: drop future-dated rows, aggregate to the supplier × region × quarter grain.
- Purchases: drop internal transfers, aggregate to the same grain.
- **Outer join** the two → the "supply panel" (`in_mir` / `in_purchase` flags show which side each row came from).
- **Winsorize** obvious field-entry outliers so one fat-fingered estimate can't distort everything.
- Compute `our_share_of_dispatch_pct` and roll rainfall up to fiscal quarters.

**Validate (DQ gates)** — `dq_validate.py`. Hard, fail-fast checks (no negative tonnages, keys present, shares within 0–100, etc.). If data quality fails, the pipeline **stops** rather than shipping bad numbers.

**Warehouse (→ SQL)** — `load_warehouse.py`. Silver and gold tables load into **SQLite locally** (swappable to PostgreSQL). Reporting reads from SQL views, exactly like it will in production.

**ML (→ gold)** — `ml_train_score.py`. Trains and scores the four models (Section 5), writing ranked gold tables.

**Agent** — `agent_brief.py`. Turns the gold tables into a human weekly sourcing brief (Section 6).

**Dashboard** — `report_dashboard.py` builds the React/HTML dashboard; Metabase can sit on the same warehouse views for BI.

## 5. The four models (what each one answers)

All models are validated on **held-out quarters** every run — we never let future data leak into training, so the accuracy numbers are honest.

| Model | Question it answers | How | Output |
| --- | --- | --- | --- |
| **Decline risk** | "Which mills will sharply drop next quarter?" | Logistic regression on 19 features (lags, momentum, our share, monsoon). Target = next-quarter dispatch falling >50% below the trailing 4-quarter average. | Risk % + band (Low / Moderate / Critical), AUC ≈ 0.76 |
| **Volume forecast** | "How much will each mill dispatch next quarter?" | Auto-picks the best of ridge regression / seasonal-naïve / a blend, chosen by backtest error (WAPE). | Next-quarter MT per supplier |
| **Opportunity score** | "Where should we source more?" | Size percentile × share headroom × recent growth. | Ranked sourcing shortlist + untapped MT |
| **Concentration** | "How dependent are we on a few mills?" | HHI and top-5 supplier share per fiscal year. | Dependency trend over time |

The forecast being a *champion selection* (not one fixed model) is deliberate: for suppliers with strong seasonality, the simple seasonal-naïve baseline often wins, and the code picks whichever is actually most accurate on backtest.

## 6. The agent — the weekly sourcing brief

`agent_brief.py` reads only the **aggregated** gold facts (never raw rows) and writes a manager-ready brief: a headline, "Act this week" critical-risk suppliers (each with *why flagged*, expected volume, and one concrete action), "Grow here" opportunities, and a 16-day monsoon outlook. It runs in **template mode** offline (deterministic text) or **LLM mode** if an API key is set. In production the same facts go to Azure AI Foundry. This is what makes the analytics *actionable* rather than just charts.

## 7. Why it's trustworthy (the MLOps loop)

Every run retrains on all history through the latest complete quarter, does a small hyperparameter search, and logs the run with a versioned audit trail (`registry.jsonl`) plus an `ml_runs` table you can chart over time. So model quality is monitored, not assumed.

## 8. What the graphs show (in the dashboard)

Open `dashboard/supply_radar_analysis.html` and load `supply_radar_joined.csv`. Talking points per chart:

- **Market vs. our buying, by year** — the grey bars (whole-market dispatch) tower over the green bars (what we bought). That visible gap *is* the business case: most of the market is not yet ours.
- **Sourcing opportunity map** — every mill as a bubble; the **bottom-right** (big dispatchers we barely buy from) are the prospects. Red bubbles = under 15% share.
- **Top suppliers by dispatch** — the biggest mills that year, coloured by our share; red/amber bars among the leaders are the priority targets.
- **Concentration risk** — top-5 supplier share and HHI over time; rising = we're getting more dependent on fewer mills (a risk to flag).
- **Our share distribution** — a histogram piled up at the low end means broad headroom to grow offtake.
- **Estimate vs. actual reconciliation** — field offtake estimate next to actual purchases per year; close bars mean field reporting is reliable.
- **Weather impact** — live quarterly rainfall overlaid on dispatch per region, with a correlation readout (monsoon quarters vs dispatch dips).
- **Top 10 opportunities table** — the shortlist, ranked by untapped MT, ready to hand to the buying team.

## 9. Next step — Azure migration (after the local stack)

The local stack maps one-to-one to Azure, so nothing gets rebuilt — only re-hosted:

| Local | Azure |
| --- | --- |
| cron / n8n | Logic Apps |
| `extract_*.py` | Data Factory |
| transform + DQ | Azure Functions |
| SQLite / PostgreSQL | Azure SQL Database |
| `ml_train_score.py` | Azure ML (batch endpoint + registry) |
| `agent_brief.py` | Azure AI Foundry (in-tenant) |
| React / Metabase | Power BI |
| n8n alerts | Power Automate → Teams |

## 10. How to demo it

```bash
pip install -r requirements.txt
python3 src/pipeline.py data/raw      # ~30s: builds warehouse, models, brief, dashboard
```
Outputs: `data/supply_radar.db` (warehouse), `docs/weekly_sourcing_brief.md` (the brief),
`docs/validation_report.md` (DQ), and the dashboard. For a no-setup visual, just open
`dashboard/supply_radar_analysis.html` and load the joined CSV.

---
*Internal — contains supplier-identifying data. Keep the repository private.*
