"""STAGE 7 — AGENT. Azure equivalent: Azure AI Foundry agent (in-tenant LLM).

Reads the scored tables from the SQL warehouse and writes the weekly sourcing
brief in plain language: what happened, why each supplier is flagged, and what
to do. Locally this uses deterministic reasoning templates; in Azure the same
inputs go to an AI Foundry agent for richer language — the interface (SQL in,
markdown brief out) is identical, so it is a drop-in swap.
"""
import json
import os
import sqlite3

import pandas as pd

from common import DB_PATH, GOLD, ROOT, banner


def reason_for(r):
    """Plain-language root-cause hypothesis for one flagged supplier."""
    parts = []
    if r["latest_q_dispatch_MT"] == 0:
        parts.append(f"supplied nothing last quarter after averaging "
                     f"{r['trailing_4q_avg_MT']:.0f} MT/quarter")
    elif r["latest_q_dispatch_MT"] < 0.5 * r["trailing_4q_avg_MT"]:
        drop = (1 - r["latest_q_dispatch_MT"] / r["trailing_4q_avg_MT"]) * 100
        parts.append(f"dispatch fell {drop:.0f}% below its 4-quarter average")
    else:
        parts.append("momentum is weakening vs its own trend")
    if r["our_share_pct"] == 0:
        parts.append("we currently take none of its output — likely competitor capture "
                     "or the mill paused production")
    elif r["our_share_pct"] < 15:
        parts.append(f"our share is only {r['our_share_pct']:.0f}%, so the volume at risk "
                     "is mostly upside we could defend")
    return "; ".join(parts)


def action_for(r):
    if r["latest_q_dispatch_MT"] == 0:
        return "Field visit this week: confirm mill status (closed / seasonal / competitor contract)."
    if r["our_share_pct"] > 30:
        return "Priority call: we depend on this mill — secure next quarter's volume now."
    return "Contact the mill; if capacity exists, negotiate offtake before a competitor locks it."


def run():
    banner("AGENT", "Composing the weekly sourcing brief from the warehouse")
    con = sqlite3.connect(DB_PATH)
    watch = pd.read_sql("SELECT * FROM watchlist_decline_risk ORDER BY decline_risk DESC", con)
    opps = pd.read_sql("SELECT * FROM vw_top_opportunities", con)
    conc = pd.read_sql("SELECT * FROM concentration_risk ORDER BY fiscal_year", con)
    con.close()
    m = json.loads(open(os.path.join(GOLD, "model_metrics.json")).read())

    crit = watch[watch["risk_band"] == "Critical"].head(10)
    mod = watch[watch["risk_band"] == "Moderate"]
    lines = [
        f"# Supply Radar — Weekly Sourcing Brief",
        f"*Scored quarter: {m['latest_quarter']} · predictions for {m['next_quarter']} · "
        f"generated {pd.Timestamp.today():%d %b %Y}*",
        "",
        "## Headline",
        f"Of {m['n_suppliers_scored']} active suppliers, **{m['n_critical']} are at critical "
        f"risk** of a sharp supply drop next quarter and {m['n_moderate']} at moderate risk "
        f"(model AUC {m['decline_auc']} on held-out quarters). "
        f"Top-5 supplier dependency is {conc['top5_share_pct'].iloc[-1]}% "
        f"({'improving' if conc['top5_share_pct'].iloc[-1] < conc['top5_share_pct'].iloc[0] else 'worsening'} "
        f"from {conc['top5_share_pct'].iloc[0]}% in FY{int(conc['fiscal_year'].iloc[0])}).",
        "",
        "## Act this week — critical decline risks",
        "",
    ]
    for i, (_, r) in enumerate(crit.iterrows(), 1):
        lines += [f"**{i}. {r['supplier']}** ({r['region']}) — risk {r['decline_risk']*100:.0f}%",
                  f"   - Why: {reason_for(r)}.",
                  f"   - Expected next quarter: {r['expected_next_q_MT']:.0f} MT "
                  f"(vs {r['trailing_4q_avg_MT']:.0f} MT normal).",
                  f"   - Action: {action_for(r)}", ""]
    lines += ["## Grow here — top sourcing opportunities", ""]
    for i, (_, r) in enumerate(opps.head(5).iterrows(), 1):
        growth = f", growing {r['yoy_growth_pct']:.0f}% YoY" if pd.notna(r['yoy_growth_pct']) else ""
        lines.append(f"{i}. **{r['supplier']}** ({r['region']}): dispatched "
                     f"{r['dispatched_MT']:,.0f} MT{growth}, our share just "
                     f"{r['our_share_pct']}% — {r['untapped_MT']:,.0f} MT untapped.")
    lines += ["", "---",
              f"*Every flagged item needs an owner, root cause and resolution date before "
              f"the next review (per Finance's exception-review standard). Forecast model in "
              f"use: {m['forecast_champion']} (WAPE {m['forecast_wape_pct']}%).*"]

    out = os.path.join(ROOT, "docs", "weekly_sourcing_brief.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"  -> docs/weekly_sourcing_brief.md ({len(crit)} critical items, "
          f"{len(opps.head(5))} opportunities)")
    return out


if __name__ == "__main__":
    run()
