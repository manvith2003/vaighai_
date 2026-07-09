"""AGENT — weekly sourcing brief from the warehouse.

Two modes, chosen automatically:
  LLM mode      when LLM_API_KEY is set (any OpenAI-compatible endpoint;
                default is Groq's free tier). The LLM receives ONLY aggregated
                facts (no row-level raw data) and writes the brief.
  Template mode offline fallback — deterministic reasoning templates.

Production swap: same facts go to Azure AI Foundry (in-tenant) instead.
"""
import json
import os
import urllib.request

import pandas as pd

import config
from utils import GOLD, ROOT, banner, wh_connect


def reason_for(r):
    parts = []
    if r["latest_q_dispatch_MT"] == 0:
        parts.append(f"supplied nothing last quarter after averaging "
                     f"{r['trailing_4q_avg_MT']:.0f} MT/quarter")
    elif r["latest_q_dispatch_MT"] < 0.5 * r["trailing_4q_avg_MT"]:
        drop = (1 - r["latest_q_dispatch_MT"] / r["trailing_4q_avg_MT"]) * 100
        parts.append(f"dispatch fell {drop:.0f}% below its 4-quarter average")
    else:
        parts.append("momentum weakening vs its own trend")
    if r["our_share_pct"] == 0:
        parts.append("we take none of its output — possible competitor capture or pause")
    elif r["our_share_pct"] < 15:
        parts.append(f"our share is only {r['our_share_pct']:.0f}%")
    return "; ".join(parts)


def action_for(r):
    if r["latest_q_dispatch_MT"] == 0:
        return "Field visit this week: confirm mill status (closed / seasonal / competitor contract)."
    if r["our_share_pct"] > 30:
        return "Priority call: we depend on this mill — secure next quarter's volume now."
    return "Contact the mill; negotiate offtake before a competitor locks the capacity."


def gather_facts():
    con, backend = wh_connect()
    watch = pd.read_sql("SELECT * FROM watchlist_decline_risk ORDER BY decline_risk DESC", con)
    opps = pd.read_sql("SELECT * FROM vw_top_opportunities", con)
    conc = pd.read_sql("SELECT * FROM concentration_risk ORDER BY fiscal_year", con)
    if backend == "sqlite":
        con.close()
    m = json.load(open(os.path.join(GOLD, "model_metrics.json")))
    crit = watch[watch["risk_band"] == "Critical"].head(10)
    return m, crit, opps, conc


def template_brief(m, crit, opps, conc):
    lines = [
        "# Supply Radar — Weekly Sourcing Brief",
        f"*Scored quarter: {m['latest_quarter']} · predictions for {m['next_quarter']} · "
        f"generated {pd.Timestamp.today():%d %b %Y}*", "",
        "## Headline",
        f"Of {m['n_suppliers_scored']} active suppliers, **{m['n_critical']} are at critical "
        f"risk** of a sharp supply drop next quarter and {m['n_moderate']} at moderate risk "
        f"(model AUC {m['decline_auc']}). Top-5 supplier dependency is "
        f"{conc['top5_share_pct'].iloc[-1]}% (was {conc['top5_share_pct'].iloc[0]}% in "
        f"FY{int(conc['fiscal_year'].iloc[0])}).", "",
        "## Act this week — critical decline risks", "",
    ]
    for i, (_, r) in enumerate(crit.iterrows(), 1):
        lines += [f"**{i}. {r['supplier']}** ({r['region']}) — risk {r['decline_risk']*100:.0f}%",
                  f"   - Why: {reason_for(r)}.",
                  f"   - Expected next quarter: {r['expected_next_q_MT']:.0f} MT "
                  f"(vs {r['trailing_4q_avg_MT']:.0f} MT normal).",
                  f"   - Action: {action_for(r)}", ""]
    lines += ["## Grow here — top sourcing opportunities", ""]
    for i, (_, r) in enumerate(opps.head(5).iterrows(), 1):
        growth = f", growing {r['yoy_growth_pct']:.0f}% YoY" if pd.notna(r["yoy_growth_pct"]) else ""
        lines.append(f"{i}. **{r['supplier']}** ({r['region']}): {r['dispatched_MT']:,.0f} MT "
                     f"dispatched{growth}, our share {r['our_share_pct']}% — "
                     f"{r['untapped_MT']:,.0f} MT untapped.")
    lines += ["", "---", f"*Every flagged item needs an owner, root cause and resolution date. "
              f"Forecast model: {m['forecast_champion']} (WAPE {m['forecast_wape_pct']}%).*"]
    return "\n".join(lines)


def llm_brief(m, crit, opps, conc):
    facts = {
        "metrics": m,
        "critical_suppliers": [
            {"supplier": r["supplier"], "region": r["region"],
             "risk_pct": round(r["decline_risk"] * 100),
             "latest_q_MT": r["latest_q_dispatch_MT"], "normal_4q_avg_MT": r["trailing_4q_avg_MT"],
             "our_share_pct": r["our_share_pct"], "expected_next_q_MT": r["expected_next_q_MT"]}
            for _, r in crit.iterrows()],
        "top_opportunities": opps.head(5).to_dict(orient="records"),
        "dependency_top5_share_by_fy": conc[["fiscal_year", "top5_share_pct"]].to_dict(orient="records"),
    }
    prompt = (
        "You are the procurement intelligence agent for Vaighai, a coco-coir manufacturer in "
        "Tamil Nadu that buys coir from small mills. Write this week's sourcing brief in "
        "markdown for purchase managers. Use ONLY the facts given — do not invent numbers. "
        "Structure: title, one-paragraph headline, 'Act this week' section with each critical "
        "supplier (why flagged, expected volume, one concrete action), 'Grow here' section with "
        "the opportunities, closing line on ownership of flagged items. Be direct and brief.\n\n"
        f"FACTS:\n{json.dumps(facts, default=str)}"
    )
    req = urllib.request.Request(
        f"{config.LLM_BASE_URL}/chat/completions",
        data=json.dumps({"model": config.LLM_MODEL,
                         "messages": [{"role": "user", "content": prompt}],
                         "temperature": 0.3}).encode(),
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"]


def run():
    banner("AGENT", "Composing the weekly sourcing brief")
    m, crit, opps, conc = gather_facts()
    mode = "llm" if config.LLM_API_KEY else "template"
    if mode == "llm":
        try:
            text = llm_brief(m, crit, opps, conc)
            print(f"  written by LLM ({config.LLM_MODEL} via {config.LLM_BASE_URL})")
        except Exception as e:
            print(f"  LLM call failed ({type(e).__name__}) — falling back to templates")
            text, mode = template_brief(m, crit, opps, conc), "template"
    else:
        text = template_brief(m, crit, opps, conc)
    if mode == "template":
        print("  written by deterministic templates (set LLM_API_KEY for LLM mode)")
    out = os.path.join(ROOT, "docs", "weekly_sourcing_brief.md")
    with open(out, "w") as f:
        f.write(text)
    print(f"  -> docs/weekly_sourcing_brief.md ({len(crit)} critical items)")


if __name__ == "__main__":
    run()
