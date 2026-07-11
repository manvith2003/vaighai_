import { useEffect, useMemo, useRef, useState } from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

/* ---------- helpers ---------- */
const fmt = (n) =>
  n === "" || n == null || Number.isNaN(Number(n))
    ? "–"
    : Number(n).toLocaleString("en-IN", { maximumFractionDigits: 0 });

function useCountUp(target, ms = 1200) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (typeof target !== "number") { setV(target); return; }
    let raf, t0;
    const step = (t) => {
      if (!t0) t0 = t;
      const p = Math.min((t - t0) / ms, 1);
      setV(target * (1 - Math.pow(1 - p, 3)));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

/* 3D tilt-on-hover card (optionally fires onEnter for a plain-English AI insight) */
function Tilt({ children, className = "", style, onEnter }) {
  const ref = useRef(null);
  const onMove = (e) => {
    const el = ref.current;
    const r = el.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width - 0.5;
    const y = (e.clientY - r.top) / r.height - 0.5;
    el.style.transform =
      `perspective(900px) rotateX(${(-y * 5).toFixed(2)}deg) rotateY(${(x * 6).toFixed(2)}deg) translateY(-3px)`;
  };
  const onLeave = () => { ref.current.style.transform = ""; };
  return (
    <div ref={ref} className={`card tilt ${className}`} style={style}
         onMouseMove={onMove} onMouseLeave={onLeave} onMouseEnter={onEnter}>
      {children}
    </div>
  );
}

/* tiny markdown renderer for the agent's brief (headings, bullets, **bold**) */
function MarkdownLite({ text }) {
  if (!text) return null;
  const bold = (s) =>
    s.split(/(\*\*[^*]+\*\*)/g).map((p, i) =>
      p.startsWith("**") && p.endsWith("**") ? <b key={i}>{p.slice(2, -2)}</b> : p);
  return (
    <div className="md">
      {text.split("\n").map((ln, i) => {
        const t = ln.trim();
        if (!t) return null;
        if (t.startsWith("## ")) return <h4 key={i}>{bold(t.slice(3))}</h4>;
        if (t.startsWith("# ")) return <h3 key={i}>{bold(t.slice(2))}</h3>;
        if (t.startsWith("* ") || t.startsWith("- "))
          return <div key={i} className="mdli">• {bold(t.slice(2))}</div>;
        return <p key={i}>{bold(t)}</p>;
      })}
    </div>
  );
}

function Kpi({ value, label, tone = "", decimals = 0, suffix = "" }) {
  const v = useCountUp(typeof value === "number" ? value : 0);
  return (
    <Tilt className={`kpi ${tone}`}>
      <div className="v">
        {typeof value === "number" ? v.toFixed(decimals) : value}{suffix}
      </div>
      <div className="l">{label}</div>
    </Tilt>
  );
}

const tipStyle = {
  background: "rgba(10,15,35,.92)", border: "1px solid rgba(255,255,255,.15)",
  borderRadius: 12, fontSize: 12.5, color: "#eef2ff",
};

/* live agent proxy (keeps the LLM key server-side) — run: python3 src/agent_server.py */
const AGENT_URL = "http://localhost:8000/api/explain";
const INSIGHT_URL = "http://localhost:8000/api/insight";

/* ---------- app ---------- */
export default function App() {
  const [data, setData] = useState(null);
  const [q, setQ] = useState("");

  /* ---- live agent-on-hover state (supplier rows AND charts) ---- */
  const [ai, setAi] = useState({ open: false, kind: "", title: "", band: "", text: "", loading: false, facts: null });
  const [notifOpen, setNotifOpen] = useState(true);
  const aiCache = useRef({});
  const hoverT = useRef(null);
  const hover = (fn) => { clearTimeout(hoverT.current); hoverT.current = setTimeout(fn, 320); };
  const leave = () => clearTimeout(hoverT.current);

  async function askAgent(r) {
    const key = "sup:" + r.supplier;
    const cached = aiCache.current[key];
    setAi({ open: true, kind: "supplier", title: r.supplier, band: r.risk_band, facts: r,
            text: cached || "", loading: !cached });
    if (cached) return;
    try {
      const res = await fetch(AGENT_URL, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          supplier: r.supplier, region: r.region,
          latest_q_dispatch_MT: r.latest_q_dispatch_MT, trailing_4q_avg_MT: r.trailing_4q_avg_MT,
          our_share_pct: r.our_share_pct, decline_risk: r.decline_risk, risk_band: r.risk_band,
          expected_next_q_MT: r.expected_next_q_MT, forecast_next_q_MT: r.forecast_next_q_MT,
          forecast_p10_MT: r.forecast_p10_MT, forecast_p90_MT: r.forecast_p90_MT,
          next_quarter: data?.meta?.next_quarter,
        }),
      });
      const j = await res.json();
      aiCache.current[key] = j.text;
      setAi((a) => (a.title === r.supplier ? { ...a, text: j.text, loading: false } : a));
    } catch (e) {
      setAi((a) => (a.title === r.supplier
        ? { ...a, loading: false, text: "Agent offline — start it with:  python3 src/agent_server.py" }
        : a));
    }
  }

  async function askInsight(title, summary) {
    const key = "ins:" + title;
    const cached = aiCache.current[key];
    setAi({ open: true, kind: "insight", title, band: "", facts: null,
            text: cached || "", loading: !cached });
    if (cached) return;
    try {
      const res = await fetch(INSIGHT_URL, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, summary }),
      });
      const j = await res.json();
      aiCache.current[key] = j.text;
      setAi((a) => (a.title === title ? { ...a, text: j.text, loading: false } : a));
    } catch (e) {
      setAi((a) => (a.title === title
        ? { ...a, loading: false, text: "Agent offline — start it with:  python3 src/agent_server.py" }
        : a));
    }
  }

  useEffect(() => {
    fetch("./dashboard_data.json").then((r) => r.json()).then(setData)
      .catch(() => setData({ error: true }));
  }, []);

  const watch = useMemo(() => {
    if (!data?.watchlist) return [];
    const needle = q.toLowerCase();
    return data.watchlist.filter(
      (r) => !needle || `${r.supplier} ${r.region}`.toLowerCase().includes(needle)
    );
  }, [data, q]);

  if (!data) return <div className="wrap hero"><p>Loading Supply Radar…</p></div>;
  if (data.error)
    return (
      <div className="wrap hero">
        <h1>Supply Radar</h1>
        <p>dashboard_data.json not found — run <b>python3 src/pipeline.py data/raw</b> first.</p>
      </div>
    );

  const m = data.meta;

  /* short factual summaries the agent turns into plain English on chart hover */
  const sums = {
    trend: (() => {
      const t = data.trend || []; const last = t[t.length - 1] || {};
      return `Whole-market dispatch vs our offtake vs actual purchases over time. Latest quarter ${last.label}: market ${fmt(last.dispatched_MT)} MT, our offtake ${fmt(last.offtake_MT)} MT, purchased ${fmt(last.purchased_MT)} MT.`;
    })(),
    region: `Dispatch by region (MT): ` +
      [...(data.region_mix || [])].sort((a, b) => b.dispatched_MT - a.dispatched_MT)
        .slice(0, 4).map((x) => `${x.region} ${fmt(x.dispatched_MT)}`).join(", ") + ".",
    conc: (() => {
      const c = data.concentration || []; const a = c[0] || {}, z = c[c.length - 1] || {};
      return `Top-5 supplier share of our purchases went from ${a.top5_share_pct}% (FY${a.fiscal_year}) to ${z.top5_share_pct}% (FY${z.fiscal_year}); HHI ${a.hhi} to ${z.hhi}.`;
    })(),
    opps: `Biggest mills where our share is low: ` +
      (data.opportunities || []).slice(0, 4).map((o) => `${o.supplier} ${fmt(o.dispatched_MT)} MT at ${Number(o.share_pct).toFixed(0)}% share`).join(", ") + ".",
    monsoon: `Next-16-day rainfall vs normal: ` +
      (data.monsoon_outlook || []).map((o) => `${o.region} ${o.vs_normal_x}x`).join(", ") + ".",
  };

  const critical = (data.watchlist || [])
    .filter((r) => r.risk_band === "Critical")
    .sort((a, b) => b.decline_risk - a.decline_risk);

  return (
    <>
      <div className="bg" />
      <div className="orb orb1" /><div className="orb orb2" /><div className="orb orb3" />
      <div className="shape cube" /><div className="shape ring" /><div className="shape tri" />

      <div className="wrap">
        <header className="hero">
          <span className="badge">Vaighai · Procurement Intelligence · Use Case 6</span>
          <h1>Supply Radar</h1>
          <p>
            Scored <b>{m.latest_quarter}</b> · predictions for <b>{m.next_quarter}</b> ·
            opportunity view FY{m.latest_complete_fy} · decline model AUC <b>{m.decline_auc}</b> on held-out quarters
          </p>
        </header>

        <section className="kpis">
          <Kpi value={m.n_suppliers_scored} label="active suppliers scored" />
          <Kpi value={m.n_critical} label="critical decline risk" tone="red" />
          <Kpi value={m.n_moderate} label="moderate decline risk" tone="amber" />
          <Kpi value={m.decline_auc} label="risk model AUC (holdout)" decimals={3} tone="green" />
          <Kpi value={m.forecast_wape_pct} label={`forecast WAPE · champion: ${m.forecast_champion}`} decimals={1} suffix="%" />
        </section>

        <div className="grid">
          {data.brief_md && (
            <Tilt className="full brief">
              <h2>🧠 This week's sourcing brief <span className="byai">written by the AI agent</span></h2>
              <div className="sub">Plain-English summary the agent generated from the numbers below.</div>
              <MarkdownLite text={data.brief_md} />
            </Tilt>
          )}

          <Tilt className="full" onEnter={() => hover(() => askInsight("Market vs Vaighai", sums.trend))}>
            <h2>Market vs Vaighai — quarterly volumes (MT)</h2>
            <div className="sub">Mill dispatch = whole market (MIR estimates) · offtake = our estimated take · purchased = actual receipts</div>
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={data.trend}>
                <defs>
                  <linearGradient id="gBlue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.5} />
                    <stop offset="100%" stopColor="#60a5fa" stopOpacity={0.03} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} />
                <XAxis dataKey="label" tick={{ fill: "#64748b", fontSize: 10 }} interval={2} />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} width={54} />
                <Tooltip contentStyle={tipStyle} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Area type="monotone" dataKey="dispatched_MT" name="Mill dispatch"
                      stroke="#60a5fa" strokeWidth={2} fill="url(#gBlue)" />
                <Area type="monotone" dataKey="offtake_MT" name="Our offtake"
                      stroke="#34d399" strokeWidth={2} fill="transparent" />
                <Area type="monotone" dataKey="purchased_MT" name="Purchased (actual)"
                      stroke="#fbbf24" strokeWidth={2} strokeDasharray="6 4" fill="transparent" />
              </AreaChart>
            </ResponsiveContainer>
          </Tilt>

          <Tilt className="full">
            <h2>Decline-risk watchlist — predicted for {m.next_quarter}</h2>
            <div className="sub">
              Risk = probability next-quarter dispatch falls &gt;50% below the supplier's trailing 4-quarter average ·
              <b style={{ color: "#a5c4ff" }}> hover a row for a live AI explanation</b>
            </div>
            <input className="search" placeholder="Filter supplier / region…"
                   value={q} onChange={(e) => setQ(e.target.value)} />
            <div className="scroll">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Supplier</th><th>Region</th>
                    <th className="num">Latest Q (MT)</th><th className="num">4-Q avg (MT)</th>
                    <th className="num">Share %</th><th className="num">Expected next Q (MT)</th>
                    <th>Risk</th><th>Band</th>
                  </tr>
                </thead>
                <tbody>
                  {watch.map((r) => (
                    <tr key={r.supplier}
                        className={ai.kind === "supplier" && ai.title === r.supplier ? "airow" : ""}
                        onMouseEnter={() => hover(() => askAgent(r))} onMouseLeave={leave}>
                      <td>{r.supplier}</td>
                      <td style={{ color: "#94a3b8" }}>{r.region}</td>
                      <td className="num">{fmt(r.latest_q_dispatch_MT)}</td>
                      <td className="num">{fmt(r.trailing_4q_avg_MT)}</td>
                      <td className="num">{r.our_share_pct === "" ? "–" : Number(r.our_share_pct).toFixed(0)}</td>
                      <td className="num">{fmt(r.expected_next_q_MT)}</td>
                      <td>
                        <span className="riskbar"><i style={{ width: `${r.decline_risk * 100}%` }} /></span>
                        {(r.decline_risk * 100).toFixed(0)}%
                      </td>
                      <td><span className={`pill ${r.risk_band}`}>{r.risk_band}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Tilt>

          <Tilt className="full" onEnter={() => hover(() => askInsight("Sourcing opportunities", sums.opps))}>
            <h2>Sourcing opportunities — FY{m.latest_complete_fy}</h2>
            <div className="sub">Big, growing mills where our share is low · score = size × headroom × growth</div>
            <div className="opps">
              {data.opportunities.slice(0, 8).map((o, i) => (
                <div className="opp" key={o.supplier}>
                  <div className="rank">#{i + 1}</div>
                  <div className="name">{o.supplier}</div>
                  <div className="reg">{o.region}</div>
                  <div className="row"><span>Dispatched</span><b>{fmt(o.dispatched_MT)} MT</b></div>
                  <div className="row"><span>Our share</span><b>{Number(o.share_pct).toFixed(1)}%</b></div>
                  <div className="row"><span>Untapped</span><b>{fmt(o.untapped_MT)} MT</b></div>
                  <div className="row"><span>Score</span><span className="score">{o.opportunity_score}</span></div>
                </div>
              ))}
            </div>
          </Tilt>

          <Tilt onEnter={() => hover(() => askInsight("Region mix", sums.region))}>
            <h2>Region mix — FY{m.latest_complete_fy}</h2>
            <div className="sub">Market dispatch vs our offtake by region (MT)</div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={data.region_mix}>
                <CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} />
                <XAxis dataKey="region" tick={{ fill: "#64748b", fontSize: 11 }} />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} width={54} />
                <Tooltip contentStyle={tipStyle} cursor={{ fill: "rgba(96,165,250,.06)" }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="dispatched_MT" name="Mill dispatch" fill="#60a5fa" radius={[6, 6, 0, 0]} />
                <Bar dataKey="offtake_MT" name="Our offtake" fill="#34d399" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Tilt>

          <Tilt onEnter={() => hover(() => askInsight("Supply-base concentration", sums.conc))}>
            <h2>Supply-base concentration</h2>
            <div className="sub">Top-5 supplier share of purchases — dependency risk trend</div>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={data.concentration}>
                <CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} />
                <XAxis dataKey="fiscal_year" tickFormatter={(v) => `FY${v}`}
                       tick={{ fill: "#64748b", fontSize: 11 }} />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} width={40} unit="%" />
                <Tooltip contentStyle={tipStyle} labelFormatter={(v) => `FY${v}`} />
                <Line type="monotone" dataKey="top5_share_pct" name="Top-5 share %"
                      stroke="#f87171" strokeWidth={2.5}
                      dot={{ r: 4, fill: "#f87171" }} activeDot={{ r: 6 }} />
              </LineChart>
            </ResponsiveContainer>
          </Tilt>

          {data.monsoon_outlook?.length > 0 && (
            <Tilt className="full" onEnter={() => hover(() => askInsight("Monsoon outlook", sums.monsoon))}>
              <h2>Monsoon outlook — next 16 days (live forecast)</h2>
              <div className="sub">Heavy rain = wet coir &amp; drying delays · dry = production runs freely</div>
              <div className="chips">
                {data.monsoon_outlook.map((o) => (
                  <div key={o.region}
                       className={`chip ${o.vs_normal_x > 1.3 ? "wet" : o.vs_normal_x < 0.7 ? "dry" : ""}`}>
                    <b>{o.region} · {o.vs_normal_x}× normal</b>
                    <span className="mm">{o.forecast_16d_mm} mm forecast (normal {o.normal_mm} mm)</span>
                  </div>
                ))}
              </div>
            </Tilt>
          )}
        </div>

        <div className="foot">
          Supply Radar · data refreshed each pipeline run · local stack (Postgres/SQLite + numpy ML + LLM agent) ·
          Azure-ready — <b>generated {new Date().toLocaleDateString("en-IN")}</b>
        </div>
      </div>

      {notifOpen && critical.length > 0 && (
        <div className="notif">
          <button className="aiclose" onClick={() => setNotifOpen(false)}>×</button>
          <div className="notifhead"><span className="bell">🔔</span> {critical.length} mills at critical decline risk</div>
          <ul className="notiflist">
            {critical.slice(0, 5).map((r) => (
              <li key={r.supplier}>
                <b>{r.supplier}</b> <span className="nreg">{r.region}</span>
                <span className="nrk">{(r.decline_risk * 100).toFixed(0)}%</span>
              </li>
            ))}
          </ul>
          <div className="notiffoot">Hover a table row or any chart for the agent's plain-English reason.</div>
        </div>
      )}

      {ai.open && (
        <div className="aipanel">
          <button className="aiclose" onClick={() => setAi((a) => ({ ...a, open: false }))}>×</button>
          <div className="aihead">
            <span className="aidot" /> {ai.kind === "supplier" ? "AI risk insight" : "AI chart insight"}
          </div>
          <div className="ainame">
            {ai.title} {ai.band && <span className={`pill ${ai.band}`}>{ai.band}</span>}
          </div>
          {ai.loading
            ? <div className="aiload">Asking the agent…</div>
            : <p className="aitext">{ai.text}</p>}
          {ai.kind === "supplier" && ai.facts && (
            <div className="airange">
              Forecast next Q: <b>{fmt(ai.facts.forecast_next_q_MT)} MT</b>
              {ai.facts.forecast_p10_MT !== "" && ai.facts.forecast_p10_MT != null &&
                ` · P10–P90 ${fmt(ai.facts.forecast_p10_MT)}–${fmt(ai.facts.forecast_p90_MT)} MT`}
            </div>
          )}
        </div>
      )}
    </>
  );
}
