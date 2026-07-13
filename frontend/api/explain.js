// Vercel serverless function — POST /api/explain
// Per-supplier risk explanation. The LLM key stays server-side (process.env.LLM_API_KEY).
// Falls back to a deterministic template when no key is set or the call fails.
const GROQ_URL = (process.env.LLM_BASE_URL || "https://api.groq.com/openai/v1") + "/chat/completions";
const MODEL = process.env.LLM_MODEL || "llama-3.3-70b-versatile";

const num = (v, d = 0) => { const n = parseFloat(v); return Number.isFinite(n) ? n : d; };

function template(f) {
  const name = f.supplier || "This mill";
  const band = f.risk_band || "flagged";
  const risk = Math.round(num(f.decline_risk) * 100);
  const avg = num(f.trailing_4q_avg_MT), last = num(f.latest_q_dispatch_MT);
  const share = f.our_share_pct, pt = num(f.forecast_next_q_MT);
  const p10 = f.forecast_p10_MT, p90 = f.forecast_p90_MT;
  const why = [];
  if (avg && last < 0.5 * avg) why.push(`last-quarter dispatch dropped sharply below its ${Math.round(avg)} MT average`);
  else if (band === "Critical") why.push("momentum is weakening fast versus its own trend");
  else why.push("its trend is softening");
  if (share !== "" && share != null && num(share) < 15) why.push(`we take only ${Math.round(num(share))}% of its output`);
  let rng = "";
  if (p10 !== "" && p10 != null && p90 !== "" && p90 != null)
    rng = ` (likely ${Math.round(num(p10))}-${Math.round(num(p90))} MT)`;
  return `${name} is ${band} risk (~${risk}%): ${why.join("; ")}. Expect roughly ${Math.round(pt)} MT next quarter${rng}.`;
}

function readBody(req) {
  if (req.body && typeof req.body === "object") return Promise.resolve(req.body);
  return new Promise((resolve) => {
    let d = ""; req.on("data", (c) => (d += c));
    req.on("end", () => { try { resolve(JSON.parse(d || "{}")); } catch { resolve({}); } });
  });
}

module.exports = async (req, res) => {
  if (req.method !== "POST") { res.status(405).json({ error: "POST only" }); return; }
  const f = await readBody(req);
  const key = process.env.LLM_API_KEY;
  if (!key) { res.status(200).json({ text: template(f), mode: "template" }); return; }
  try {
    const prompt =
      "You are Vaighai's procurement intelligence agent (coco-coir buying, Tamil Nadu). " +
      "In 2-3 short sentences for a purchase manager, say whether this mill is at risk next quarter, " +
      "WHY, and what volume to expect. Use ONLY these facts, do not invent numbers. Be direct.\n\nFACTS:\n" +
      JSON.stringify(f);
    const r = await fetch(GROQ_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model: MODEL, messages: [{ role: "user", content: prompt }], temperature: 0.3 }),
    });
    const j = await r.json();
    const text = j?.choices?.[0]?.message?.content?.trim() || template(f);
    res.status(200).json({ text, mode: "llm" });
  } catch (e) {
    res.status(200).json({ text: template(f), mode: "template" });
  }
};
