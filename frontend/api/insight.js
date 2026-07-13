// Vercel serverless function — POST /api/insight
// Plain-English explanation of a chart. Key stays server-side; template fallback offline.
const GROQ_URL = (process.env.LLM_BASE_URL || "https://api.groq.com/openai/v1") + "/chat/completions";
const MODEL = process.env.LLM_MODEL || "llama-3.3-70b-versatile";

function template(f) {
  return `${f.title || "This chart"}: ${f.summary || ""}`.trim();
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
      "In ONE or TWO plain-English sentences, explain to a purchase manager what this chart is " +
      "telling them and why it matters. Use ONLY the facts given, do not invent numbers. Be direct.\n\n" +
      `CHART: ${f.title || ""}\nFACTS: ${f.summary || ""}`;
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
