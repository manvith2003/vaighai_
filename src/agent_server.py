"""Live agent proxy for the React dashboard — explains ONE supplier on hover.

Keeps the LLM API key server-side (never shipped to the browser). Stdlib only, no
extra dependencies. The frontend POSTs a supplier's aggregated facts to /api/explain
and gets back a 2-3 sentence plain-English risk + forecast note.

Run:   python3 src/agent_server.py          # serves http://localhost:8000
Uses Groq (config.LLM_*) when LLM_API_KEY is set; otherwise a deterministic template
so hover still works offline. The pipeline/data are unaffected — this only reads facts
sent by the browser.
"""
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config

PORT = 8000


def _num(v, default=0.0):
    try:
        if v in ("", None):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _template(f):
    name = f.get("supplier", "This mill")
    band = f.get("risk_band", "")
    risk = round(_num(f.get("decline_risk")) * 100)
    avg = _num(f.get("trailing_4q_avg_MT"))
    last = _num(f.get("latest_q_dispatch_MT"))
    share = f.get("our_share_pct")
    pt = _num(f.get("forecast_next_q_MT"))
    p10, p90 = f.get("forecast_p10_MT"), f.get("forecast_p90_MT")
    why = []
    if avg and last < 0.5 * avg:
        why.append(f"last-quarter dispatch dropped sharply below its {avg:.0f} MT four-quarter average")
    elif band == "Critical":
        why.append("momentum is weakening fast versus its own trend")
    else:
        why.append("its trend is softening")
    if share not in ("", None) and _num(share) < 15:
        why.append(f"we currently take only {_num(share):.0f}% of its output")
    rng = ""
    if p10 not in ("", None) and p90 not in ("", None):
        rng = f" (likely {_num(p10):.0f}–{_num(p90):.0f} MT range)"
    return (f"{name} is {band or 'flagged'} risk (~{risk}%): " + "; ".join(why) +
            f". Expect roughly {pt:.0f} MT next quarter{rng}.")


def _insight_template(f):
    return f"{f.get('title','This chart')}: {f.get('summary','')}".strip()


def _llm_insight(f):
    prompt = ("You are Vaighai's procurement intelligence agent (coco-coir buying, Tamil Nadu). "
              "In ONE or TWO plain-English sentences, explain to a purchase manager what this "
              "chart is telling them and why it matters. Use ONLY the facts given — do not invent "
              "numbers. Be direct.\n\n"
              f"CHART: {f.get('title','')}\nFACTS: {f.get('summary','')}")
    return _call_groq(prompt)


def _llm(f):
    prompt = ("You are Vaighai's procurement intelligence agent (coco-coir buying, Tamil Nadu). "
              "In 2-3 short sentences for a purchase manager, say whether this mill is at risk "
              "next quarter, WHY, and what volume to expect. Use ONLY these facts — do not invent "
              "numbers. Be direct.\n\nFACTS:\n" + json.dumps(f, default=str))
    return _call_groq(prompt)


def _call_groq(prompt):
    body = json.dumps({"model": config.LLM_MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.3}).encode()
    req = urllib.request.Request(
        f"{config.LLM_BASE_URL}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}",
                 "Content-Type": "application/json", "Accept": "application/json",
                 "User-Agent": "Mozilla/5.0 (compatible; SupplyRadar/1.0)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_POST(self):
        if self.path not in ("/api/explain", "/api/insight"):
            self.send_response(404); self._cors(); self.end_headers(); return
        n = int(self.headers.get("Content-Length", 0) or 0)
        try:
            facts = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            facts = {}
        is_insight = self.path == "/api/insight"
        llm_fn = _llm_insight if is_insight else _llm
        tpl_fn = _insight_template if is_insight else _template
        mode = "template"
        if config.LLM_API_KEY:
            try:
                text = llm_fn(facts); mode = "llm"
            except Exception:
                text = tpl_fn(facts)
        else:
            text = tpl_fn(facts)
        payload = json.dumps({"text": text, "mode": mode}).encode()
        self.send_response(200); self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"Agent proxy → http://localhost:{PORT}/api/explain  "
          f"(LLM mode: {'ON (Groq)' if config.LLM_API_KEY else 'OFF → template'})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
