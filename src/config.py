"""Environment-driven configuration. Copy .env.example -> .env or export vars.

WAREHOUSE_URL  empty            -> SQLite (zero-setup default)
               postgresql+psycopg2://radar:radar@localhost:5432/supplyradar -> Postgres
LLM_API_KEY    empty            -> agent uses deterministic templates (offline)
               a free-tier key  -> agent writes the brief with an LLM
LLM_BASE_URL   any OpenAI-compatible endpoint (default: Groq free tier)
"""
import os

WAREHOUSE_URL = os.getenv("WAREHOUSE_URL", "").strip()

LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# weather extraction window (history needed for model training)
WEATHER_START = os.getenv("WEATHER_START", "2020-04-01")
