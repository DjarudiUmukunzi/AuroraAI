"""
explain_agent.py

MVP version of Phase 3. One agent, not four: it takes the model's Kp
forecast plus the latest NOAA alert text and asks Azure OpenAI (gpt-5-mini)
to explain what it means in plain language, aurora-visibility terms. This
*is* your RAG-lite - "retrieval" is just reading the latest cached
alerts.csv, no Azure AI Search index needed yet.

Uses Azure OpenAI via the standard openai SDK pointed at your Foundry
resource's v1 endpoint. Set these in .env (see .env.example):
    AZURE_OPENAI_ENDPOINT   e.g. https://foundry-auroraai-yourname.openai.azure.com/
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_DEPLOYMENT e.g. gpt-5-mini (must match your deployment name exactly)

Falls back to a template explanation if no key is configured, so the
dashboard still runs end-to-end while you set that up.

Run directly for a quick test:
    python explain_agent.py
"""

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")


def _get_client():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        return None

    from openai import OpenAI

    # Azure OpenAI's v1 API is OpenAI-SDK compatible - just point base_url
    # at your resource's /openai/v1/ path instead of using the separate
    # AzureOpenAI class + api_version dance.
    base_url = endpoint.rstrip("/") + "/openai/v1/"
    return OpenAI(api_key=api_key, base_url=base_url)


def latest_alert_text() -> str:
    path = os.path.join(DATA_DIR, "alerts.csv")
    if not os.path.exists(path):
        return "(no alert bulletin available - run fetch_noaa_data.py)"
    df = pd.read_csv(path)
    msg_col = next((c for c in df.columns if "message" in c.lower()), None)
    if msg_col is None or df.empty:
        return "(no alert bulletin available)"
    return str(df.iloc[-1][msg_col])[:1500]  # keep prompt small


def explain_forecast(predicted_kp: float, bulletin_text: str | None = None) -> str:
    bulletin_text = bulletin_text or latest_alert_text()
    client = _get_client()

    if client is None:
        # No API key configured yet - deterministic fallback so the rest
        # of the pipeline (dashboard) still runs end-to-end.
        level = (
            "quiet" if predicted_kp < 4 else
            "unsettled to active" if predicted_kp < 6 else
            "storm-level"
        )
        return (
            f"[template - no LLM configured] Forecast Kp is {predicted_kp:.1f}, "
            f"indicating {level} geomagnetic conditions. Add AZURE_OPENAI_* vars "
            f"to .env for a real narrative explanation."
        )

    prompt = f"""You are a space weather assistant. Given a forecast Kp index
value and the latest NOAA alert bulletin, write a short (3-4 sentence),
plain-language explanation for a general audience: what the forecast means,
whether aurora may be visible at mid-latitudes, and how confident to be.

Forecast Kp index (next 3-hour window): {predicted_kp:.2f}

Latest NOAA bulletin:
{bulletin_text}
"""

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "developer", "content": "You are a concise, accurate space weather assistant."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=400,
        reasoning_effort="low",  # this task doesn't need deep reasoning - keep it fast/cheap
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    print(explain_forecast(predicted_kp=4.3))