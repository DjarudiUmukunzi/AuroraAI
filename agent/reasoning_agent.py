"""
reasoning_agent.py

Reasoning Agent: takes the Forecasting Agent's predicted Kp value and the
Research Agent's retrieved bulletins, and synthesizes them into a
plain-language explanation with a recommendation. This is the final
synthesis step in the Phase 3 multi-agent pipeline.

Run directly for a quick test:
    python reasoning_agent.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")


def _get_client():
    from openai import OpenAI

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    base_url = endpoint.rstrip("/") + "/openai/v1/"
    return OpenAI(api_key=api_key, base_url=base_url)


def reason(predicted_kp: float, research_results: list[dict]) -> str:
    """Synthesize a forecast + retrieved bulletins into a plain-language
    explanation and recommendation."""
    client = _get_client()

    bulletins_text = "\n\n".join(
        f"[{r['published_at']}] {r['content'][:500]}" for r in research_results
    )

    prompt = f"""You are the Reasoning Agent in a space weather forecasting
system. You've been given a quantitative Kp-index forecast from the
Forecasting Agent and relevant NOAA bulletins retrieved by the Research
Agent. Synthesize these into a clear, plain-language explanation (4-5
sentences) covering: what the forecast means, how it's supported or
complicated by the retrieved bulletins, aurora-visibility implications at
mid-latitudes, and a practical recommendation (e.g. "check again in a few
hours" or "aurora watchers at high latitudes should be alert tonight").

Forecasting Agent output (predicted Kp index, next 3-hour window): {predicted_kp:.2f}

Research Agent output (retrieved NOAA bulletins):
{bulletins_text}
"""

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "developer", "content": "You are a concise, accurate space weather reasoning assistant."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=700,
        reasoning_effort="low",
    )
    return (resp.choices[0].message.content or "").strip()


if __name__ == "__main__":
    from forecasting_agent import forecast_next_kp
    from research_agent import research

    predicted = forecast_next_kp()
    hits = research("geomagnetic storm aurora forecast")
    explanation = reason(predicted, hits)
    print(f"\nPredicted Kp: {predicted:.2f}\n")
    print(explanation)