"""
vision_agent.py

Phase 5's computer vision component. Fetches NOAA SWPC's live OVATION
Aurora Forecast model image (a real-time map of the predicted auroral
oval) and uses gpt-5-mini's multimodal (vision) capability to analyze
it - describing the oval's current extent, intensity, and which regions
are likely to see aurora. Reuses the existing Azure OpenAI deployment
from Phase 0/3 rather than provisioning a separate Computer Vision
resource, since gpt-5-mini already supports image input.

Run directly for a quick test:
    python vision_agent.py
"""

import base64
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OVATION_IMAGE_URL = "https://services.swpc.noaa.gov/images/animations/ovation/north/latest.jpg"
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")


def _get_client():
    from openai import OpenAI

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    base_url = endpoint.rstrip("/") + "/openai/v1/"
    return OpenAI(api_key=api_key, base_url=base_url)


def fetch_ovation_image() -> bytes:
    resp = requests.get(OVATION_IMAGE_URL, timeout=30)
    resp.raise_for_status()
    return resp.content


def analyze_aurora_image(image_bytes: bytes | None = None) -> str:
    """Fetch (if not provided) and analyze the latest OVATION aurora
    forecast image using gpt-5-mini's vision capability."""
    if image_bytes is None:
        image_bytes = fetch_ovation_image()

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    client = _get_client()

    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "developer",
                "content": (
                    "You are a space weather analyst reading NOAA's OVATION "
                    "Aurora Forecast model image. This map shows the predicted "
                    "auroral oval intensity over the Northern Hemisphere as a "
                    "colored band (typically green/yellow/red = higher "
                    "intensity) over a world map. Describe: (1) the oval's "
                    "approximate latitude extent and whether it looks expanded "
                    "or contracted compared to a typical quiet-condition oval, "
                    "(2) intensity level based on color, (3) which named "
                    "regions/countries the oval currently reaches into, if any "
                    "are visible in the image. Be concise - 3-4 sentences."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this aurora forecast image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ],
            },
        ],
        max_completion_tokens=500,
        reasoning_effort="low",
    )
    return (resp.choices[0].message.content or "").strip()


if __name__ == "__main__":
    print(f"Fetching aurora forecast image from {OVATION_IMAGE_URL} ...")
    analysis = analyze_aurora_image()
    print("\nVision Agent analysis:\n")
    print(analysis)