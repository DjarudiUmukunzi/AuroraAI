"""
research_agent.py

Research Agent: embeds a query, retrieves the most relevant NOAA bulletins
from Azure AI Search via vector similarity, and returns them as grounding
context for the Reasoning Agent. This is the RAG half of Phase 3.

Run directly for a quick test:
    python research_agent.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "noaa-bulletins")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")


def _get_openai_client() -> OpenAI:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    base_url = endpoint.rstrip("/") + "/openai/v1/"
    return OpenAI(api_key=api_key, base_url=base_url)


def research(query: str, top_k: int = 3) -> list[dict]:
    """Embed the query and retrieve the top_k most similar NOAA bulletins."""
    openai_client = _get_openai_client()
    embedding = openai_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT, input=[query]
    ).data[0].embedding

    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_KEY),
    )

    vector_query = VectorizedQuery(
        vector=embedding, k_nearest_neighbors=top_k, fields="content_vector"
    )
    results = search_client.search(
        search_text=None, vector_queries=[vector_query], select=["content", "source", "published_at"]
    )

    return [
        {"content": r["content"], "source": r["source"], "published_at": r["published_at"]}
        for r in results
    ]


if __name__ == "__main__":
    hits = research("geomagnetic storm warning solar wind")
    for i, hit in enumerate(hits, 1):
        print(f"\n--- Result {i} ({hit['source']}, {hit['published_at']}) ---")
        print(hit["content"][:300])