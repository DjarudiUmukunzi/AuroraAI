"""
build_search_index.py

Phase 3, step 1: creates a vector search index in Azure AI Search and
populates it with NOAA alert bulletins, embedded via Azure OpenAI's
text-embedding-3-small deployment. This is the knowledge base the
Research Agent queries via RAG.

Run:
    python agents/build_search_index.py
"""

import os
import uuid

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_SEARCH_KEY"]
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "noaa-bulletins")

EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small's native output size

credential = AzureKeyCredential(SEARCH_KEY)


def get_openai_client() -> OpenAI:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    base_url = endpoint.rstrip("/") + "/openai/v1/"
    return OpenAI(api_key=api_key, base_url=base_url)


def create_index():
    index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=credential)

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="published_at", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters=HnswParameters(m=4, ef_construction=400, ef_search=500, metric="cosine"),
            )
        ],
        profiles=[
            VectorSearchProfile(name="vector-profile", algorithm_configuration_name="hnsw-config")
        ],
    )

    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)
    index_client.create_or_update_index(index)
    print(f"Index '{INDEX_NAME}' created/updated.")


def load_bulletins() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "alerts.csv")
    df = pd.read_csv(path)
    msg_col = next((c for c in df.columns if "message" in c.lower()), None)
    time_col = next((c for c in df.columns if "issue" in c.lower() or "time" in c.lower()), None)
    if msg_col is None:
        raise ValueError(f"No message column found in alerts.csv. Columns: {list(df.columns)}")
    df = df[[msg_col] + ([time_col] if time_col else [])].dropna(subset=[msg_col])
    df = df.rename(columns={msg_col: "content"})
    if time_col:
        df = df.rename(columns={time_col: "published_at"})
    else:
        df["published_at"] = ""
    return df.drop_duplicates(subset=["content"])


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    # Batch in groups of 16 to keep requests reasonably sized
    embeddings = []
    for i in range(0, len(texts), 16):
        batch = texts[i : i + 16]
        resp = client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=batch)
        embeddings.extend([d.embedding for d in resp.data])
    return embeddings


def main():
    create_index()

    bulletins = load_bulletins()
    print(f"Loaded {len(bulletins)} unique bulletins from alerts.csv")

    openai_client = get_openai_client()
    vectors = embed_texts(openai_client, bulletins["content"].tolist())

    search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=INDEX_NAME, credential=credential)

    docs = []
    for (_, row), vector in zip(bulletins.iterrows(), vectors):
        docs.append(
            {
                "id": str(uuid.uuid4()),
                "content": str(row["content"])[:32000],  # field length safety
                "source": "NOAA SWPC alerts",
                "published_at": str(row.get("published_at", "")),
                "content_vector": vector,
            }
        )

    result = search_client.upload_documents(documents=docs)
    succeeded = sum(1 for r in result if r.succeeded)
    print(f"Uploaded {succeeded}/{len(docs)} bulletins to the search index.")


if __name__ == "__main__":
    main()