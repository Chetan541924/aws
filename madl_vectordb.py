# madl_vectordb.py
from __future__ import annotations
from typing import Dict, List
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
)

from madl_settings import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_USE_HTTPS,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_VECTOR_SIZE,
    get_embedding_model,
)

# Initialize the embedding model and Qdrant client
_embedding_model = get_embedding_model()

_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    https=QDRANT_USE_HTTPS,
    api_key=QDRANT_API_KEY,
)


def ensure_collection_exists():
    """Create Qdrant collection if missing."""
    try:
        _client.get_collection(QDRANT_COLLECTION_NAME)
        return  # Exists already
    except Exception:
        pass

    _client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=QDRANT_VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )


def embed(text: str) -> List[float]:
    """Compute embedding using SentenceTransformer."""
    return _embedding_model.encode(text).tolist()


def push_madl_to_qdrant(madl: Dict, extra: Dict | None = None):
    """
    Push MADL metadata into Qdrant.
    """
    ensure_collection_exists()

    emb_text = madl["semantic_description"] or madl["intent"]
    vector = embed(emb_text)

    payload = {
        "method_name": madl["method_name"],
        "class_name": madl["class_name"],
        "intent": madl["intent"],
        "semantic_description": madl["semantic_description"],
        "keywords": madl.get("keywords", []),
        "parameters": madl["parameters"],
        "method_code": madl["method_code"],
    }

    if extra:
        payload.update(extra)

    point = PointStruct(
        id=uuid4().hex,
        vector=vector,
        payload=payload,
    )

    _client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[point],
    )
