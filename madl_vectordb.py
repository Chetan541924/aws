
# madl_vectordb.py
from __future__ import annotations
from typing import Dict, Any

from madl_engine.azure_embeddings import generate_embedding
from madl_engine.opensearch_client import get_os_client

INDEX_NAME = "madl_methods_v2"


INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "class_name": {"type": "keyword"},
            "method_name": {"type": "keyword"},
            "full_signature": {"type": "text"},
            "method_code": {"type": "text"},
            "intent": {"type": "text"},
            "semantic_description": {"type": "text"},
            "keywords": {"type": "keyword"},
            "file_path": {"type": "keyword"},
            "language": {"type": "keyword"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,  # text-embedding-ada-002
            },
        }
    },
    "settings": {
        "index": {
            "knn": True
        }
    },
}


def ensure_index():
    client = get_os_client()
    if not client.indices.exists(index=INDEX_NAME):
        client.indices.create(index=INDEX_NAME, body=INDEX_MAPPING)
        print(f"Created OpenSearch index: {INDEX_NAME}")


def madl_to_opensearch_doc(madl: Dict[str, Any], extra_meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    intent = madl.get("intent", "")
    sem_desc = madl.get("semantic_description", "")
    text_for_embedding = sem_desc or intent

    embedding = generate_embedding(text_for_embedding)

    doc = {
        "class_name": madl["class_name"],
        "method_name": madl["method_name"],
        "full_signature": madl.get("parameters", ""),
        "method_code": madl["method_code"],
        "intent": intent,
        "semantic_description": sem_desc,
        "keywords": madl.get("keywords", []),
        "embedding": embedding,
    }

    if extra_meta:
        doc.update(extra_meta)

    return doc


def push_madl_to_opensearch(madl: Dict[str, Any], extra_meta: Dict[str, Any] | None = None):
    """
    Index / update a single MADL method into OpenSearch.
    """
    ensure_index()
    client = get_os_client()

    doc = madl_to_opensearch_doc(madl, extra_meta)

    # Use class+method as deterministic ID to avoid duplicates
    doc_id = f"{doc['class_name']}.{doc['method_name']}"

    client.index(
        index=INDEX_NAME,
        id=doc_id,
        body=doc,
        refresh=True,
    )
    print(f"Indexed MADL method into OpenSearch: {doc_id}")
