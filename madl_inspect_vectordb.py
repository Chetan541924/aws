# madl_inspect_vectordb.py

from madl_settings import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_USE_HTTPS,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
)
from qdrant_client import QdrantClient
from qdrant_client.models import ScrollRequest

import json

def inspect_vectordb(limit: int = 50, show_vectors: bool = False):
    print("Connecting to Qdrant...")

    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        https=QDRANT_USE_HTTPS,
        api_key=QDRANT_API_KEY,
    )

    print(f"Connected to collection: {QDRANT_COLLECTION_NAME}\n")

    # Fetch total count
    try:
        info = client.get_collection(QDRANT_COLLECTION_NAME)
        total = info.points_count
    except Exception as e:
        print("Error fetching collection info:", e)
        return

    print(f"Total methods stored in Qdrant: {total}\n")

    # Scroll (paginate) through points
    next_page = None
    fetched = 0

    while True:
        scroll_result = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=None,
            limit=limit,
            offset=next_page,
            with_vectors=show_vectors,
            with_payload=True,
        )

        points, next_page = scroll_result

        if not points:
            print("\nNo more points found.\n")
            break

        for p in points:
            fetched += 1
            print("=" * 80)
            print(f"POINT ID: {p.id}")

            # Show metadata
            print("\n--- MADL METADATA ---")
            try:
                print(json.dumps(p.payload, indent=2))
            except:
                print(p.payload)

            # Optionally show vector
            if show_vectors:
                print("\n--- VECTOR (Embedding) ---")
                print(p.vector[:10], "... (truncated)")  # only print first 10 dims

            print("=" * 80 + "\n")

        if next_page is None:
            break

    print(f"\nFinished. Total fetched: {fetched}/{total}")


if __name__ == "__main__":
    # customize:
    # - increase limit
    # - enable vector printing
    inspect_vectordb(limit=100, show_vectors=False)
