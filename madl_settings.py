# madl_settings.py
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from project root/app folder
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

# Database connection details
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_MQHJ6cGn9gOh@ep-broad-resonance-a1thxhee.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
)

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
ALGORITHM = "HS256"

# Gemini API Key
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    os.getenv("MADL_GEMINI_API_KEY", "AIzaSyAtsqWGZrRN6fDhR_yMWyAeVDdjIoZ9S94")
)

MADL_QDRANT_URL = os.getenv("MADL_QDRANT_URL", "https://localhost:6333")
MADL_QDRANT_API_KEY = os.getenv("MADL_QDRANT_API_KEY", None)
MADL_QDRANT_COLLECTION_NAME = os.getenv("MADL_QDRANT_COLLECTION_NAME", "madl_methods")
MADL_EMBEDDING_MODEL = os.getenv("MADL_EMBEDDING_MODEL", "all-mpnet-base-v2")

QDRANT_URL = os.getenv("MADL_QDRANT_URL", "https://localhost:6333")
QDRANT_API_KEY = os.getenv("MADL_QDRANT_API_KEY", None)

# Parse cloud Qdrant URL
# Example: https://6803a6c5-cfe2-48e8-96d0-37cf86d2223b.eu-west-2-0.aws.cloud.qdrant.io
try:
    from urllib.parse import urlparse
    parsed_url = urlparse(MADL_QDRANT_URL)
    QDRANT_HOST = parsed_url.hostname or "localhost"
    QDRANT_PORT = parsed_url.port or 443
    QDRANT_SCHEME = parsed_url.scheme or "https"
    QDRANT_USE_HTTPS = QDRANT_SCHEME == "https"
except Exception:
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    QDRANT_USE_HTTPS = False

QDRANT_API_KEY = MADL_QDRANT_API_KEY
QDRANT_COLLECTION_NAME = MADL_QDRANT_COLLECTION_NAME

from sentence_transformers import SentenceTransformer
# NOTE: this loads model once at import
_embedding_model = SentenceTransformer(MADL_EMBEDDING_MODEL)
QDRANT_VECTOR_SIZE = _embedding_model.get_sentence_embedding_dimension()

QDRANT_SIMILARITY_THRESHOLD = 0.6

# MADL Configuration
MADL_ENABLE_FILE_WRITING = os.getenv("MADL_ENABLE_FILE_WRITING", "true").lower() == "true"
MADL_AI_CACHE_SIZE = int(os.getenv("MADL_AI_CACHE_SIZE", "1000"))
MADL_MAX_CONCURRENT_FILES = int(os.getenv("MADL_MAX_CONCURRENT_FILES", "5"))
MADL_MAX_CONCURRENT_AI_CALLS = int(os.getenv("MADL_MAX_CONCURRENT_AI_CALLS", "3"))
MADL_MAX_FILE_SIZE_MB = int(os.getenv("MADL_MAX_FILE_SIZE_MB", "5"))
MADL_ALLOWED_BASE_PATHS = os.getenv("MADL_ALLOWED_BASE_PATHS", "[]")
try:
    import json
    MADL_ALLOWED_BASE_PATHS = json.loads(MADL_ALLOWED_BASE_PATHS)
except Exception:
    MADL_ALLOWED_BASE_PATHS = []

# Service Configuration
MADL_SERVICE_URL = os.getenv("MADL_SERVICE_URL", "http://localhost:8003")
MADL_TIMEOUT = 30
MADL_ENABLED = os.getenv("MADL_ENABLED", "true").lower() == "true"

# Execution Configuration
EXECUTION_TIMEOUT = 300
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def get_embedding_model() -> SentenceTransformer:
    """Return the singleton embedding model loaded above."""
    return _embedding_model
