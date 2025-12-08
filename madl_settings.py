import os
from dotenv import load_dotenv

load_dotenv()

# Only keep flags actually used
MADL_ENABLE_FILE_WRITING = os.getenv("MADL_ENABLE_FILE_WRITING", "true").lower() == "true"

MADL_MAX_CONCURRENT_FILES = int(os.getenv("MADL_MAX_CONCURRENT_FILES", "5"))
MADL_MAX_CONCURRENT_AI_CALLS = int(os.getenv("MADL_MAX_CONCURRENT_AI_CALLS", "3"))
MADL_MAX_FILE_SIZE_MB = int(os.getenv("MADL_MAX_FILE_SIZE_MB", "5"))

# Allowed paths for scanning
import json
try:
    MADL_ALLOWED_BASE_PATHS = json.loads(os.getenv("MADL_ALLOWED_BASE_PATHS", "[]"))
except:
    MADL_ALLOWED_BASE_PATHS = []
