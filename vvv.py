"""
FINAL JPM-READY INSERT SCRIPT
--------------------------------
• Connects to OpenSearch
• Uses Azure OpenAI embedding (text-embedding-ada-002)
• Creates index if not exists
• Inserts the combined performLogin() method
• Deletes old login fragments to avoid wrong matches
"""

import os
from opensearchpy import OpenSearch
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------
# 🔥 OpenSearch Configuration
# --------------------------------------------
OPENSEARCH_URL = "https://learn-e779669-os-9200.tale-sandbox.dev.aws.jpmchase.net"
INDEX_NAME = "madl_methods_v2"
VECTOR_FIELD = "embedding"
EMBED_DIM = 1536

client_os = OpenSearch(
    hosts=[OPENSEARCH_URL],
    use_ssl=True,
    verify_certs=False
)

print("Connected to OpenSearch:", client_os.info())


# --------------------------------------------
# 🔥 Azure OpenAI Embedding Client
# --------------------------------------------
client_ai = OpenAI()


def embed(text: str):
    """Generate OpenAI ada-002 embedding"""
    response = client_ai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


# --------------------------------------------
# STEP 1 — Create index if needed
# --------------------------------------------
if not client_os.indices.exists(index=INDEX_NAME):
    print(f"Creating index: {INDEX_NAME}")

    index_body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "method_name": {"type": "keyword"},
                "class_name": {"type": "keyword"},
                "intent": {"type": "text"},
                "semantic_description": {"type": "text"},
                "keywords": {"type": "keyword"},
                "parameters": {"type": "text"},
                "return_type": {"type": "keyword"},
                "full_signature": {"type": "text"},
                "method_code": {"type": "text"},

                VECTOR_FIELD: {
                    "type": "knn_vector",
                    "dimension": EMBED_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "faiss"
                    }
                }
            }
        }
    }

    client_os.indices.create(index=INDEX_NAME, body=index_body)
    print("✅ Index created!")
else:
    print(f"ℹ️ Index already exists: {INDEX_NAME}")


# --------------------------------------------
# STEP 2 — CLEANUP OLD LOGIN METHODS (IMPORTANT)
# --------------------------------------------
delete_ids = [
    "login",
    "enter_username",
    "enter_password",
    "click_login"
]

for did in delete_ids:
    try:
        client_os.delete(index=INDEX_NAME, id=did)
        print(f"🗑️ Deleted old method: {did}")
    except:
        pass


# --------------------------------------------
# STEP 3 — Insert the FINAL combined login method
# --------------------------------------------
method_id = "perform_login_combined"

method_doc = {
    "method_name": "performLogin",
    "class_name": "LoginPage",
    "intent": "login with username and password",
    "semantic_description": "enter username, enter password, and click login button",
    "keywords": [
        "login", "username", "password",
        "enter username", "enter password", "click login button"
    ],
    "parameters": "(username, password)",
    "return_type": "void",
    "full_signature": "performLogin(username, password)",
    "method_code": """
def performLogin(self, username, password):
    self.page.get_by_placeholder("Username").fill(username)
    self.page.get_by_placeholder("Password").fill(password)
    self.page.get_by_role("button", name="Login").click()
"""
}

# Create embedding text
embedding_text = " ".join([
    method_doc["semantic_description"],
    method_doc["intent"],
    " ".join(method_doc["keywords"]),
    method_doc["method_name"]
])

method_doc[VECTOR_FIELD] = embed(embedding_text)

# Insert into OpenSearch
response = client_os.index(
    index=INDEX_NAME,
    id=method_id,
    body=method_doc,
    refresh=True
)

print("✅ Successfully inserted combined performLogin() method!")
print(response)
