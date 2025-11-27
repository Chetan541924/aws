from opensearchpy import OpenSearch
from openai import OpenAI

# --------------------------------------------
# 🔥 CONFIG
# --------------------------------------------
OPENSEARCH_URL = "https://learn-e779669-os-9200.tale-sandbox.dev.aws.jpmchase.net"
INDEX_NAME = "madl_methods_v2"
VECTOR_FIELD = "embedding"

# Connect to OpenSearch
os_client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    use_ssl=True,
    verify_certs=False
)

print("Connected to OpenSearch:", os_client.info())

# OpenAI client
ai_client = OpenAI()

# --------------------------------------------
# Generate embedding using ADA-002 (1536 dims)
# --------------------------------------------
def embed(text: str):
    response = ai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


# --------------------------------------------
# Ensure index exists
# --------------------------------------------
if not os_client.indices.exists(INDEX_NAME):
    print(f"Creating index: {INDEX_NAME}")

    index_body = {
        "settings": {
            "index": {
                "knn": True
            }
        },
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
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "faiss"
                    }
                }
            }
        }
    }

    os_client.indices.create(index=INDEX_NAME, body=index_body)
    print("Index created successfully.")
else:
    print(f"Index '{INDEX_NAME}' already exists.")


# --------------------------------------------
# 🔥 Example Method metadata
#   (Replace with your own values)
# --------------------------------------------
method = {
    "method_name": "login",
    "class_name": "LoginPage",
    "intent": "login with username and password",
    "semantic_description": "perform login using username and password",
    "keywords": ["login", "username", "password", "authenticate"],
    "parameters": "(username: str, password: str)",
    "return_type": "void",
    "full_signature": "login(username: str, password: str)",
    "method_code": """
def login(self, username, password):
    self.page.get_by_placeholder("Username").fill(username)
    self.page.get_by_placeholder("Password").fill(password)
    self.page.get_by_role("button", name="Login").click()
"""
}

# --------------------------------------------
# Build embedding text
# --------------------------------------------
embedding_text = " ".join([
    method["semantic_description"],
    method["intent"],
    " ".join(method["keywords"]),
    method["method_name"],
    method["parameters"]
])

vector = embed(embedding_text)

# --------------------------------------------
# Insert document into OpenSearch
# --------------------------------------------
document = {**method, VECTOR_FIELD: vector}

response = os_client.index(
    index=INDEX_NAME,
    body=document,
    id=f"{method['class_name']}.{method['method_name']}",
    refresh=True
)

print("✅ Successfully inserted reusable method into OpenSearch!")
print(response)
