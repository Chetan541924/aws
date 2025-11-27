import os
import logging
from dotenv import load_dotenv
from azure.identity import CertificateCredential
from openai import AzureOpenAI
from opensearchpy import OpenSearch

load_dotenv()
logging.basicConfig(level=logging.INFO)


# ============================================================
# 1️⃣ GET ACCESS TOKEN (certificate auth)
# ============================================================
def get_access_token():
    cert_path = os.environ["CERTIFICATE_PATH"]
    scope = os.environ["SCOPE"]

    credential = CertificateCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        certificate_path=cert_path
    )

    token = credential.get_token(scope).token
    logging.info("Access token retrieved.")
    return token


# ============================================================
# 2️⃣ BUILD AZURE OPENAI CLIENT (JPMC gateway)
# ============================================================
def get_azure_client():
    access_token = get_access_token()

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],

        # JPM custom headers
        default_headers={
            "Authorization": f"Bearer {access_token}",
            "api-key": os.environ["AZURE_OPENAI_API_KEY"],
            "user_sid": "REPLACE"
        }
    )

    return client


# ============================================================
# 3️⃣ CONNECT TO OPENSEARCH
# ============================================================
OPENSEARCH_URL = "https://learn-e779669-os-9200.tale-sandbox.dev.aws.jpmchase.net"
INDEX_NAME = "madl_methods_v2"
VECTOR_FIELD = "embedding"

os_client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    use_ssl=True,
    verify_certs=False
)

print("Connected to OpenSearch:", os_client.info())


# ============================================================
# 4️⃣ EMBEDDING FUNCTION
# ============================================================
def embed(text: str):
    azure_client = get_azure_client()

    response = azure_client.embeddings.create(
        model=os.environ["EMBEDDING_DEPLOYMENT_NAME"],   # <= FIXED
        input=text
    )

    return response.data[0].embedding


# ============================================================
# 5️⃣ Create index if needed
# ============================================================
if not os_client.indices.exists(index=INDEX_NAME):
    index_body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                VECTOR_FIELD: {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "engine": "faiss",
                        "name": "hnsw",
                        "space_type": "l2"
                    }
                },
                "method_name": {"type": "keyword"},
                "class_name": {"type": "keyword"},
                "intent": {"type": "text"},
                "semantic_description": {"type": "text"},
                "keywords": {"type": "keyword"},
                "parameters": {"type": "text"},
                "return_type": {"type": "keyword"},
                "full_signature": {"type": "text"},
                "method_code": {"type": "text"}
            }
        }
    }

    os_client.indices.create(index=INDEX_NAME, body=index_body)
    print("Index created successfully.")
else:
    print("Index already exists.")


# ============================================================
# 6️⃣ Insert sample reusable method
# ============================================================
method = {
    "method_name": "login",
    "class_name": "LoginPage",
    "intent": "login with username and password",
    "semantic_description": "perform login using username and password",
    "keywords": ["login", "username", "password"],
    "parameters": "(username: str, password: str)",
    "return_type": "void",
    "full_signature": "login(username, password)",
    "method_code": "print('dummy code')"
}

embedding_text = " ".join([
    method["semantic_description"],
    method["intent"],
    " ".join(method["keywords"]),
    method["method_name"],
    method["parameters"]
])

vector = embed(embedding_text)

doc = {**method, VECTOR_FIELD: vector}

res = os_client.index(
    index=INDEX_NAME,
    body=doc,
    id=f"{method['class_name']}.{method['method_name']}",
    refresh=True
)

print("🎉 Successfully inserted into OpenSearch!")
print(res)
