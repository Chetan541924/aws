import os
import logging
from azure.identity import CertificateCredential
from openai import AzureOpenAI
from opensearchpy import OpenSearch

# =====================================================================
# 1️⃣ JPMC Certificate-Based AAD Authentication (REQUIRED)
# =====================================================================
def get_access_token():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # path to .pem file your client gave you
    cert_path = os.path.join(dir_path, "JPMC1_cert/uatagent.azure.jpmchase.new.pem")

    scope = "https://cognitiveservices.azure.com/.default"

    credential = CertificateCredential(
        client_id=os.environ["AZURE_CLIENT_ID"],
        tenant_id=os.environ["AZURE_TENANT_ID"],
        certificate_path=cert_path,
    )

    token = credential.get_token(scope).token
    logging.info("Access token retrieved")
    return token


# =====================================================================
# 2️⃣ CONNECT TO AZURE OPENAI USING JPMC TOKEN + API KEY
# =====================================================================
def get_azure_client():
    access_token = get_access_token()

    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        default_headers={
            "Authorization": f"Bearer {access_token}",
            "user_sid": "REPLACE",
            "api-key": os.environ["AZURE_OPENAI_API_KEY"],
        }
    )

    return client


# =====================================================================
# 3️⃣ Connect to OpenSearch
# =====================================================================
OPENSEARCH_URL = "https://learn-e779669-os-9200.tale-sandbox.dev.aws.jpmchase.net"
INDEX_NAME = "madl_methods_v2"
VECTOR_FIELD = "embedding"

os_client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    use_ssl=True,
    verify_certs=False
)

print("Connected to OpenSearch:", os_client.info())


# =====================================================================
# 4️⃣ Embedding Function (using JPMC Azure Auth)
# =====================================================================
def embed(text: str):
    azure_client = get_azure_client()

    response = azure_client.embeddings.create(
        model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],  # ex: text-embedding-ada-002
        input=text
    )

    return response.data[0].embedding


# =====================================================================
# 5️⃣ Create index if not exists
# =====================================================================
if not os_client.indices.exists(index=INDEX_NAME):
    print(f"Creating index: {INDEX_NAME}")

    index_body = {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                VECTOR_FIELD: {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
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
                "method_code": {"type": "text"},
            }
        }
    }

    os_client.indices.create(index=INDEX_NAME, body=index_body)
    print("Index created successfully.")

else:
    print("Index already exists")


# =====================================================================
# 6️⃣ INSERT SAMPLE METHOD
# =====================================================================
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


# Build embedding text
embedding_text = " ".join([
    method["semantic_description"],
    method["intent"],
    " ".join(method["keywords"]),
    method["method_name"],
    method["parameters"]
])

vector = embed(embedding_text)

document = {**method, VECTOR_FIELD: vector}

response = os_client.index(
    index=INDEX_NAME,
    body=document,
    id=f"{method['class_name']}.{method['method_name']}",
    refresh=True
)

print("🎉 Successfully inserted reusable method into OpenSearch!")
print(response)
