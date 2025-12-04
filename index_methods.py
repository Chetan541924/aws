import os
import logging
from dotenv import load_dotenv
from azure.identity import CertificateCredential
from openai import AzureOpenAI
from opensearchpy import OpenSearch

# --------------------------------------------
# LOAD ENV
# --------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)

INDEX_NAME = "madl_methods_v2"
VECTOR_FIELD = "embedding"


# ----------------------------------------------------------
# 🔥 1. Get Azure Access Token (Certificate + Tenant + Client)
# ----------------------------------------------------------
def get_access_token():
    cert_path = os.environ["CERTIFICATE_PATH"]
    tenant_id = os.environ["AZURE_TENANT_ID"]
    client_id = os.environ["AZURE_CLIENT_ID"]
    scope = "https://cognitiveservices.azure.com/.default"

    credential = CertificateCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        certificate_path=cert_path,
    )

    token = credential.get_token(scope).token
    logging.info("✔ Azure Access Token Retrieved")
    return token


# ----------------------------------------------------------
# 🔥 2. Azure OpenAI Client (Embeddings)
# ----------------------------------------------------------
client_ai = AzureOpenAI(
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    default_headers={
        "Authorization": f"Bearer {get_access_token()}",
        "api-key": os.environ["AZURE_OPENAI_API_KEY"],
    }
)


def embed(text: str):
    """Generate Azure OpenAI Embedding"""
    res = client_ai.embeddings.create(
        model=os.environ["EMBEDDING_DEPLOYMENT_NAME"],
        input=[text]
    )
    return res.data[0].embedding


# ----------------------------------------------------------
# 🔥 3. OpenSearch Client
# ----------------------------------------------------------
OPENSEARCH_URL = "https://learn-e779669-os-9200.tale-sandbox.dev.aws.jpmchase.net"

client_os = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_compress=True,
    use_ssl=True,
    verify_certs=False,
)

print("Connected to OpenSearch:", client_os.info())


# ----------------------------------------------------------
# 🔥 4. Create Index (if missing)
# ----------------------------------------------------------
def ensure_index():
    if client_os.indices.exists(index=INDEX_NAME):
        print(f"ℹ️ Index already exists: {INDEX_NAME}")
        return

    print(f"Creating index: {INDEX_NAME}")

    index_body = {
        "settings": {
            "index": {"knn": True}
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

    client_os.indices.create(index=INDEX_NAME, body=index_body)
    print("✅ Index created successfully!")


# ----------------------------------------------------------
# 🔥 5. Insert Method Metadata
# ----------------------------------------------------------
def insert_method(
    method_name: str,
    class_name: str,
    intent: str,
    semantic_description: str,
    keywords: list,
    parameters: str,
    method_code: str,
):
    full_signature = f"{method_name}{parameters}"

    embedding_text = " ".join([
        semantic_description,
        intent,
        " ".join(keywords),
        method_name,
        parameters
    ])

    vector = embed(embedding_text)

    doc = {
        "method_name": method_name,
        "class_name": class_name,
        "intent": intent,
        "semantic_description": semantic_description,
        "keywords": keywords,
        "parameters": parameters,
        "return_type": "void",
        "full_signature": full_signature,
        "method_code": method_code,
        VECTOR_FIELD: vector
    }

    response = client_os.index(
        index=INDEX_NAME,
        id=f"method_{method_name.lower()}",
        body=doc,
        refresh=True
    )

    print(f"✅ Successfully inserted {method_name}() method!")
    print(response)


# ----------------------------------------------------------
# 🔥 6. MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    ensure_index()

    # Example method: LOGIN
    insert_method(
        method_name="login",
        class_name="LoginPage",
        intent="login with username and password",
        semantic_description="perform login using username and password and click login button",
        keywords=["login", "username", "password", "credentials", "click", "submit"],
        parameters="(username: str, password: str)",
        method_code="""
def login(self, username, password):
    self.page.get_by_placeholder("Username").fill(username)
    self.page.get_by_placeholder("Password").fill(password)
    self.page.get_by_role("button", name="Login").click()
"""
    )
