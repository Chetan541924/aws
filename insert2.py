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
# 1. Get Azure Access Token using Certificate
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
# 2. Azure OpenAI Embedding Client
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
    """Generate embedding using Azure OpenAI"""
    response = client_ai.embeddings.create(
        model=os.environ["EMBEDDING_DEPLOYMENT_NAME"],
        input=text
    )
    return response.data[0].embedding


# ----------------------------------------------------------
# 3. Connect to OpenSearch
# ----------------------------------------------------------
OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL")

client_os = OpenSearch(
    hosts=[OPENSEARCH_URL],
    http_compress=True,
    use_ssl=True,
    verify_certs=False,
)

print("Connected to OpenSearch:", client_os.info())


# ----------------------------------------------------------
# 4. Ensure Index Exists
# ----------------------------------------------------------
def ensure_index():
    if client_os.indices.exists(index=INDEX_NAME):
        print(f"ℹ️ Index already exists: {INDEX_NAME}")
        return

    print(f"Creating index: {INDEX_NAME}")

    index_body = {
        "settings": {
            "index": { "knn": True }
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
# 5. Insert a Reusable Method
# ----------------------------------------------------------
def insert_method(method_name, class_name, intent, semantic_description,
                  keywords, parameters, method_code):

    full_signature = f"{method_name}{parameters}"

    # Text used for embedding
    embedding_text = " ".join([
        semantic_description,
        intent,
        " ".join(keywords),
        method_name,
        parameters
    ])

    print("Generating embedding...")
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
        "method_code": method_code.strip(),
        VECTOR_FIELD: vector
    }

    response = client_os.index(
        index=INDEX_NAME,
        id=f"method_{method_name.lower()}",
        body=doc,
        refresh=True
    )

    print(f"✅ Successfully inserted method → {method_name}()")
    print(response)


# ----------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    ensure_index()

    # 🚀 Insert Login Method for JC0003 / JC0004
    insert_method(
      method_name="login",
      class_name="LoginPage",
      intent="logs the user into the application",
      semantic_description="""
      User navigates to login page, enters username, enters password,
      clicks the Signin button and logs into the application, 
      then user is redirected to the home page.
      """,
      keywords=[
          "login", "signin", "navigate login page",
          "enter username", "enter password",
          "click signin", "submit", "home page",
          "authentication", "credentials"
      ],
      parameters="(username: str, password: str)",
      method_code="""
  def login(self, username, password):
      self.page.goto("https://ccsui-test.jpmchase.net:8443/ccs/")
      self.page.get_by_placeholder("Username").fill(username)
      self.page.get_by_placeholder("Password").fill(password)
      self.page.get_by_role("button", name="Signin").click()
  """
  )
