import os
import logging
from azure.identity import CertificateCredential
from openai import AzureOpenAI

def get_azure_client():
    """
    Returns a configured AzureOpenAI client using certificate-based authentication.
    """
    tenant_id = os.environ["AZURE_TENANT_ID"]
    client_id = os.environ["AZURE_CLIENT_ID"]
    cert_path = os.environ["CERTIFICATE_PATH"]
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    api_version = os.environ["AZURE_OPENAI_API_VERSION"]

    scope = "https://cognitiveservices.azure.com/.default"

    logging.info("Fetching access token using CertificateCredential...")

    credential = CertificateCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        certificate_path=cert_path
    )

    access_token = credential.get_token(scope).token
    logging.info("Access token retrieved.")

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        default_headers={
            "Authorization": f"Bearer {access_token}"
        }
    )

    return client
-------------------------------------------------




import os
from .azure_client import get_azure_client

def generate_embedding(text: str):
    client = get_azure_client()

    deployment = os.environ["EMBEDDING_DEPLOYMENT_NAME"]

    response = client.embeddings.create(
        model=deployment,
        input=text
    )

    return response.data[0].embedding
------------------------------------------------


from opensearchpy import OpenSearch
import os

# Your OS URL
OPENSEARCH_URL = "https://learn-e779669-os-9200.tale-sandbox.dev.aws.jpmchase.net"

def get_os_client():
    return OpenSearch(
        hosts=[OPENSEARCH_URL],
        verify_certs=False,
        use_ssl=True,
        ssl_show_warn=False,
    )


------------------------------------------------------

from .opensearch_client import get_os_client
from .azure_embeddings import generate_embedding

INDEX_NAME = "madl_methods_v2"

def insert_method(doc_id: str, payload: dict):
    os_client = get_os_client()

    # Build short embedding text
    combined_text = " ".join([
        payload.get("semantic_description", ""),
        payload.get("intent", ""),
        " ".join(payload.get("keywords", [])),
        payload.get("method_name", ""),
        payload.get("parameters", "")
    ])

    vector = generate_embedding(combined_text)

    body = {
        "method_name": payload["method_name"],
        "class_name": payload["class_name"],
        "intent": payload["intent"],
        "semantic_description": payload["semantic_description"],
        "keywords": payload["keywords"],
        "parameters": payload["parameters"],
        "return_type": payload["return_type"],
        "full_signature": payload["full_signature"],
        "method_code": payload["method_code"],
        "embedding": vector
    }

    os_client.index(
        index=INDEX_NAME,
        id=doc_id,
        body=body
    )

    print(f"Inserted method: {doc_id}")


-------------------------------------------------------------------
from .opensearch_client import get_os_client
from .azure_embeddings import generate_embedding

INDEX_NAME = "madl_methods_v2"
VECTOR_FIELD = "embedding"

def knn_search(query: str, top_k: int = 5):
    os_client = get_os_client()

    query_vector = generate_embedding(query)

    body = {
        "size": top_k,
        "query": {
            "knn": {
                VECTOR_FIELD: {
                    "vector": query_vector,
                    "k": top_k
                }
            }
        }
    }

    response = os_client.search(
        index=INDEX_NAME,
        body=body
    )

    hits = response["hits"]["hits"]
    return hits
----------------------------------------------------
