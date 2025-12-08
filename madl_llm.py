# madl_llm.py   (Certificate-based Azure OpenAI authentication)
from __future__ import annotations
import json
import os
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

from azure.identity import CertificateCredential
from openai import AzureOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
CERTIFICATE_PATH = os.getenv("CERTIFICATE_PATH")
SCOPE = os.getenv("SCOPE")   # usually: https://cognitiveservices.azure.com/.default

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# --------------------------------------------
# 1️⃣ Acquire Token Using CertificateCredential
# --------------------------------------------
credential = CertificateCredential(
    tenant_id=TENANT_ID,
    client_id=CLIENT_ID,
    certificate_path=CERTIFICATE_PATH,
)

token = credential.get_token(SCOPE).token


# ---------------------------------------------------
# 2️⃣ Initialize Azure OpenAI with azure_ad_token
# ---------------------------------------------------
def get_fresh_client():
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token=token,                        # ✔ certificate token
        api_version=AZURE_OPENAI_API_VERSION,
    )


SYSTEM_PROMPT = """
You analyze code and determine whether a method is a reusable automation method.

Reusable methods must:
- Perform a meaningful automation action
- NOT be a trivial helper
- Be reusable across test cases

Return STRICT JSON ONLY.
""".strip()


# ---------------------------------------------------
# 3️⃣ Main Function
# ---------------------------------------------------
def generate_madl_for_method(method_code: str, class_name: str, parameters: str, language: str):
    
    client = get_fresh_client()  # important: always fresh client

    prompt = f"""
Analyze this method.

Language: {language}
Class/File: {class_name}
Parameters: {parameters}

Method code:
{method_code}
"""

    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,    # ← deployment NAME only
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
    except Exception as e:
        logger.error(f"Azure OpenAI error: {e}", exc_info=True)
        return None

    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
    except Exception:
        logger.error(f"JSON parse failed. RAW:\n{raw}")
        return None

    if not data.get("reusable"):
        return None

    return {
        "reusable": True,
        "method_name": data.get("method_name", "").strip(),
        "class_name": data.get("class_name", class_name),
        "intent": data.get("intent", "").strip(),
        "semantic_description": data.get("semantic_description", "").strip(),
        "keywords": data.get("keywords", []),
        "parameters": data.get("parameters", parameters),
        "method_code": data.get("method_code", method_code),
    }
