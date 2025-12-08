# madl_llm.py  (Azure OpenAI – JPM Certificate Auth Version)
from __future__ import annotations
import json
import os
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

from azure.identity import CertificateCredential
from openai import AzureOpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load ENV
load_dotenv()

TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
CERTIFICATE_PATH = os.getenv("CERTIFICATE_PATH")
SCOPE = os.getenv("SCOPE")  # should be https://cognitiveservices.azure.com/.default

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

if not (TENANT_ID and CLIENT_ID and CERTIFICATE_PATH and AZURE_OPENAI_ENDPOINT):
    raise RuntimeError("Missing Azure OpenAI certificate config in .env")

# -------------------------------------------------------------
# 1️⃣ Acquire Azure AD Token using CertificateCredential
# -------------------------------------------------------------
credential = CertificateCredential(
    tenant_id=TENANT_ID,
    client_id=CLIENT_ID,
    certificate_path=CERTIFICATE_PATH
)

token = credential.get_token(SCOPE).token


# -------------------------------------------------------------
# 2️⃣ Initialize Azure OpenAI client
# -------------------------------------------------------------
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_ad_token=token,
    api_version=AZURE_OPENAI_API_VERSION,
)

SYSTEM_PROMPT = """
You analyze code and determine whether a method is a reusable automation method.

Reusable methods must:
- Perform a meaningful automation action (login, navigate, fill form, validate UI, etc.)
- NOT be trivial helpers or getters/setters
- Be general enough to reuse across multiple test cases

You MUST return STRICT JSON ONLY.

If reusable:
{
  "reusable": true,
  "method_name": "...",
  "class_name": "...",
  "intent": "...",
  "semantic_description": "...",
  "keywords": ["..."],
  "parameters": "...",
  "method_code": "..."
}

If NOT reusable:
{
  "reusable": false
}
""".strip()


# -------------------------------------------------------------
# 3️⃣ Main MADL generation function
# -------------------------------------------------------------
def generate_madl_for_method(method_code: str, class_name: str, parameters: str, language: str) -> Optional[Dict]:

    user_prompt = f"""
Analyze this method and decide if it is reusable.

Language: {language}
Class/File: {class_name}
Parameters: {parameters}

Method code:
{method_code}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
    except Exception as e:
        logger.error(f"Azure OpenAI error: {e}", exc_info=True)
        return None

    raw = resp.choices[0].message.content.strip()

    # strip markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
    except Exception as e:
        logger.error(f"Failed JSON parse: {e}\nRAW:\n{raw}")
        return None

    if not isinstance(data, dict) or not data.get("reusable"):
        return None

    # Normalize keywords list
    keywords = data.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []

    return {
        "reusable": True,
        "method_name": data.get("method_name", "").strip(),
        "class_name": data.get("class_name", class_name).strip(),
        "intent": data.get("intent", "").strip(),
        "semantic_description": data.get("semantic_description", "").strip(),
        "keywords": keywords,
        "parameters": data.get("parameters", parameters).strip(),
        "method_code": data.get("method_code", method_code).strip(),
    }
