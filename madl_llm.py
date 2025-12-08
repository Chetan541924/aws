from __future__ import annotations
import json
import os
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

from azure.identity import CertificateCredential
from openai import AzureOpenAI

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# ENV VARIABLES
# ---------------------------------------------------------
TENANT_ID = os.getenv("AZURE_TENANT_ID")
CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
CERTIFICATE_PATH = os.getenv("CERTIFICATE_PATH")
SCOPE = os.getenv("SCOPE")  # https://cognitiveservices.azure.com/.default

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  # dummy key (required but not validated)

if not (TENANT_ID and CLIENT_ID and CERTIFICATE_PATH and AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT):
    raise RuntimeError("Missing Azure OpenAI certificate config in .env")

# ---------------------------------------------------------
# Certificate Credential
# ---------------------------------------------------------
credential = CertificateCredential(
    tenant_id=TENANT_ID,
    client_id=CLIENT_ID,
    certificate_path=CERTIFICATE_PATH,
)

# ---------------------------------------------------------
# Function to build a fresh AzureOpenAI client
# ---------------------------------------------------------
def get_fresh_client() -> AzureOpenAI:
    """
    JPM Azure OpenAI requires:
    - New AAD token per request
    - Bearer token passed manually in headers
    - api_key still required (dummy / unused)
    - user_sid header required by JPM infra
    """
    try:
        token = credential.get_token(SCOPE).token
        logger.info("New Azure AD token acquired")
    except Exception as e:
        logger.error(f"Failed to acquire Azure AD token: {e}", exc_info=True)
        raise

    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        default_headers={
            "Authorization": f"Bearer {token}",
            "user_sid": "REPLACE"
        },
    )

# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You analyze code and determine whether a method is a reusable automation method.

Reusable methods must:
- Perform a meaningful automation action (login, navigate, fill form, validate UI, etc.)
- NOT be trivial helpers or getters/setters
- Be general enough to reuse across multiple test cases

You MUST return STRICT JSON ONLY with NO additional text, markdown, or explanation.

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

# ---------------------------------------------------------
# MAIN MADL GENERATION FUNCTION
# ---------------------------------------------------------
def generate_madl_for_method(
    method_code: str,
    class_name: str,
    parameters: str,
    language: str
) -> Optional[Dict]:
    """
    Call Azure OpenAI and, if the method is reusable, return a MADL dict.
    Otherwise return None.
    """
    user_prompt = f"""
Analyze this method and decide if it is reusable.

Language: {language}
Class/File: {class_name}
Parameters: {parameters}

Method code:
```{language}
{method_code}
```
""".strip()

    try:
        client = get_fresh_client()
    except Exception as e:
        logger.error(f"Failed to create Azure OpenAI client: {e}")
        return None

    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
    except Exception as e:
        logger.error(f"Azure OpenAI API error: {e}", exc_info=True)
        return None

    if not resp.choices:
        logger.warning("No response choices from Azure OpenAI")
        return None

    raw = resp.choices[0].message.content
    if not raw:
        logger.warning("Empty response content from Azure OpenAI")
        return None

    raw = raw.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split('\n')
        # Remove first line if it's a fence marker
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Remove last line if it's a fence marker
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = '\n'.join(lines).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Raw response: {raw}")
        return None

    if not isinstance(data, dict):
        logger.warning("Response is not a dictionary")
        return None

    if not data.get("reusable"):
        return None

    # Validate required fields
    method_name = data.get("method_name", "").strip()
    intent = data.get("intent", "").strip()

    if not method_name or not intent:
        logger.warning("Missing required fields: method_name or intent")
        return None

    # Ensure keywords is a list of strings
    keywords = data.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(k).strip() for k in keywords if k]

    madl = {
        "reusable": True,
        "method_name": method_name,
        "class_name": data.get("class_name", class_name).strip(),
        "intent": intent,
        "semantic_description": data.get("semantic_description", "").strip(),
        "keywords": keywords,
        "parameters": data.get("parameters", parameters).strip(),
        "method_code": data.get("method_code", method_code).strip(),
    }

    return madl
