# madl_llm.py  (Azure OpenAI – JPM Certificate Auth Version)
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



AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")   # KEEP as-is
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
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


# ------------------------
# MAIN FUNCTION
# ------------------------
def generate_madl_for_method(method_code: str, class_name: str, parameters: str, language: str) -> Optional[Dict]:

    user_prompt = f"""
Analyze this method and decide if it is reusable.

Language: {language}
Class/File: {class_name}
Parameters: {parameters}

Method code:
{method_code}
""".strip()

    # 👉 IMPORTANT: Create a NEW Azure client for every call
    client = get_fresh_client()

    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,   # ✔ use your .env deployment
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

    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(raw)
    except Exception as e:
        logger.error(f"Failed JSON parse: {e}\nRAW:\n{raw}")
        return None

    if not isinstance(data, dict) or not data.get("reusable"):
        return None

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


