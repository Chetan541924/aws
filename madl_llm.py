from __future__ import annotations
import json
import os
import logging
from typing import Dict, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env (AZURE_OPENAI_* vars)
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT):
    raise RuntimeError("Missing Azure OpenAI config in environment (.env).")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

SYSTEM_PROMPT = """
You analyze code and determine whether a method is a reusable automation method.

Reusable methods must:
- Perform a meaningful automation action (login, navigate, fill form, validate UI, etc.)
- NOT be trivial helpers or getters/setters
- Be general enough to reuse across multiple test cases

You MUST return STRICT JSON with NO additional text, markdown, or explanation.

IF reusable, return:
{
  "reusable": true,
  "method_name": "...",
  "class_name": "...",
  "intent": "...",
  "semantic_description": "...",
  "keywords": ["...", "..."],
  "parameters": "...",
  "method_code": "..."
}

IF NOT reusable, return:
{
  "reusable": false
}
""".strip()


def generate_madl_for_method(
    method_code: str,
    class_name: str,
    parameters: str,
    language: str,
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
        logger.error(f"Azure OpenAI error: {e}", exc_info=True)
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

    # Validate and construct MADL
    method_name = data.get("method_name", "").strip()
    intent = data.get("intent", "").strip()
    
    if not method_name or not intent:
        logger.warning("Missing required fields: method_name or intent")
        return None

    # Ensure keywords is a list
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
