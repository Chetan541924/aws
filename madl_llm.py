# madl_llm.py
from __future__ import annotations
import json
from typing import Dict, Optional

import google.generativeai as genai
from madl_settings import GEMINI_API_KEY

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Use a supported model
_model = genai.GenerativeModel("gemini-2.5-flash-lite")

SYSTEM_PROMPT = """
You analyze code and determine whether a method is a reusable automation method.

Reusable methods must:
- Perform a meaningful automation action (login, navigate, fill form, validate UI, etc.)
- NOT be trivial helpers or getters/setters.
- Be general enough to reuse across multiple test cases.

You MUST return STRICT JSON.

IF reusable:
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

IF NOT reusable:
{
  "reusable": false
}
"""


def generate_madl_for_method(
    method_code: str,
    class_name: str,
    parameters: str,
    language: str,
) -> Optional[Dict]:
    """
    Returns MADL JSON dict if reusable, otherwise None.
    """

    user_prompt = f"""
Analyze this method and decide if it is reusable.

Language: {language}
Class/File: {class_name}
Parameters: {parameters}

Method code:
{method_code}
"""

    try:
        response = _model.generate_content(
            [
                {"role": "model", "parts": [SYSTEM_PROMPT]},
                {"role": "user", "parts": [user_prompt]},
            ],
            generation_config={"temperature": 0.1},
        )
    except Exception as e:
        print("Gemini Error:", e)
        return None

    if not response or not hasattr(response, "text"):
        return None

    raw = response.text.strip()

    # Handle ```json blocks
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    # Parse JSON
    try:
        data = json.loads(raw)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None
    if not data.get("reusable"):
        return None

    # Normalize output
    return {
        "reusable": True,
        "method_name": data.get("method_name", "").strip(),
        "class_name": data.get("class_name", class_name).strip(),
        "intent": data.get("intent", "").strip(),
        "semantic_description": data.get("semantic_description", "").strip(),
        "keywords": data.get("keywords", []),
        "parameters": data.get("parameters", parameters).strip(),
        "method_code": data.get("method_code", method_code).strip(),
    }
