# ai_healing.py
# Fully converted from Gemini → Azure OpenAI (Certificate + API key)
# Compatible with JPM security requirements
# Author: ChatGPT (rewritten for your infra)

import os
import json
import logging
from fastapi import HTTPException
from azure.identity import CertificateCredential
from openai import AzureOpenAI

logger = logging.getLogger("ai_healing")
logger.setLevel(logging.INFO)


# ======================================================
# 1) CERTIFICATE + BEARER TOKEN AUTH
# ======================================================

def _get_bearer_token() -> str:
    """
    Returns a valid Azure AD token using JP Morgan's certificate-based auth.
    Requires these env vars:
        CERTIFICATE_PATH
        AZURE_TENANT_ID
        AZURE_CLIENT_ID
    """
    cert_path = os.getenv("CERTIFICATE_PATH")
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID")

    if not cert_path or not tenant_id or not client_id:
        raise RuntimeError("Missing Azure certificate auth environment variables")

    cred = CertificateCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        certificate_path=cert_path
    )

    scope = "https://cognitiveservices.azure.com/.default"
    return cred.get_token(scope).token


def _build_azure_client() -> AzureOpenAI:
    """
    Creates an AzureOpenAI client configured for JPM certificate auth.
    """

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint or not api_key:
        raise RuntimeError("Azure OpenAI endpoint/API key not configured")

    token = _get_bearer_token()

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        default_headers={
            "Authorization": f"Bearer {token}",
            "api-key": api_key
        }
    )
    return client


# Instantiate global Azure OpenAI client
try:
    azure_client = _build_azure_client()
    logger.info("[AI_HEALING] Azure OpenAI client initialized")
except Exception as e:
    azure_client = None
    logger.error(f"[AI_HEALING] Failed to initialize Azure client: {e}")


# ======================================================
# 2) SELF HEALING FUNCTION (MAIN ENTRYPOINT)
# ======================================================

async def self_heal(
    testplan_output: str,
    generated_script: str,
    execution_logs: str,
    screenshot: str = None,
    dom_snapshot: str = None
):
    """
    Generates a healed version of the user's script.
    Uses Azure OpenAI GPT model.
    """

    if azure_client is None:
        raise HTTPException(status_code=500, detail="Azure OpenAI Client is not initialized")

    model = os.getenv("AZURE_OPENAI_MODEL")
    if not model:
        raise RuntimeError("AZURE_OPENAI_MODEL is not configured")

    logger.info("[AI_HEALING] Healing request started")

    # Build system prompt
    system_message = (
        "You are a highly skilled AI debugging assistant. "
        "Your job is to FIX the provided Python test automation script. "
        "You must output ONLY valid runnable Python code. "
        "Do NOT include any explanations, markdown, fences, or comments outside the script. "
        "If the script cannot be fully healed, still output the corrected script best as possible."
    )

    # -------------------------
    # BUILD USER PAYLOAD
    # -------------------------
    user_payload = {
        "testplan": testplan_output,
        "broken_script": generated_script,
        "logs": execution_logs
    }

    if screenshot:
        user_payload["screenshot"] = screenshot[:5000]

    if dom_snapshot:
        user_payload["dom_snapshot"] = dom_snapshot[:5000]

    user_message = (
        "A test script failed during execution.\n"
        "Here is the information you must use to heal the script:\n\n"
        f"=== TEST PLAN JSON ===\n{testplan_output}\n\n"
        f"=== ORIGINAL SCRIPT ===\n{generated_script}\n\n"
        f"=== EXECUTION LOGS ===\n{execution_logs}\n\n"
        "If provided, here are failure context snapshots:\n"
        f"Screenshot (truncated): {str(screenshot)[:3000] if screenshot else 'None'}\n"
        f"DOM (truncated): {str(dom_snapshot)[:3000] if dom_snapshot else 'None'}\n\n"
        "Now FIX the script. Output ONLY corrected runnable Python code."
    )

    # ======================================================
    # 3) MAKE AZURE OPENAI CALL
    # ======================================================

    try:
        response = azure_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=4000
        )

        healed_code = response.choices[0].message["content"]

        if not healed_code or len(healed_code.strip()) == 0:
            raise RuntimeError("Azure OpenAI returned empty healed script")

        logger.info("[AI_HEALING] Healed script generated successfully")

        # return RAW text, NOT wrapped in JSON, because execution pipeline expects STRING
        class SimpleResponse:
            body = healed_code.encode("utf-8")

        return SimpleResponse()

    except Exception as e:
        logger.error(f"[AI_HEALING] Healing failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI healing failed: {str(e)}")
