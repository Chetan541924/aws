from typing import Optional, List, Dict, Any
import json
import os
import logging
from app.routers.structured_logging import LogCategory

# We assume azure_client is imported from executions.py or a shared module
from app.routers.executions import azure_client


async def generate_script_with_madl(
    testcase_id: str,
    script_type: str,
    script_lang: str,
    testplan: dict,
    selected_madl_methods: Optional[List[dict]] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Azure OpenAI version of script generator.

    - Uses certificate-based Azure auth (Bearer + API key)
    - Supports reusable MADL methods
    - Removes markdown fences
    - Returns ONLY executable Python code
    """

    if azure_client is None:
        raise RuntimeError("Azure OpenAI client is not initialized")

    model = os.getenv("AZURE_OPENAI_MODEL")
    if not model:
        raise RuntimeError("AZURE_OPENAI_MODEL environment variable not set")

    if logger:
        logger.info(LogCategory.GENERATION, "Starting script generation (Azure OpenAI)")

    # ----------------------------------------------------
    # BUILD MADL CONTEXT
    # ----------------------------------------------------
    madl_context = ""
    if selected_madl_methods:
        madl_context += "\n\n# AVAILABLE REUSABLE METHODS (MADL):\n"
        for m in selected_madl_methods:
            madl_context += f"- {m.get('signature')}: {m.get('intent')}\n"

    # ----------------------------------------------------
    # BUILD PROMPT
    # ----------------------------------------------------
    prompt = f"""
Generate a test script for test case ID: {testcase_id}
Script type: {script_type}
Language: {script_lang}

Test plan JSON:
{json.dumps(testplan)}

{madl_context}

Requirements:
- If reusable MADL methods are provided, USE them when possible
- Include comments above each action describing the step
- Do NOT use pytest
- Every action must be inside a try/except block
- Add timestamped logs before and after each action
- On failure print: "Action <step> failed at <time> due to <error>"
- Output ONLY the script code. No markdown, no explanations.
"""

    # ----------------------------------------------------
    # CALL AZURE OPENAI
    # ----------------------------------------------------
    system_msg = (
        "You are an expert automation engineer. "
        "Your output must be ONLY runnable Python code. "
        "No markdown code fences, no explanations."
    )

    try:
        response = azure_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=3500
        )
        raw_text = response.choices[0].message["content"]

    except Exception as e:
        if logger:
            logger.error(LogCategory.GENERATION, f"Azure OpenAI error: {str(e)}")
        raise RuntimeError(f"Azure OpenAI script generation failed: {str(e)}")

    # ----------------------------------------------------
    # CLEAN CODE FENCES
    # ----------------------------------------------------
    if not raw_text:
        raise ValueError("Azure OpenAI returned empty script")

    cleaned = raw_text.strip()

    # Remove ``` fences if any
    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) > 1:
            block = parts[1]
            # Remove language tag
            lines = block.splitlines()
            if lines and lines[0].lower().startswith("python"):
                lines = lines[1:]
            cleaned = "\n".join(lines).strip()

    if logger:
        logger.success(LogCategory.GENERATION, f"Script generated ({len(cleaned)} bytes)")

    return cleaned


async def collect_enhanced_error_context(
    logs: str,
    testplan: str,
    generated_script: str
) -> Dict[str, Any]:
    """
    Collects enhanced diagnostic context for Azure self-healing.
    """

    try:
        error_context = {
            "execution_logs": logs,
            "testplan": testplan,
            "generated_script": generated_script,
            "timestamp": datetime.now().isoformat(),
            "diagnostics": {
                "error_patterns": [],
                "failed_actions": []
            }
        }

        # Extract error lines
        errors = [
            line.strip()
            for line in logs.split('\n')
            if "error" in line.lower() or "failed" in line.lower() or "exception" in line.lower()
        ]

        error_context["diagnostics"]["error_patterns"] = errors[:10]
        error_context["diagnostics"]["failed_actions"] = errors

        return error_context

    except Exception as e:
        utils.logger.error(f"[HEALING] Error context collection failed: {str(e)}")
        return {"error": str(e), "execution_logs": logs}
