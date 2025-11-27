from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

# Import new MADL engine pipeline (Azure OpenAI + OpenSearch)
from madl_engine.core import analyze_test_steps_for_reusability

router = APIRouter(
    prefix="/reusable-methods",
    tags=["Reusable Methods"]
)

# =======================================================================
# REQUEST / RESPONSE MODELS
# =======================================================================

class ReusableMethodsRequest(BaseModel):
    testcase_id: str
    generated_script: str = ""        # FULLY IGNORED
    bdd_steps: List[str] = []         # Only manual BDD steps are used


class MethodMatch(BaseModel):
    step_label: str
    step_details: str
    query: str
    method_name: str
    class_name: str
    score: float
    signature: str
    method_code: str | None = None


class ReusableMethodsResponse(BaseModel):
    testcase_id: str
    results: List[MethodMatch]


# =======================================================================
# Extract BDD from script
# =======================================================================
def extract_bdd_from_script(script: str):
    if not script:
        return []

    steps = []
    for line in script.split("\n"):
        line = line.strip()
        if line.startswith(("Given ", "When ", "Then ", "And ", "But ")):
            steps.append(line)

    return steps


# =======================================================================
# MAIN API ENDPOINT (works with AZURE + OPENSEARCH)
# =======================================================================
@router.post("/check", response_model=ReusableMethodsResponse)
async def check_reusable_methods(req: ReusableMethodsRequest):

    # 1️⃣ Extract BDD from script if provided
    steps = extract_bdd_from_script(req.generated_script)

    # 2️⃣ Fallback to explicit list
    if not steps:
        steps = [s.strip() for s in req.bdd_steps if s and s.strip()]

    if not steps:
        return ReusableMethodsResponse(testcase_id=req.testcase_id, results=[])

    try:
        # ⛓️ 3️⃣ Call the Azure + OpenSearch semantic analyzer
        grouped_results = analyze_test_steps_for_reusability(steps)

        final_results: List[MethodMatch] = []

        for item in grouped_results:

            step_label  = item.get("step_label", "")
            step_details = item.get("step_details", "")
            query_text = step_details     # UI Query column = entire BDD block

            matches = item.get("matches", [])

            if matches:
                top = matches[0]
                method_name = top.get("method_name", "")
                class_name  = top.get("class_name", "")
                score       = float(top.get("score", 0.0))
                signature   = top.get("signature", "")
                method_code = top.get("method_code", "")
            else:
                method_name = ""
                class_name  = ""
                score       = 0.0
                signature   = ""
                method_code = ""

            final_results.append(MethodMatch(
                step_label   = step_label,
                step_details = step_details,
                query        = query_text,
                method_name  = method_name,
                class_name   = class_name,
                score        = score,
                signature    = signature,
                method_code  = method_code,
            ))

        return ReusableMethodsResponse(
            testcase_id=req.testcase_id,
            results=final_results
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing reusable methods: {str(e)}"
        )
