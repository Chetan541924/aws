from fastapi import Body, Query, HTTPException

@app.post("/generate-test-script/{testcase_id}", response_model=GeneratedScriptResponse)
async def generate_test_script(
    testcase_id: str,

    # Accept script_type from either query OR body
    script_type: str = Query(None),
    script_lang: str = Query(None),
    include_prereq: bool = Query(None),

    # Entire testplan still comes in body
    testplan: Dict[str, Any] = Body(None)
):
    try:
        # Body fallback handling
        body_data = testplan or {}

        # Extract from body if query missing
        if script_type is None:
            script_type = body_data.get("script_type")
        if script_lang is None:
            script_lang = body_data.get("script_lang")
        if include_prereq is None:
            include_prereq = body_data.get("include_prereq", False)

        # Extract testplan if nested inside body
        if "testplan" in body_data:
            testplan = body_data["testplan"]

        # Validation
        if not script_type:
            raise HTTPException(422, "script_type missing")
        if not script_lang:
            raise HTTPException(422, "script_lang missing")
        if testplan is None:
            raise HTTPException(422, "testplan missing")

        script_type = script_type.lower()
        script_lang = script_lang.lower()

        if script_type not in ["playwright", "selenium"]:
            raise HTTPException(400, "script_type must be playwright or selenium")

        if script_lang not in ["python", "java"]:
            raise HTTPException(400, "script_lang must be python or java")

        # (REMAINING LOGIC SAME AS BEFORE ↓)
        # ----------------------------------------------------
        # Build prompt, call GPT, parse JSON, return script
        # ----------------------------------------------------

        return GeneratedScriptResponse(
            testcase_id=testcase_id,
            script_type=script_type,
            script_lang=script_lang,
            generated_script=full_script,
            reusable_methods=reusable_methods,
            saved_count=len(reusable_methods),
            model_used=model_used
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"[GENERATE] Failed for {testcase_id}: {e}")
        raise HTTPException(500, f"Script generation failed: {str(e)}")
