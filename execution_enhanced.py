"""
Enhanced Execution Router with MADL Integration
Incorporates MADL search, method selection, enhanced logging, and error context.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, Form
from typing import List, Dict, Any, Optional
import json
import subprocess
import tempfile
import os
import asyncio
import sys
from datetime import datetime
import google.generativeai as genai

from app import models as models
from app import utils
from app import database as db
from app import config
from app.routers.users import get_current_any_user
#from app.routers.madl_integration import madl_client, search_for_reusable_methods
#from app.routers.madl_storage import store_successful_execution_to_madl
from app.routers.structured_logging import StructuredLogger, LogLevel, LogCategory
from app.routers import ai_healing

router = APIRouter()

genai.configure(api_key=config.GEMINI_API_KEY)


async def generate_script_with_madl(
    testcase_id: str,
    script_type: str,
    script_lang: str,
    testplan: dict,
    selected_madl_methods: Optional[List[dict]] = None,
    logger: Optional[StructuredLogger] = None
):
    """
    Generate test script using Gemini AI with MADL method integration.

    Enhanced to:
    - Include selected MADL methods in prompt
    - Strip markdown code fences (```python ... ```) from the model output
    """
    try:
        if logger:
            logger.info(LogCategory.GENERATION, "Starting script generation with Gemini")

        # Build MADL methods context
        madl_context = ""
        if selected_madl_methods:
            madl_context = "\n\n# AVAILABLE REUSABLE METHODS (from MADL):\n"
            for method in selected_madl_methods:
                madl_context += f"- {method['signature']}: {method['intent']}\n"
                madl_context += f"  Example: {method['example']}\n"

        prompt = f"""
        Generate a test script for test case ID: {testcase_id}
        Script type: {script_type}, Language: {script_lang}
        Test plan JSON: {json.dumps(testplan)}

        {madl_context}

        Requirements:
        - If AVAILABLE REUSABLE METHODS are provided, USE them where applicable
        - Include comments above each action describing the step
        - Don't use pytest
        - Wrap each action in try-catch block
        - Add print statements with timestamps before and after each action
        - Format: 'Running action: <step> at <timestamp>' and 'Action completed: <step> at <timestamp>'
        - If action fails, print 'Action <step> failed at <timestamp> due to: <error>'
        - Handle errors gracefully with context collection (screenshot, DOM snapshot)
        - Use appropriate imports and syntax
        - Output ONLY the code, no additional explanations or markdown
        """

        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)

        raw_text = (response.text or "").strip()
        if not raw_text:
            raise ValueError("Gemini returned empty script content")

        # ---- CLEAN MARKDOWN CODE FENCES ----
        script_content = raw_text
        if "```" in raw_text:
            # split on fences
            parts = raw_text.split("```")
            # typical structure: 0: before, 1: "python\ncode...", 2: after
            if len(parts) >= 2:
                code_block = parts[1]

                # remove optional language tag on first line
                lines = code_block.splitlines()
                if lines and lines[0].strip().lower().startswith("python"):
                    lines = lines[1:]

                script_content = "\n".join(lines).strip()
        else:
            script_content = raw_text

        if not script_content:
            raise ValueError("Script content empty after cleaning code fences")

        if logger:
            logger.success(
                LogCategory.GENERATION,
                f"Script generated ({len(script_content)} bytes, cleaned markdown fences)"
            )

        return script_content

    except Exception as e:
        if logger:
            logger.error(LogCategory.GENERATION, f"Script generation failed: {str(e)}")
        raise Exception(f"Script generation failed: {str(e)}")

async def collect_enhanced_error_context(
    logs: str,
    testplan: str,
    generated_script: str
) -> Dict[str, Any]:
    """
    Collect comprehensive error context for self-healing
    Includes execution logs, test plan, script, and attempts to extract diagnostics
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
        
        # Extract error patterns from logs
        error_lines = [line for line in logs.split('\n') if 'error' in line.lower() or 'failed' in line.lower()]
        error_context["diagnostics"]["error_patterns"] = error_lines[:10]  # Top 10 errors
        
        # Extract failed actions
        for line in logs.split('\n'):
            if 'failed' in line.lower() or 'exception' in line.lower():
                error_context["diagnostics"]["failed_actions"].append(line.strip())
        
        return error_context
    
    except Exception as e:
        utils.logger.error(f"[HEALING] Error context collection failed: {str(e)}")
        return {"error": str(e), "execution_logs": logs}

# Use this function to replace your current endpoint implementation
# backend: patched endpoint
@router.websocket("/testcases/{testcase_id}/execute-with-madl")
async def execute_testcase_with_madl(
    websocket: WebSocket,
    testcase_id: str,
    script_type: str
):
    """
    Modified endpoint:
    - Builds testplan (unchanged)
    - Sends PLAN_READY
    - Sends REQUEST_EDIT with testplan and waits (blocks) for frontend to send either:
        - {"type":"EDITED_TESTPLAN", "testplan_json": "..."}  OR
        - {"type":"SKIP_EDIT"}
      Backend proceeds using edited plan if provided (does NOT persist to DB).
    - Continues with generation → execution → auto-heal → logging pipeline unchanged.
    """
    await websocket.accept()
    utils.logger.debug(f"[EXEC] WebSocket opened for {testcase_id}, {script_type}")

    # ---------------------------
    # 1. Extract & validate token
    # ---------------------------
    token = None
    if "headers" in websocket.scope:
        headers = dict(websocket.scope["headers"])
        auth_header = headers.get(b"authorization")
        if auth_header and isinstance(auth_header, bytes) and auth_header.startswith(b"Bearer "):
            token = auth_header.decode().split("Bearer ")[1].strip()

    if not token:
        await websocket.send_text(json.dumps({"error": "Authorization token missing", "status": "FAILED"}))
        await websocket.close()
        return

    from jose import jwt, JWTError
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        userid = payload.get("userid")
        role = payload.get("role")
        current_user = {"userid": userid, "role": role}
    except JWTError:
        await websocket.send_text(json.dumps({"error": "Invalid token", "status": "FAILED"}))
        await websocket.close()
        return

    conn = None
    logger = StructuredLogger(testcase_id)

    try:
        conn = await db.get_db_connection()
        userid = current_user["userid"]
        logger.info(LogCategory.INITIALIZATION, f"Execution started for {testcase_id}")

        # -----------------------------
        # 2. Validate access & test case
        # -----------------------------
        tc_project = await conn.fetchrow("SELECT projectid FROM testcase WHERE testcaseid = $1", testcase_id)
        if not tc_project:
            await websocket.send_text(json.dumps({"error": "Test case not found", "status": "FAILED"}))
            await websocket.close()
            return

        access = await conn.fetchrow(
            "SELECT 1 FROM projectuser WHERE userid = $1 AND projectid && $2",
            userid, tc_project["projectid"]
        )
        if not access:
            await websocket.send_text(json.dumps({"error": "Unauthorized", "status": "FAILED"}))
            await websocket.close()
            return

        # -----------------------------
        # 3. Build test plan (unchanged)
        # -----------------------------
        await websocket.send_text(json.dumps({"status": "BUILDING_PLAN", "log": "Building test plan..."}))
        logger.info(LogCategory.PLAN_BUILDING, "Building test plan from prerequisites and steps")

        prereq_chain = await utils.get_prereq_chain(conn, testcase_id)
        testplan_dict = {
            "pretestid_steps": {},
            "current_testid": testcase_id,
            "current_bdd_steps": {}
        }

        # Prerequisite testcases
        for tc_id in prereq_chain[:-1]:
            steps_row = await conn.fetchrow("SELECT steps, args FROM teststep WHERE testcaseid = $1", tc_id)
            if steps_row and steps_row["steps"]:
                testplan_dict["pretestid_steps"][tc_id] = dict(zip(steps_row["steps"], steps_row["args"]))

        # Current testcase steps
        current_steps = await conn.fetchrow("SELECT steps, args FROM teststep WHERE testcaseid = $1", testcase_id)
        if current_steps and current_steps["steps"]:
            testplan_dict["current_bdd_steps"] = dict(zip(current_steps["steps"], current_steps["args"]))

        testplan_json = json.dumps(testplan_dict)
        await websocket.send_text(json.dumps({"status": "PLAN_READY", "log": "Test plan built"}))
        logger.success(LogCategory.PLAN_BUILDING, "Test plan built successfully")

        # ---------------------------------------------------
        # NEW: Request frontend to edit (blocking) — WAIT FOR RESPONSE
        # ---------------------------------------------------
        await websocket.send_text(json.dumps({
            "status": "REQUEST_EDIT",
            "log": "Please edit test data or skip",
            "testplan": testplan_dict
        }))
        logger.info(LogCategory.PLAN_BUILDING, "Sent REQUEST_EDIT to frontend; waiting for user response")

        try:
            edit_msg = await websocket.receive_text()   # block until client responds (or disconnects)
            try:
                edit_data = json.loads(edit_msg)
            except Exception:
                edit_data = {}

            if edit_data.get("type") == "EDITED_TESTPLAN":
                edited_plan = edit_data.get("testplan_json")
                if edited_plan:
                    testplan_json = edited_plan
                    try:
                        testplan_dict = json.loads(testplan_json)
                    except Exception:
                        logger.warning(LogCategory.PLAN_BUILDING, "Edited testplan JSON invalid; continuing with original plan")
                logger.info(LogCategory.PLAN_BUILDING, "Received edited testplan from frontend")
            elif edit_data.get("type") == "SKIP_EDIT":
                logger.info(LogCategory.PLAN_BUILDING, "User skipped editing testplan")
            else:
                # unexpected payload — continue with original plan
                logger.info(LogCategory.PLAN_BUILDING, "No edit received (unexpected payload); continuing with original plan")
        except WebSocketDisconnect:
            logger.warning(LogCategory.PLAN_BUILDING, "Client disconnected while waiting for edit; aborting execution")
            await websocket.close()
            return
        except Exception as e:
            logger.error(LogCategory.PLAN_BUILDING, f"Error while waiting for edited plan: {str(e)}")
            # Continue with original plan if waiting fails

        # ---------------------------------------------------
        # 4. (REMOVED MADL SEARCH) - send stub message instead
        # ---------------------------------------------------
        await websocket.send_text(json.dumps({
            "status": "NO_MADL_METHODS",
            "log": "MADL search skipped in this environment"
        }))
        logger.info(LogCategory.SEARCH, "MADL disabled for this deployment")

        selected_methods = None  # Placeholder for generator

        # ------------------------------
        # 5. Generate script (same logic)
        # ------------------------------
        await websocket.send_text(json.dumps({"status": "GENERATING", "log": "Generating script..."}))
        try:
            generated_script = await generate_script_with_madl(
                testcase_id=testcase_id,
                script_type=script_type,
                script_lang="python",
                testplan=testplan_dict,
                selected_madl_methods=None,
                logger=logger
            )
        except Exception as e:
            await websocket.send_text(json.dumps({"error": f"Generation failed: {str(e)}", "status": "FAILED"}))
            await websocket.close()
            return

        # ------------------------------------
        # 6. Execute (unchanged pipeline)
        # ------------------------------------
        first_attempt_passed = False
        autoheal_attempt_passed = False
        execution_status = None
        execution_message = None

        await websocket.send_text(json.dumps({"status": "EXECUTING", "log": "Starting execution..."}))
        execution_output = ""
        temp_file_path = None

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(generated_script)
                temp_file_path = temp_file.name

            logger.info(LogCategory.EXECUTION, f"Executing script from {temp_file_path}")

            process = subprocess.Popen(
                [sys.executable, temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1
            )

            for line in process.stdout:
                line = line.rstrip('\n')
                if line.strip():
                    execution_output += line + "\n"
                    await websocket.send_text(json.dumps({"status": "RUNNING", "log": line}))
                    await asyncio.sleep(0.02)

            return_code = process.wait()

            if return_code == 0:
                logger.success(LogCategory.EXECUTION, "Script executed successfully")
                first_attempt_passed = True
                execution_status = "SUCCESS"
                execution_message = "Script executed successfully"

            else:
                logger.error(LogCategory.EXECUTION, "Script execution failed")

                # AUTO-HEAL
                error_context = await collect_enhanced_error_context(
                    logs=execution_output,
                    testplan=testplan_json,
                    generated_script=generated_script
                )

                await websocket.send_text(json.dumps({
                    "status": "AUTO_HEALING",
                    "log": "Execution failed. Starting auto-healing with context..."
                }))
                logger.info(LogCategory.HEALING, "Initiating auto-healing")

                try:
                    healed_response = await ai_healing.self_heal(
                        testplan_output=testplan_json,
                        generated_script=generated_script,
                        execution_logs=execution_output,
                        screenshot=None,
                        dom_snapshot=None
                    )

                    healed_code = healed_response.body.decode('utf-8') if hasattr(healed_response, 'body') else str(healed_response)

                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as healed_file:
                        healed_file.write(healed_code)
                        healed_temp_path = healed_file.name

                    logger.info(LogCategory.HEALING, "Executing healed script")

                    healed_process = subprocess.Popen(
                        [sys.executable, healed_temp_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        bufsize=1
                    )

                    healed_output = ""
                    for line in healed_process.stdout:
                        line = line.rstrip('\n')
                        if line.strip():
                            healed_output += line + "\n"
                            await websocket.send_text(json.dumps({
                                "status": "RUNNING",
                                "log": f"[AUTO-HEALED] {line}"
                            }))
                            await asyncio.sleep(0.02)

                    healed_return_code = healed_process.wait()

                    if healed_return_code == 0:
                        logger.success(LogCategory.HEALING, "Healed script executed successfully")
                        autoheal_attempt_passed = True
                        execution_output = healed_output
                        execution_message = "[AUTO-HEALED] Healed execution completed"
                    else:
                        logger.error(LogCategory.HEALING, "Healed script still failed")
                        execution_status = "FAILED"
                        execution_message = "[AUTO-HEALED] Script failed even after healing"
                        execution_output = healed_output

                    if 'healed_temp_path' in locals() and os.path.exists(healed_temp_path):
                        os.unlink(healed_temp_path)

                except Exception as healing_error:
                    logger.error(LogCategory.HEALING, f"Healing failed: {str(healing_error)}")
                    execution_status = "FAILED"
                    execution_message = f"Healing failed: {str(healing_error)}"

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

        # ------------------------------
        # 7. Final status normalization
        # ------------------------------
        if first_attempt_passed:
            final_status = "success"
        elif autoheal_attempt_passed:
            final_status = "healed"
        else:
            final_status = "fail"

        # ------------------------------
        # 8. Store into `execution` table
        # ------------------------------
        exeid = await utils.get_next_exeid(conn)
        datestamp = datetime.now().date()
        exetime = datetime.now().time()

        await conn.execute(
            """
            INSERT INTO execution (exeid, testcaseid, scripttype, datestamp, exetime, message, output, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            exeid, testcase_id, script_type, datestamp, exetime, execution_message, execution_output, final_status
        )

        # ----------------------------------------------------------------------
        # 9. (REMOVED) — STORAGE TO MADL / VECTOR DB
        #     Instead send a harmless message
        # ----------------------------------------------------------------------
        await websocket.send_text(json.dumps({
            "status": "STORAGE_SKIPPED",
            "log": "MADL storage disabled in this environment"
        }))
        logger.info(LogCategory.STORAGE, "MADL storage skipped")

        # ------------------------------
        # 10. Send final summary
        # ------------------------------
        await websocket.send_text(json.dumps({
            "status": "COMPLETED",
            "log": execution_message,
            "final_status": final_status,
            "summary": logger.get_summary()
        }))

        logger.success(LogCategory.INITIALIZATION, "Execution completed")

    except WebSocketDisconnect:
        logger.warning(LogCategory.INITIALIZATION, "Client disconnected")
    except Exception as e:
        logger.error(LogCategory.INITIALIZATION, f"Unexpected error: {str(e)}")
        try:
            await websocket.send_text(json.dumps({"error": str(e), "status": "FAILED"}))
        except:
            pass
    finally:
        if conn:
            await conn.close()
        try:
            await websocket.close()
        except:
            pass

        utils.logger.info(f"[EXEC] Execution finished for {testcase_id}")
        utils.logger.debug(f"[EXEC] Final logs:\n{logger.get_readable_logs()}")
