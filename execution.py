from fastapi import (
    APIRouter,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    UploadFile,        # ✅ ADD THIS
    File               # (optional – if you use File(...))
)

from typing import List, Dict, Any
from fastapi.responses import StreamingResponse
import json
import subprocess
import logging
import os
import tempfile
from datetime import datetime

from app import models
from app import utils
from app import database as db
from app.routers.users import get_current_any_user

from concurrent.futures import ThreadPoolExecutor
import asyncio
import sys
import io
import traceback

from selenium import webdriver
import google.generativeai as genai
from app import config

# MADL Imports
from app.routers.users import get_current_any_user
#from app.routers.madl_integration import madl_client, search_for_reusable_methods
#from app.routers.madl_storage import store_successful_execution_to_madl
#from app.routers.structured_logging import extract_madl_from_logs
from app.routers.structured_logging import StructuredLogger, LogLevel, LogCategory
from app.routers import ai_healing


router = APIRouter()

genai.configure(api_key=config.GEMINI_API_KEY)

async def generate_script(testcase_id: str, script_type: str, script_lang: str, testplan: dict):
    """Generate test script using Gemini AI"""
    try:
        prompt = f"Generate a test script for test case ID: {testcase_id}\n"
        prompt += f"Script type: {script_type}, Language: {script_lang}\n"
        prompt += "Test plan JSON: " + json.dumps(testplan) + "\n"
        prompt += "Requirements:\n"
        prompt += "- Include comments above each action describing the step.\n"
        prompt += "- Don't use pytest\n"
        prompt += "- Wrap each action in a try-catch block.\n"
        prompt += "- Add print statements with timestamps before and after each action (e.g., 'Running action: <step> at <timestamp>' and 'Action runned: <step> at <timestamp>').\n"
        prompt += "- If an action fails, print 'Action <step> failed at <timestamp> due to: <error>'.\n"
        prompt += "- Use appropriate imports and syntax for the chosen script type and language.\n"
        prompt += "- Handle actions: 'Navigate to login', 'Enter credentials', 'Submit form' (assume credentials are in 'user/pass' format, split by '/').\n"
        prompt += "- Output only the code, no additional explanations or markdown (e.g., no ''' or # comments outside actions).\n"

        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        script_content = response.text.strip()

        if not script_content:
            raise ValueError("Gemini returned empty script content")

        return script_content
    except Exception as e:
        raise Exception(f"Script generation failed: {str(e)}")

async def execute_with_auto_healing(
    testcase_id: str,
    script_type: str,
    conn,
    user_id: str
):
    """
    Execute test script with automatic self-healing on failure.
    Steps:
    1. Execute script
    2. If fails, collect error context (logs, screenshot, DOM)
    3. Call self-healing API
    4. Re-execute healed script
    """
    from app.routers import ai_healing
    import tempfile
    
    try:
        # Fetch script
        script_row = await conn.fetchrow(
            "SELECT script FROM testscript WHERE testcaseid = $1",
            testcase_id
        )
        if not script_row or not script_row["script"]:
            return {
                "status": "FAILED",
                "message": "No script found",
                "healed": False,
                "logs": []
            }
        
        # Parse script
        script_json = script_row["script"]
        script_data = json.loads(script_json)
        script_obj = script_data.get("script", {})
        script_content_lines = script_obj.get("code", [])
        script_content = '\n'.join(line for line in script_content_lines if line.strip())
        
        # Fetch test plan
        testplan_row = await conn.fetchrow(
            "SELECT * FROM testplan WHERE testcaseid = $1",
            testcase_id
        )
        testplan_output = json.dumps(dict(testplan_row)) if testplan_row else "{}"
        
        # First execution attempt
        logs = []
        temp_file_path = None
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(script_content)
                temp_file_path = temp_file.name
            
            utils.logger.info(f"[EXEC] Executing script: {temp_file_path}")
            process = subprocess.Popen(
                [sys.executable, temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            stdout, stderr = process.communicate()
            logs.append(stdout)
            
            if process.returncode == 0:
                utils.logger.info(f"[EXEC] Script executed successfully for {testcase_id}")
                return {
                    "status": "SUCCESS",
                    "message": "Script executed successfully",
                    "healed": False,
                    "logs": logs,
                    "output": stdout
                }
            else:
                # First execution failed - trigger self-healing
                execution_logs = stderr or stdout
                
                utils.logger.warning(f"[HEALING] Script failed for {testcase_id}, triggering self-healing...")
                utils.logger.info(f"[HEALING] Error logs: {execution_logs[:200]}")
                
                try:
                    healed_response = await ai_healing.self_heal(
                        testplan_output=testplan_output,
                        generated_script=script_content,
                        execution_logs=execution_logs,
                        screenshot=None,
                        dom_snapshot=None
                    )
                    
                    # Extract healed code from Response object
                    healed_code = healed_response.body.decode('utf-8') if hasattr(healed_response, 'body') else str(healed_response)
                    logs.append(f"\n[SELF-HEALING] Healed script generated:\n{healed_code[:200]}...")
                    
                    # Execute healed script
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as healed_file:
                        healed_file.write(healed_code)
                        healed_temp_path = healed_file.name
                    
                    utils.logger.info(f"[HEALING] Executing healed script: {healed_temp_path}")
                    healed_process = subprocess.Popen(
                        [sys.executable, healed_temp_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',
                        errors='replace'
                    )
                    
                    healed_stdout, healed_stderr = healed_process.communicate()
                    logs.append(healed_stdout)
                    
                    if healed_process.returncode == 0:
                        utils.logger.info(f"[HEALING] Healed script executed successfully for {testcase_id}")
                        return {
                            "status": "SUCCESS",
                            "message": "Script executed successfully after self-healing",
                            "healed": True,
                            "logs": logs,
                            "output": healed_stdout
                        }
                    else:
                        utils.logger.error(f"[HEALING] Healed script still failed: {healed_stderr}")
                        return {
                            "status": "FAILED",
                            "message": f"Script failed even after self-healing: {healed_stderr}",
                            "healed": True,
                            "logs": logs,
                            "output": healed_stderr
                        }
                    
                finally:
                    if 'healed_temp_path' in locals() and os.path.exists(healed_temp_path):
                        os.unlink(healed_temp_path)
        
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    except Exception as e:
        utils.logger.error(f"[HEALING] Auto-healing failed: {str(e)}")
        return {
            "status": "FAILED",
            "message": f"Auto-healing error: {str(e)}",
            "healed": False,
            "logs": logs if 'logs' in locals() else []
        }

from jose import jwt, JWTError
from jose import jwt, JWTError
from fastapi import WebSocket, WebSocketDisconnect
import json
import tempfile
import subprocess
import sys
import asyncio
import os
from datetime import datetime

from app import utils, config, database as db
#from app.routers.madl_integration import search_for_reusable_methods
from app.routers.execution_enhanced import collect_enhanced_error_context, generate_script_with_madl
#from app.routers import madl_storage

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


@router.get("/execution", response_model=List[Dict])
async def get_all_execution_logs(current_user: dict = Depends(get_current_any_user)):
    conn = None
    try:
        conn = await db.get_db_connection()
        userid = current_user["userid"]

        # 1. Get all project IDs the user is assigned to
        user_projects = await conn.fetchrow(
            "SELECT projectid FROM projectuser WHERE userid = $1",
            userid
        )
        if not user_projects or not user_projects["projectid"]:
            return []

        allowed_project_ids = set(user_projects["projectid"])

        # 2. Get all test case IDs in those projects
        testcases = await conn.fetch(
            """
            SELECT testcaseid FROM testcase
            WHERE projectid && $1::varchar[]
            """,
            list(allowed_project_ids)
        )
        if not testcases:
            return []

        accessible_testcase_ids = [tc["testcaseid"] for tc in testcases]

        # 3. Fetch execution logs
        logs = await conn.fetch(
            """
            SELECT exeid, testcaseid, scripttype, datestamp, exetime, message, output, status
            FROM execution
            WHERE testcaseid = ANY($1::varchar[])
            ORDER BY datestamp DESC, exetime DESC
            """,
            accessible_testcase_ids
        )

        return [dict(log) for log in logs]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching execution logs: {str(e)}")
    finally:
        if conn:
            await conn.close()
            
@router.post("/execute-code")
async def execute_code(
    file: UploadFile,
    script_type: str,
    current_user: dict = Depends(get_current_any_user)
):
    try:
        filename = file.filename.lower()

        # ---- Validate extension ----
        if not (filename.endswith(".py") or filename.endswith(".java")):
            raise HTTPException(status_code=400, detail="Only .py or .java files are supported.")

        # ---- Store as temp file ----
        suffix = ".py" if filename.endswith(".py") else ".java"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # =====================================================================
        # -------------------------- PYTHON EXECUTION --------------------------
        # =====================================================================
        if filename.endswith(".py"):

            script_type_lower = script_type.lower()
            if script_type_lower not in ["selenium", "playwright"]:
                raise HTTPException(status_code=400, detail="script_type must be 'selenium' or 'playwright'.")

            # Validate required library
            required_lib = {"selenium": "selenium", "playwright": "playwright"}[script_type_lower]
            try:
                __import__(required_lib)
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail=f"{required_lib} is not installed on server."
                )

            cmd = ["python", temp_path]

        # =====================================================================
        # --------------------------- JAVA EXECUTION ---------------------------
        # =====================================================================
        else:
            class_name = os.path.splitext(os.path.basename(filename))[0]

            compile_cmd = ["javac", temp_path]
            try:
                subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                os.unlink(temp_path)
                raise HTTPException(
                    status_code=500,
                    detail=f"Java compilation failed:\n{e.stderr}"
                )

            cmd = ["java", class_name]

        # =====================================================================
        # ----------------------- EXECUTION STREAM LOGS ------------------------
        # =====================================================================

        def generate_logs():
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Stream output line by line
            while True:
                out_line = process.stdout.readline()
                err_line = process.stderr.readline()

                if out_line:
                    yield f"[{datetime.now().strftime('%H:%M:%S')}] {out_line.strip()}\n"

                if err_line:
                    yield f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {err_line.strip()}\n"

                if process.poll() is not None:
                    break

            # Send remaining buffered output
            for remaining in process.stdout:
                yield f"[{datetime.now().strftime('%H:%M:%S')}] {remaining.strip()}\n"

            for remaining in process.stderr:
                yield f"[{datetime.now().strftime('%H:%M:%S')}] ERROR: {remaining.strip()}\n"

            # Cleanup temp file after execution
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return StreamingResponse(generate_logs(), media_type="text/plain")

    except Exception as e:
        utils.logger.error(f"Unexpected error: {e}")
        # ensure cleanup in case failure happens before generator starts
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/projects/{project_id}/executions/summary")
async def get_execution_summary(
    project_id: str,
    limit: int = 10,
    current_user: dict = Depends(get_current_any_user)
):
    """Get summary of last N executions for a project"""
    conn = None
    try:
        conn = await db.get_db_connection()
        userid = current_user["userid"]

        # Verify user has access to project
        access = await conn.fetchrow(
            "SELECT 1 FROM projectuser WHERE userid = $1 AND $2 = ANY(projectid)",
            userid, project_id
        )
        if not access:
            raise HTTPException(status_code=403, detail="You are not assigned to this project")

        # Get test cases in project
        testcases = await conn.fetch(
            "SELECT testcaseid FROM testcase WHERE $1 = ANY(projectid)",
            project_id
        )
        if not testcases:
            return {
                "total_executions": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "recent_executions": []
            }

        testcase_ids = [tc["testcaseid"] for tc in testcases]

        # Get execution statistics
        stats = await conn.fetchrow(
            """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed
            FROM execution
            WHERE testcaseid = ANY($1::varchar[])
            """,
            testcase_ids
        )

        # Get last N executions
        recent = await conn.fetch(
            """
            SELECT exeid, testcaseid, scripttype, datestamp, exetime, status, message
            FROM execution
            WHERE testcaseid = ANY($1::varchar[])
            ORDER BY datestamp DESC, exetime DESC
            LIMIT $2
            """,
            testcase_ids, limit
        )

        total = stats["total"] or 0
        successful = stats["successful"] or 0
        failed = stats["failed"] or 0
        success_rate = (successful / total * 100) if total > 0 else 0.0

        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": round(success_rate, 2),
            "recent_executions": [
                {
                    "exeid": r["exeid"],
                    "testcaseid": r["testcaseid"],
                    "scripttype": r["scripttype"],
                    "datestamp": str(r["datestamp"]),
                    "exetime": str(r["exetime"]),
                    "status": r["status"],
                    "message": r["message"]
                }
                for r in recent
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching execution summary: {str(e)}")
    finally:
        if conn:
            await conn.close()

@router.get("/projects/{project_id}/executions/history")
async def get_execution_history(
    project_id: str,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_any_user)
):
    """Get paginated execution history for a project"""
    conn = None
    try:
        conn = await db.get_db_connection()
        userid = current_user["userid"]

        # Verify user has access to project
        access = await conn.fetchrow(
            "SELECT 1 FROM projectuser WHERE userid = $1 AND $2 = ANY(projectid)",
            userid, project_id
        )
        if not access:
            raise HTTPException(status_code=403, detail="You are not assigned to this project")

        # Get test cases in project
        testcases = await conn.fetch(
            "SELECT testcaseid FROM testcase WHERE $1 = ANY(projectid)",
            project_id
        )
        testcase_ids = [tc["testcaseid"] for tc in testcases] if testcases else []

        if not testcase_ids:
            return {"total": 0, "executions": []}

        # Get total count
        count_result = await conn.fetchval(
            "SELECT COUNT(*) FROM execution WHERE testcaseid = ANY($1::varchar[])",
            testcase_ids
        )

        # Get paginated executions
        executions = await conn.fetch(
            """
            SELECT exeid, testcaseid, scripttype, datestamp, exetime, status, message, output
            FROM execution
            WHERE testcaseid = ANY($1::varchar[])
            ORDER BY datestamp DESC, exetime DESC
            LIMIT $2 OFFSET $3
            """,
            testcase_ids, limit, offset
        )

        return {
            "total": count_result,
            "executions": [
                {
                    "exeid": e["exeid"],
                    "testcaseid": e["testcaseid"],
                    "scripttype": e["scripttype"],
                    "datestamp": str(e["datestamp"]),
                    "exetime": str(e["exetime"]),
                    "status": e["status"],
                    "message": e["message"],
                    "output": e["output"]
                }
                for e in executions
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching execution history: {str(e)}")
    finally:
        if conn:
            await conn.close()
