from fastapi import (
    APIRouter,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    UploadFile,
    File
)

from typing import List, Dict, Any
from fastapi.responses import StreamingResponse
import json
import subprocess
import logging
import os
import tempfile
from datetime import datetime
import asyncio
import sys

from app import utils
import aiosqlite
from app.routers.users import get_current_any_user
from app.routers.structured_logging import StructuredLogger, LogLevel, LogCategory
from app.routers import ai_healing
DB_PATH = "genai.db"   # your SQLite file; change if needed

async def get_db():
    return await aiosqlite.connect(DB_PATH)


# -------------------------------------------------------
# 🔥 Azure OpenAI + Certificate Authentication (correct)
# -------------------------------------------------------
from azure.identity import CertificateCredential
from openai import AzureOpenAI
import traceback

router = APIRouter()

# -------------------------------------------------------
# Load Azure env vars
# -------------------------------------------------------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
CERT_PATH = os.getenv("CERTIFICATE_PATH")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_MODEL]):
    raise RuntimeError("❌ Missing required Azure env vars.")


# -------------------------------------------------------
# ⭐ Step 1 — Generate Bearer Token using PEM certificate
# -------------------------------------------------------
def get_bearer_token():
    credential = CertificateCredential(
        tenant_id=AZURE_TENANT_ID,
        client_id=AZURE_CLIENT_ID,
        certificate_path=CERT_PATH
    )
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    return token.token


# -------------------------------------------------------
# ⭐ Step 2 — Build AzureOpenAI Client (correct format)
# -------------------------------------------------------
def build_azure_client():
    bearer = get_bearer_token()

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        default_headers={
            "Authorization": f"Bearer {bearer}",
            "api-key": AZURE_OPENAI_API_KEY
        }
    )
    return client


azure_client = build_azure_client()


# -------------------------------------------------------
# ⭐ Step 3 — Script Generation using Azure ChatCompletion
# -------------------------------------------------------
async def generate_script(testcase_id: str, script_type: str, script_lang: str, testplan: dict):
    """Generate Python automation script via Azure OpenAI."""
    prompt = f"""
Generate a test script for test case ID: {testcase_id}
Script type: {script_type}, Language: {script_lang}
Test plan JSON: {json.dumps(testplan)}

Requirements:
- Include comments above each action describing the step.
- Don't use pytest.
- Wrap each action in try/except.
- Add print statements with timestamps before and after each action.
- Print failures with reasons.
- Output only code — no markdown, no explanation.
"""

    system_msg = (
        "You are an assistant that outputs only runnable Python test code. "
        "No markdown, no explanation. Only script content."
    )

    try:
        response = azure_client.chat.completions.create(
            model=AZURE_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=2800
        )

        script = response.choices[0].message.content.strip()
        return script

    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Azure OpenAI script generation failed: {e}")


# -------------------------------------------------------
# ⭐ WebSocket Execution Endpoint
# -------------------------------------------------------
@router.websocket("/testcases/{testcase_id}/execute-with-madl")
async def execute_testcase_with_madl(websocket: WebSocket, testcase_id: str, script_type: str):

    await websocket.accept()
    logger = StructuredLogger(testcase_id)

    # Validate token
    token_bytes = websocket.scope["headers"]
    auth = dict(token_bytes).get(b"authorization", b"").decode()
    if not auth.startswith("Bearer "):
        await websocket.send_text(json.dumps({"error": "Missing token"}))
        return

    token = auth.split("Bearer ")[1]
    try:
        from jose import jwt
        payload = jwt.decode(token, utils.config.SECRET_KEY, algorithms=[utils.config.ALGORITHM])
        userid = payload["userid"]
    except:
        await websocket.send_text(json.dumps({"error": "Invalid auth token"}))
        return

    conn = await get_db()
    conn.row_factory = aiosqlite.Row


    # -------------------------------------------------------
    # 1. Build Test Plan
    # -------------------------------------------------------
    await websocket.send_text(json.dumps({"status": "BUILDING_PLAN"}))

    prereq_chain = await utils.get_prereq_chain(conn, testcase_id)
    testplan = {"pretestid_steps": {}, "current_testid": testcase_id, "current_bdd_steps": {}}

    # Load steps
    for tc in prereq_chain[:-1]:
        row = await conn.fetchrow("SELECT steps, args FROM teststep WHERE testcaseid=$1", tc)
        if row and row["steps"]:
            testplan["pretestid_steps"][tc] = dict(zip(row["steps"], row["args"]))

    row = await conn.fetchrow("SELECT steps, args FROM teststep WHERE testcaseid=$1", testcase_id)
    if row and row["steps"]:
        testplan["current_bdd_steps"] = dict(zip(row["steps"], row["args"]))

    # FRONTEND EDIT REQUEST
    await websocket.send_text(json.dumps({
        "status": "REQUEST_EDIT",
        "testplan": testplan
    }))

    msg = await websocket.receive_text()
    try:
        j = json.loads(msg)
        if j.get("type") == "EDITED_TESTPLAN":
            testplan = json.loads(j["testplan_json"])
    except:
        pass

    # -------------------------------------------------------
    # 2. Generate Script
    # -------------------------------------------------------
    await websocket.send_text(json.dumps({"status": "GENERATING"}))

    script = await generate_script(
        testcase_id=testcase_id,
        script_type=script_type,
        script_lang="python",
        testplan=testplan
    )

    # -------------------------------------------------------
    # 3. Execute Script
    # -------------------------------------------------------
    await websocket.send_text(json.dumps({"status": "EXECUTING"}))

    temp_path = tempfile.mktemp(suffix=".py")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(script)

    output = ""
    process = subprocess.Popen(
        [sys.executable, temp_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    for line in process.stdout:
        output += line
        await websocket.send_text(json.dumps({"status": "RUNNING", "log": line.strip()}))

    rc = process.wait()

    if rc == 0:
        final_status = "success"
        message = "Script executed successfully"
    else:
        final_status = "fail"
        message = "Script failed"

    await conn.execute(
        """
        INSERT INTO execution (exeid, testcaseid, scripttype, datestamp, exetime, message, output, status)
        VALUES (nextval('exeid_seq'), $1, $2, CURRENT_DATE, CURRENT_TIME, $3, $4, $5)
        """,
        testcase_id, script_type, message, output, final_status
    )

    await websocket.send_text(json.dumps({
        "status": "COMPLETED",
        "final_status": final_status,
        "message": message
    }))

    await websocket.close()
    os.unlink(temp_path)
    await conn.close()

@router.websocket("/testcases/{testcase_id}/execute-with-madl")
async def execute_testcase_with_madl(
    websocket: WebSocket,
    testcase_id: str,
    script_type: str
):
    """
    Execution endpoint (patched):
     - Build testplan (unchanged)
     - Send PLAN_READY
     - Send REQUEST_EDIT and wait for response (EDITED_TESTPLAN / SKIP_EDIT)
     - Use Azure OpenAI generator (generate_script) to create script
     - Execute -> auto-heal -> log -> store (unchanged)
    """
    await websocket.accept()
    utils.logger.debug(f"[EXEC] WebSocket opened for {testcase_id}, {script_type}")

    # ---------------------------
    # 1. Extract & validate token
    # ---------------------------
    token = None
    try:
        # websocket.scope['headers'] is a list of (b'header', b'value') tuples
        headers = dict(websocket.scope.get("headers", []))
        auth_header = headers.get(b"authorization") or headers.get(b"Authorization")
        if auth_header and isinstance(auth_header, (bytes, bytearray)):
            auth_text = auth_header.decode(errors="ignore")
            if auth_text.startswith("Bearer "):
                token = auth_text.split("Bearer ", 1)[1].strip()
    except Exception:
        token = None

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
        conn = await get_db()
        conn.row_factory = aiosqlite.Row

        userid = current_user["userid"]
        logger.info(LogCategory.INITIALIZATION, f"Execution started for {testcase_id}")

        # -----------------------------
        # 2. Validate access & test case
        # -----------------------------
        cursor = await conn.execute(
            "SELECT projectid FROM testcase WHERE testcaseid = ?",
            (testcase_id,)
        )
        tc_project = await cursor.fetchone()

        if not tc_project:
            await websocket.send_text(json.dumps({"error": "Test case not found", "status": "FAILED"}))
            await websocket.close()
            return

        cursor = await conn.execute(
            "SELECT 1 FROM projectuser WHERE userid = ? AND projectid = ?",
            (userid, tc_project["projectid"])
        )
        access = await cursor.fetchone()

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
            cursor = await conn.execute(
                "SELECT steps, args FROM teststep WHERE testcaseid=?",
                (tc_id,)
            )
            steps_row = await cursor.fetchone()

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
                    # accept either dict or json-string
                    if isinstance(edited_plan, dict):
                        testplan_dict = edited_plan
                        testplan_json = json.dumps(edited_plan)
                    else:
                        try:
                            testplan_dict = json.loads(edited_plan)
                            testplan_json = json.dumps(testplan_dict)
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
        # 5. Generate script (Azure OpenAI)
        # ------------------------------
        await websocket.send_text(json.dumps({"status": "GENERATING", "log": "Generating script..."}))
        try:
            # Prefer generate_script_with_madl if available (keeps compatibility),
            # otherwise fall back to generate_script (Azure generator).
            gen_func = None
            try:
                from app.routers.execution_enhanced import generate_script_with_madl
                gen_func = generate_script_with_madl
            except Exception:
                # fallback to the local azure generator if present
                try:
                    from app.routers.execution_enhanced import generate_script  # or wherever you placed it
                    gen_func = generate_script
                except Exception:
                    gen_func = None

            if gen_func is None:
                raise RuntimeError("No script generation function available. Ensure generate_script or generate_script_with_madl is imported.")

            generated_script = await gen_func(
                testcase_id=testcase_id,
                script_type=script_type,
                script_lang="python",
                testplan=testplan_dict,
                selected_madl_methods=None,
                logger=logger
            )

            if not generated_script or not isinstance(generated_script, str):
                raise RuntimeError("Generated script is empty or invalid")

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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (exeid, testcase_id, script_type, str(datestamp), str(exetime), execution_message, execution_output, final_status)
        )
        await conn.commit()


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
