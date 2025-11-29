first_attempt = False
execution_message = ""

await websocket.send_text(json.dumps({
    "status": "EXECUTING",
    "log": "Starting script execution..."
}))

import subprocess
import sys
import asyncio
import os
from datetime import datetime

execution_logs = []

try:
    # 1. Store script inside project directory
    project_dir = os.getcwd()
    scripts_dir = os.path.join(project_dir, "generated_scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    script_filename = f"script_{testcase_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    script_path = os.path.join(scripts_dir, script_filename)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(generated_script)

    logger.info(LogCategory.EXECUTION, f"Executing script: {script_path}")

    # 2. Execute script
    process = subprocess.Popen(
        [sys.executable, script_path],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )

    execution_output = ""

    # 3. Stream logs live
    for line in process.stdout:
        line = line.rstrip("\n")
        if not line.strip():
            continue

        execution_logs.append(line)
        execution_output += line + "\n"

        await websocket.send_text(json.dumps({
            "status": "RUNNING",
            "log": line
        }))
        await asyncio.sleep(0.02)

    # 4. Completion
    return_code = process.wait()

    if return_code == 0:
        first_attempt = True
        execution_message = f"Script executed successfully. Stored at: {script_path}"

        logger.success(LogCategory.EXECUTION, execution_message)
        await websocket.send_text(json.dumps({
            "status": "SUCCESS",
            "log": execution_message
        }))

    else:
        execution_message = f"Script execution failed. Stored at: {script_path}"

        logger.error(LogCategory.EXECUTION, execution_message)
        await websocket.send_text(json.dumps({
            "status": "FAILED",
            "log": execution_message
        }))

except Exception as e:
    execution_message = f"Execution error: {str(e)}"

    logger.error(LogCategory.EXECUTION, execution_message)
    await websocket.send_text(json.dumps({
        "status": "FAILED",
        "log": execution_message
    }))

# --------------------------- FINAL STATUS ----------------------------
if first_attempt:
    final_status = "success"
else:
    final_status = "fail"

# ---------------------- STORE EXECUTION (SQLite) ---------------------
exeid = await utils.get_next_exeid(conn)

datestamp = datetime.now().date().isoformat()
exetime = datetime.now().strftime("%H:%M:%S")

await conn.execute(
    """
    INSERT INTO execution (exeid, testcaseid, scripttype, datestamp, exetime, message, output, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
    exeid, testcase_id, script_type, datestamp, exetime,
    execution_message, execution_output, final_status
)
await conn.commit()

# Final response
await websocket.send_text(json.dumps({
    "status": "COMPLETED",
    "log": execution_message,
    "final_status": final_status,
    "summary": logger.get_summary()
}))
