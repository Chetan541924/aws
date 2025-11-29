execute------

# ----------------------- EXECUTE SCRIPT -----------------------
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
    project_dir = os.getcwd()  # backend working directory
    scripts_dir = os.path.join(project_dir, "generated_scripts")

    # Create folder if not exists
    os.makedirs(scripts_dir, exist_ok=True)

    # Create unique script name
    script_filename = f"script_{testcase_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    script_path = os.path.join(scripts_dir, script_filename)

    # Save script file
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(generated_script)

    logger.info(LogCategory.EXECUTION, f"Executing script: {script_path}")

    # 2. Execute script from project directory
    process = subprocess.Popen(
        [sys.executable, script_path],
        cwd=project_dir,      # Force execution inside your project folder
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1
    )

    execution_output = ""

    # 3. Stream execution logs live
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

    # 4. Wait for completion
    return_code = process.wait()

    if return_code == 0:
        logger.success(LogCategory.EXECUTION, "Script executed successfully")
        await websocket.send_text(json.dumps({
            "status": "SUCCESS",
            "log": f"Script executed successfully.\nStored at: {script_path}"
        }))
    else:
        logger.error(LogCategory.EXECUTION, "Script execution failed")
        await websocket.send_text(json.dumps({
            "status": "FAILED",
            "log": f"Script execution failed.\nStored at: {script_path}"
        }))

except Exception as e:
    logger.error(LogCategory.EXECUTION, f"Execution error: {str(e)}")

    await websocket.send_text(json.dumps({
        "status": "FAILED",
        "log": f"Execution error: {str(e)}"
    }))
