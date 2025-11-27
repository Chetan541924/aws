# ==================== utils.py – FINAL VERSION (Single-File Ready) ====================

import string
from datetime import datetime
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException
import logging
import json
import asyncio

# Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(_name_)

# Use these from your main file (defined there)
# SECRET_KEY, ALGORITHM, get_db_connection must exist in main.py

# ==================== ROLE PREFIX ====================
def get_prefix_from_role(role: str) -> Optional[str]:
    if not role.startswith("role-"):
        return None
    try:
        role_num = int(role.split("-")[1])
        if 1 <= role_num <= 26:
            return string.ascii_lowercase[role_num - 1]
    except:
        pass
    return None

# ==================== ID GENERATORS ====================
async def get_next_projectid(conn) -> str:
    row = await (await conn.execute("SELECT projectid FROM project ORDER BY projectid DESC LIMIT 1")).fetchone()
    if not row:
        return "PJ0001"
    try:
        num = int(row["projectid"][2:]) + 1
        return f"PJ{num:04d}"
    except:
        raise HTTPException(status_code=500, detail="Corrupted projectid in database")

async def get_next_testcaseid(conn) -> str:
    row = await (await conn.execute("SELECT testcaseid FROM testcase ORDER BY testcaseid DESC LIMIT 1")).fetchone()
    if not row:
        return "TC0001"
    try:
        num = int(row["testcaseid"][2:]) + 1
        return f"TC{num:04d}"
    except:
        raise HTTPException(status_code=500, detail="Corrupted testcaseid in database")

async def get_next_exeid(conn) -> str:
    row = await (await conn.execute("SELECT exeid FROM execution ORDER BY exeid DESC LIMIT 1")).fetchone()
    if not row:
        return "EX0001"
    try:
        num = int(row["exeid"][2:]) + 1
        if num > 9999:
            raise ValueError("EX9999 limit reached")
        return f"EX{num:04d}"
    except:
        raise HTTPException(status_code=500, detail="Corrupted exeid in database")

# ==================== TOKEN VALIDATION ====================
async def validate_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        userid = payload.get("userid")
        if not userid:
            logger.error(f"Invalid token payload: {payload}")
            return None
        return {"userid": userid}
    except JWTError as e:
        logger.error(f"Token validation failed: {e}")
        return None

# ==================== INDENT HELPER ====================
def indent_block(text: str, prefix: str = "    ", skip_first_line: bool = False) -> str:
    if not text.strip():
        return text
    lines = text.splitlines()
    result = []
    do_indent = skip_first_line
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped:
            result.append("")
            continue
        if skip_first_line and i == 0:
            result.append(line)
            if "with " in stripped:
                do_indent = True
        elif do_indent:
            result.append(prefix + stripped)
            if not stripped.startswith("with"):
                do_indent = False
        else:
            result.append(prefix + stripped)
    return "\n".join(result)

# ==================== PREREQ CHAIN (RECURSIVE ====================
async def get_prereq_chain(conn, testcase_id: str, visited: set = None) -> list:
    """Return list of testcase IDs: [oldest_prereq, ..., current]"""
    if visited is None:
        visited = set()
    if testcase_id in visited:
        return []  # cycle protection
    visited.add(testcase_id)

    row = await (await conn.execute(
        "SELECT pretestid FROM testcase WHERE testcaseid = ?",
        (testcase_id,)
    )).fetchone()

    if not row or not row["pretestid"]:
        return [testcase_id]

    chain = await get_prereq_chain(conn, row["pretestid"], visited)
    chain.append(testcase_id)
    return chain

# ==================== SAVE TESTCASE + STEPS (REUSABLE) ====================
async def save_testcase_with_steps(
    conn,
    tc_id: str,
    desc: str,
    pretestid: Optional[str],
    prereq: str,
    tags: list,
    project_ids: list,
    steps: list,
    args: list,
    created_by: str
):
    """Insert or update testcase + steps (used by upload & commit"""
    await conn.execute("""
        INSERT INTO testcase 
        (testcaseid, testdesc, pretestid, prereq, tag, projectid, no_steps, created_on, created_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?)
        ON CONFLICT(testcaseid) DO UPDATE SET
            testdesc=excluded.testdesc,
            pretestid=excluded.pretestid,
            prereq=excluded.prereq,
            tag=excluded.tag,
            projectid=excluded.projectid,
            no_steps=excluded.no_steps
    """, (tc_id, desc, pretestid, prereq, to_json(tags), to_json(project_ids), len(steps), created_by))

    await conn.execute("DELETE FROM teststep WHERE testcaseid = ?", (tc_id,))
    await conn.execute(
        "INSERT INTO teststep (testcaseid, steps, args, stepnum) VALUES (?, ?, ?, ?)",
        (tc_id, to_json(steps), to_json(args), len(steps))
    )

# ==================== JSON HELPERS (if not in main) ====================
def to_json(val): return json.dumps(val or [])
def from_json(val):
    if not val or val == "[]": return []
    try: return json.loads(val)
    except: return []