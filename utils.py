import string
from datetime import datetime, date
from typing import Optional

from jose import JWTError, jwt
from fastapi import HTTPException
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
import tempfile
import subprocess
import sys
import io
import json
import traceback
from selenium import webdriver
from app.config import SECRET_KEY, ALGORITHM
from app.database import get_db_connection

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to generate prefix from role
def get_prefix_from_role(role: str) -> Optional[str]:
    if not role.startswith("role-"):
        return None
    try:
        role_num = int(role.split("-")[1])
        if role_num < 1 or role_num > 26:  # Limit to a-z
            return None
        return string.ascii_lowercase[role_num - 1]
    except (ValueError, IndexError):
        return None

# Function to generate next sequential projectid (PJ0001, PJ0002, etc.)
async def get_next_projectid(conn):
    max_pid = await conn.fetchval('SELECT MAX(projectid) FROM project')
    if max_pid is None:
        return "PJ0001"
    try:
        # Extract the numeric part after 'PJ' and increment
        num = int(max_pid[2:]) + 1  # Assumes format 'PJxxxx' where xxxx is 4-digit number
        return f"PJ{num:04d}"  # Pad to 4 digits for consistency
    except (ValueError, IndexError):
        raise HTTPException(status_code=500, detail="Invalid projectid format in database. Expected 'PJxxxx'.")

# Function to generate next sequential testcaseid (TC0001, TC0002, etc.)
async def get_next_testcaseid(conn):
    max_tid = await conn.fetchval('SELECT MAX(testcaseid) FROM testcase')
    if max_tid is None:
        return "TC0001"
    try:
        # Extract the numeric part after 'TC' and increment
        num = int(max_tid[2:]) + 1  # Assumes format 'TCxxxx' where xxxx is 4-digit number
        return f"TC{num:04d}"  # Pad to 4 digits for consistency
    except (ValueError, IndexError):
        raise HTTPException(status_code=500, detail="Invalid testcaseid format in database. Expected 'TCxxxx'.")

async def get_next_exeid(conn):
    """Generate the next sequential exeid (EX0001, EX0002, etc.) with robust handling."""
    max_eid = await conn.fetchval('SELECT MAX(exeid) FROM execution')
    if max_eid is None:
        return "EX0001"  # Start with EX0001 if table is empty
    try:
        if not max_eid.startswith("EX") or len(max_eid) != 6 or not max_eid[2:].isdigit():
            raise ValueError(f"Invalid exeid format: {max_eid}. Expected 'EX' followed by 4 digits.")
        num = int(max_eid[2:]) + 1  # Extract and increment the numeric part
        if num > 9999:  # Prevent overflow beyond EX9999
            raise ValueError("Exceeded maximum exeid limit (EX9999).")
        return f"EX{num:04d}"  # Pad to 4 digits
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Invalid exeid format in database: {str(e)}. Please correct existing data or clear the execution table.")

# Dummy user validation (replace with your logic)
async def validate_token(token: str):
    try:
        logger.debug(f"Validating token: {token}")  # Log full token for debugging
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        userid = payload.get("userid")  # Changed from "sub" to "userid"
        if not userid:
            logger.error(f"Token payload invalid: {payload}")
            return None
        logger.debug(f"Token validated, userid: {userid}")
        return {"userid": userid}
    except JWTError as e:
        logger.error(f"Token decoding failed: {str(e)} with token: {token}")
        return None

def indent_block(text, prefix, skip_first_line=False):
    """Indent each line in text with the given prefix, optionally skipping the first line and handling 'with' blocks."""
    # If no newlines, add breaks after semicolons and ensure 'with' statements stay intact
    if '\n' not in text:
        # Split by semicolons first, then clean up
        parts = text.split(';')
        text = '\n'.join(part.strip() + ';' for part in parts if part.strip())
        # Ensure 'with' statements are not split
        text = text.replace('with ', 'with\n    ').replace('\n\n', '\n')
    lines = text.split('\n')
    if not lines:
        return text
    result = []
    indent_next = skip_first_line  # Flag to indent lines after the first 'with' or skipped line
    for i, line in enumerate(lines):
        stripped_line = line.lstrip()
        if not stripped_line:
            continue
        if skip_first_line and i == 0:
            result.append(line)  # Keep the first line unindented
            if 'with' in stripped_line:
                indent_next = True  # Set flag to indent the next line after 'with'
        elif indent_next:
            result.append(prefix + stripped_line)  # Indent lines after 'with'
            if not stripped_line.startswith('with'):  # Reset indent_next after the block starts
                indent_next = False
        else:
            result.append(prefix + stripped_line)  # Indent other lines
    return '\n'.join(result)

# ------------------------------------------------------------------
# HELPER: Save one test case + steps
# ------------------------------------------------------------------
async def _save_testcase_and_steps(
    conn,
    tc_id: str,
    data: dict,
    steps: list,
    args: list,
    allowed_projects: set
):
    # --- 1. Resolve Test Case ID ---
    if not tc_id or tc_id in ("", "nan", "None"):
        tc_id = await get_next_testcaseid(conn)
    else:
        exists = await conn.fetchrow("SELECT 1 FROM testcase WHERE testcaseid = $1", tc_id)
        if exists:
            raise HTTPException(status_code=400, detail=f"Test Case ID '{tc_id}' already exists")

    # --- 2. Parse tags & projects ---
    tags = [t.strip() for t in data["tags_raw"].replace(';', ',').split(',') if t.strip()]
    projectids = [p.strip() for p in data["proj_raw"].replace(';', ',').split(',') if p.strip()]

    # --- 3. Validate project access ---
    if not projectids:
        raise HTTPException(status_code=400, detail="Project ID is required")
    if not (set(projectids) <= allowed_projects):
        raise HTTPException(
            status_code=403,
            detail=f"Access denied to project(s): {set(projectids) - allowed_projects}"
        )

    # --- 4. Validate step/arg count ---
    if len(steps) != len(args):
        raise HTTPException(
            status_code=400,
            detail=f"Step count ({len(steps)}) ≠ Argument count ({len(args)}) for {tc_id}"
        )
    if len(steps) == 0:
        raise HTTPException(status_code=400, detail=f"No steps defined for {tc_id}")

    # --- 5. Insert testcase ---
    await conn.execute(
        """
        INSERT INTO testcase (testcaseid, testdesc, pretestid, prereq, tag, projectid)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        tc_id, data["desc"], data["pretestid"], data["prereq"], tags, projectids
    )

    # --- 6. Insert teststep ---
    await conn.execute(
        """
        INSERT INTO teststep (testcaseid, steps, args, stepnum)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (testcaseid) DO UPDATE
        SET steps = EXCLUDED.steps, args = EXCLUDED.args, stepnum = EXCLUDED.stepnum
        """,
        tc_id, steps, args, len(steps)
    )

# Helper function to get prerequisite chain
async def get_prereq_chain(conn, testcase_id: str, visited: set = None):
    """Recursively get all prerequisite test cases in order."""
    if visited is None:
        visited = set()
    if testcase_id in visited:
        return []

    visited.add(testcase_id)
    chain = []

    # Fetch pretestid
    pretest_row = await conn.fetchrow(
        "SELECT pretestid FROM testcase WHERE testcaseid = $1",
        testcase_id
    )
    pretestid = pretest_row["pretestid"] if pretest_row else None

    if pretestid:
        # Recurse to get prereq chain
        prereq_chain = await get_prereq_chain(conn, pretestid, visited)
        chain.extend(prereq_chain)

    # Add current test case
    chain.append(testcase_id)

    return chain