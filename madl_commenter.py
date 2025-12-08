# madl_commenter.py
from __future__ import annotations
import json
from typing import Dict


def _build_comment_block(madl: Dict, language: str) -> str:
    # Don't include "reusable" inside comment
    clean = {k: v for k, v in madl.items() if k != "reusable"}

    text = json.dumps(clean, indent=2)

    # Python comment style
    if language == "python":
        header = "# --- MADL ---\n"
        commented = "\n".join("# " + line for line in text.splitlines())
        return header + commented + "\n"

    # Java block comment style
    if language == "java":
        return f"/* MADL\n{text}\n*/\n"

    return text


def inject_madl_comment(method_code: str, madl: Dict, language: str) -> str:
    """
    Prepend MADL metadata comment before method code.
    """
    block = _build_comment_block(madl, language)
    return block + method_code
