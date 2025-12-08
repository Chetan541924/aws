# madl_parser.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import ast
import re


def extract_python_methods(file_path: Path) -> List[Dict]:
    """Return a list of {name, parameters_str, code}."""
    text = file_path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    methods: List[Dict] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # best-effort source segment extraction
            try:
                method_code = ast.get_source_segment(text, node) or ""
            except Exception:
                method_code = ""
            params = [arg.arg for arg in node.args.args]
            params_str = "(" + ", ".join(params) + ")"
            methods.append(
                {
                    "name": node.name,
                    "parameters": params_str,
                    "code": method_code,
                    "language": "python",
                }
            )

    return methods


def _extract_java_block(content: str, start_idx: int) -> str:
    """Given index of first '{', return full method block by counting braces."""
    depth = 0
    in_block = False
    for i in range(start_idx, len(content)):
        ch = content[i]
        if ch == "{":
            depth += 1
            in_block = True
        elif ch == "}":
            depth -= 1
            if depth == 0 and in_block:
                return content[start_idx : i + 1]
    return content[start_idx:]


def extract_java_methods(file_path: Path) -> List[Dict]:
    """
    Very lightweight Java method extractor (public/protected/private ...(...) { ... }).
    For production use, replace with proper Java parser.
    """
    text = file_path.read_text(encoding="utf-8")
    pattern = r"(public|protected|private)\s+[^\(]+\s+([A-Za-z0-9_]+)\s*\(([^\)]*)\)\s*\{"
    methods: List[Dict] = []

    for match in re.finditer(pattern, text):
        method_name = match.group(2)
        params_str = "(" + match.group(3).strip() + ")"
        body = _extract_java_block(text, match.start(0))
        methods.append(
            {
                "name": method_name,
                "parameters": params_str,
                "code": body,
                "language": "java",
            }
        )

    return methods
