# madl_pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import List

from madl_settings import MADL_ENABLE_FILE_WRITING  # or replace with simple env flag if you prefer

from madl_reader import load_source_files
from madl_parser import extract_python_methods, extract_java_methods
from madl_llm import generate_madl_for_method
from madl_commenter import inject_madl_comment_into_method
from madl_vectordb import push_madl_to_opensearch


def process_file(path: Path):
    print(f"\n=== Processing file: {path} ===")

    if path.suffix.lower() == ".py":
        methods = extract_python_methods(path)
    elif path.suffix.lower() == ".java":
        methods = extract_java_methods(path)
    else:
        return

    if not methods:
        print("  No methods found in file.")
        return

    new_file_parts: List[str] = []

    for m in methods:
        name = m["name"]
        params = m["parameters"]
        code = m["code"]
        language = m["language"]

        print(f"  → Analyzing method: {name}{params}")

        madl = generate_madl_for_method(
            method_code=code,
            class_name=path.stem,
            parameters=params,
            language=language,
        )

        if not madl:
            print("    - Not reusable / LLM skipped.")
            new_file_parts.append(code)
            new_file_parts.append("\n\n")
            continue

        print(f"    ✔ Reusable method detected: {madl['method_name']}")

        # Inject MADL comment into the method source
        updated_code = inject_madl_comment_into_method(code, madl, language)
        new_file_parts.append(updated_code)
        new_file_parts.append("\n\n")

        # Push to OpenSearch
        extra_meta = {
            "file_path": str(path),
            "language": language,
        }
        push_madl_to_opensearch(madl, extra_meta=extra_meta)

    if MADL_ENABLE_FILE_WRITING:
        new_text = "".join(new_file_parts).rstrip() + "\n"
        path.write_text(new_text, encoding="utf-8")
        print("  → File updated with MADL comments.")
    else:
        print("  → MADL_ENABLE_FILE_WRITING=false, file not modified.")


def run_pipeline(root_folder: str):
    root = Path(root_folder).resolve()
    print(f"Scanning root folder: {root}")

    files = load_source_files(str(root))
    print(f"Found {len(files)} candidate source files.")

    for f in files:
        process_file(f)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python madl_pipeline.py <root_folder>")
        raise SystemExit(1)

    run_pipeline(sys.argv[1])
