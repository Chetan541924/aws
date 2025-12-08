# madl_pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import List

from madl_settings import MADL_ENABLE_FILE_WRITING
from madl_reader import load_source_files
from madl_parser import extract_python_methods, extract_java_methods
from madl_llm import generate_madl_for_method
from madl_commenter import inject_madl_comment
from madl_vectordb import push_madl_to_qdrant


def process_file(path: Path):
    print(f"\n=== Processing {path} ===")

    # Extract methods based on file type
    if path.suffix.lower() == ".py":
        methods = extract_python_methods(path)
    else:
        methods = extract_java_methods(path)

    if not methods:
        print("  No methods found.")
        return

    new_file_content_parts: List[str] = []

    for m in methods:
        name = m["name"]
        parameters = m["parameters"]
        code = m["code"]
        language = m["language"]

        print(f"  → Checking method: {name}{parameters}")

        madl = generate_madl_for_method(
            method_code=code,
            class_name=path.stem,
            parameters=parameters,
            language=language,
        )

        if not madl:
            print("    - Not reusable / LLM rejected.")
            new_file_content_parts.append(code)
            new_file_content_parts.append("\n\n")
            continue

        print(f"    ✔ Reusable: {madl['method_name']}")

        # Inject MADL as comments
        updated_method = inject_madl_comment(code, madl, language)
        new_file_content_parts.append(updated_method)
        new_file_content_parts.append("\n\n")

        # Push to Qdrant DB
        push_madl_to_qdrant(
            madl,
            extra={"file_path": str(path), "language": language},
        )
        print("    ✔ Uploaded to Qdrant")

    if MADL_ENABLE_FILE_WRITING:
        print("  → Updating source file with MADL comments.")
        final_text = "".join(new_file_content_parts)
        path.write_text(final_text, encoding="utf-8")
    else:
        print("  → MADL_ENABLE_FILE_WRITING=false, file not modified.")


def run_pipeline(root_folder: str):
    root = Path(root_folder).resolve()
    print(f"Scanning root folder: {root}")

    files = load_source_files(str(root))
    print(f"Found {len(files)} source files.")

    for f in files:
        process_file(f)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python madl_pipeline.py <folder>")
        exit(1)

    run_pipeline(sys.argv[1])
