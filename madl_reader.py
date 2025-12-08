# madl_reader.py
from pathlib import Path
from typing import List
from madl_settings import MADL_ALLOWED_BASE_PATHS, MADL_MAX_FILE_SIZE_MB

def is_under_allowed_base(path: Path) -> bool:
    if not MADL_ALLOWED_BASE_PATHS:
        return True  # if nothing configured, allow everything
    path = path.resolve()
    for base in MADL_ALLOWED_BASE_PATHS:
        try:
            base_path = Path(base).resolve()
            if base_path in path.parents or base_path == path:
                return True
        except Exception:
            continue
    return False


def load_source_files(root_folder: str) -> List[Path]:
    root = Path(root_folder).resolve()
    if not is_under_allowed_base(root):
        raise PermissionError(f"{root} is not under MADL_ALLOWED_BASE_PATHS")

    max_bytes = MADL_MAX_FILE_SIZE_MB * 1024 * 1024
    files: List[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".py", ".java"}:
            continue
        if path.stat().st_size > max_bytes:
            continue
        files.append(path)

    return files
