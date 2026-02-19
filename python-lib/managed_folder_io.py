"""Managed folder helpers that work on local and non-local backends."""

import io
import json
from pathlib import PurePosixPath


def _normalize_path(path):
    if path is None:
        return ""
    normalized = str(path).strip()
    return normalized.lstrip("/")


def list_paths(folder):
    try:
        paths = folder.list_paths_in_partition()
    except TypeError:
        # Some runtime contexts require an explicit partition argument.
        paths = folder.list_paths_in_partition("")
    return paths or []


def path_exists(folder, path):
    target = _normalize_path(path)
    for existing_path in list_paths(folder):
        if _normalize_path(existing_path) == target:
            return True
    return False


def first_existing_path(folder, path_candidates):
    existing_paths = list_paths(folder)
    normalized_to_original = {_normalize_path(path): path for path in existing_paths}
    for candidate in path_candidates:
        matched_path = normalized_to_original.get(_normalize_path(candidate))
        if matched_path is not None:
            return matched_path
    return None


def basename(path):
    return PurePosixPath(_normalize_path(path)).name


def read_bytes(folder, path):
    normalized_path = _normalize_path(path)
    with folder.get_download_stream(normalized_path) as stream:
        return stream.read()


def write_bytes(folder, path, data):
    normalized_path = _normalize_path(path)
    with io.BytesIO(data) as stream:
        folder.upload_stream(normalized_path, stream)


def read_text(folder, path, encoding="utf-8"):
    return read_bytes(folder, path).decode(encoding)


def write_text(folder, path, text, encoding="utf-8"):
    write_bytes(folder, path, text.encode(encoding))


def read_json(folder, path):
    normalized_path = _normalize_path(path)
    try:
        parsed = folder.read_json(normalized_path)
        if isinstance(parsed, (dict, list)):
            return parsed
        if isinstance(parsed, str):
            return json.loads(parsed)
        return parsed
    except Exception:
        return json.loads(read_text(folder, normalized_path))


def write_json(folder, path, obj):
    normalized_path = _normalize_path(path)
    try:
        folder.write_json(normalized_path, obj)
    except Exception:
        write_text(folder, normalized_path, json.dumps(obj))
