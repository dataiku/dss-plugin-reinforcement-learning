from dataiku.customwebapp import *

import dataiku
import datetime

import simplejson as json
from managed_folder_io import basename, list_paths, path_exists, read_json


def _folder_from_id(folder_id):
    if not folder_id:
        return None
    try:
        return dataiku.Folder(folder_id)
    except Exception:
        return None


def _read_manifest_profiles(folder, manifest_path):
    manifest = read_json(folder, manifest_path)
    runs = manifest.get("runs", [])
    if not isinstance(runs, list):
        runs = []

    profiles = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        profiles.append(
            {
                "profile_slug": run.get("profile_slug"),
                "profile_name": run.get("profile_name"),
                "agent": run.get("agent"),
                "environment": run.get("environment"),
                "model_name": run.get("model_name"),
                "training_infos_file": run.get("training_infos_file"),
            }
        )
    return {
        "generated_at": manifest.get("generated_at"),
        "total_profiles": manifest.get("total_profiles", len(profiles)),
        "profiles": profiles,
    }


def _find_manifest(config):
    folder_candidates = []
    models_folder_id = config.get("training_models_folder")
    replay_folder_id = config.get("replay_folder")
    if models_folder_id:
        folder_candidates.append(("training_models_folder", models_folder_id))
    if replay_folder_id:
        folder_candidates.append(("replay_folder", replay_folder_id))

    inspected_folders = []
    for source_name, folder_id in folder_candidates:
        folder = _folder_from_id(folder_id)
        if not folder:
            inspected_folders.append({"source": source_name, "status": "unreadable"})
            continue
        inspected_folders.append({"source": source_name, "folder_id": folder_id, "status": "inspected"})
        manifest_path = "training_runs_manifest.json"
        if path_exists(folder, manifest_path):
            manifest_data = _read_manifest_profiles(folder, manifest_path)
            return {
                "source": source_name,
                "folder_id": folder_id,
                "manifest_path": manifest_path,
                "manifest_data": manifest_data,
                "inspected_folders": inspected_folders,
            }
    return {
        "source": None,
        "folder_id": None,
        "manifest_path": None,
        "manifest_data": None,
        "inspected_folders": inspected_folders,
    }


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_scores(data):
    raw_scores = data.get("score", [])
    if not isinstance(raw_scores, list):
        return []
    numeric_scores = []
    for value in raw_scores:
        numeric_value = _to_float(value)
        if numeric_value is not None:
            numeric_scores.append(numeric_value)
    return numeric_scores


def _build_summary(data):
    scores = _extract_scores(data)
    episodes = len(scores)
    average_score = _to_float(data.get("average_score"))
    if average_score is None and episodes > 0:
        average_score = sum(scores) / float(episodes)

    best_score = max(scores) if episodes > 0 else None
    worst_score = min(scores) if episodes > 0 else None
    last_score = scores[-1] if episodes > 0 else None

    std_score = None
    if episodes > 1 and average_score is not None:
        variance = sum((score - average_score) ** 2 for score in scores) / float(episodes)
        std_score = variance ** 0.5

    return {
        "episodes": episodes,
        "average_score": average_score,
        "best_score": best_score,
        "worst_score": worst_score,
        "last_score": last_score,
        "std_score": std_score,
    }


def _parse_time(value):
    def _normalize_epoch_seconds(epoch_value):
        # Some backends return ms/us/ns; convert to seconds.
        if epoch_value > 1e17:  # nanoseconds
            return epoch_value / 1e9
        if epoch_value > 1e14:  # microseconds
            return epoch_value / 1e6
        if epoch_value > 1e11:  # milliseconds
            return epoch_value / 1e3
        return epoch_value

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _normalize_epoch_seconds(float(value))
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return None
        try:
            return _normalize_epoch_seconds(float(stripped_value))
        except Exception:
            try:
                return datetime.datetime.fromisoformat(stripped_value).timestamp()
            except Exception:
                return None
    return None


def _try_get_modified_ts(folder, path):
    try:
        details = folder.get_path_details(path)
    except Exception:
        return None

    if not isinstance(details, dict):
        return None

    for key in ["lastModified", "last_modified", "mtime", "modificationTime", "createdOn"]:
        candidate = _parse_time(details.get(key))
        if candidate is not None:
            return candidate
    return None


def _load_run(folder, json_path):
    data = read_json(folder, json_path)
    if not isinstance(data, dict):
        data = {"raw_payload": data}

    modified_ts = _try_get_modified_ts(folder, json_path)
    if modified_ts is not None:
        try:
            modified_at = datetime.datetime.fromtimestamp(modified_ts).isoformat()
        except Exception:
            modified_at = None
    else:
        modified_at = None
    filename = basename(json_path)

    return {
        "id": filename,
        "filename": filename,
        "modified_ts": modified_ts,
        "modified_at": modified_at,
        "summary": _build_summary(data),
        "data": data,
    }


@app.route('/first_api_call')
def first_call():
    input_folder = get_webapp_config()["replay_folder"]
    myfolder = dataiku.Folder(input_folder)
    folder_paths = list_paths(myfolder)

    json_paths = [path for path in folder_paths if str(path).lower().endswith(".json")]
    if len(json_paths) == 0:
        return json.dumps({"error": "No JSON results found in the selected folder"})

    runs = []
    errors = []
    for json_path in json_paths:
        try:
            runs.append(_load_run(myfolder, json_path))
        except Exception as err:
            errors.append({"file": basename(json_path), "error": str(err)})

    if len(runs) == 0:
        return json.dumps(
            {
                "error": "No valid JSON results could be parsed",
                "errors": errors,
            }
        )

    def sort_key(run):
        training_ts = _parse_time(run["data"].get("trainingdate"))
        if training_ts is not None:
            return training_ts
        if run["modified_ts"] is not None:
            return run["modified_ts"]
        return 0.0

    runs = sorted(runs, key=sort_key, reverse=True)
    video_files = sorted([path for path in folder_paths if str(path).lower().endswith(".mp4")])
    payload = {
        "runs": runs,
        "latest_run_id": runs[0]["id"],
        "video_count": len(video_files),
        "video_files": [basename(video_path) for video_path in video_files],
        "errors": errors,
    }
    return json.dumps(payload, ignore_nan=True)


@app.route('/manifest_profiles')
def manifest_profiles():
    config = get_webapp_config()
    manifest_lookup = _find_manifest(config)
    if not manifest_lookup["manifest_data"]:
        return json.dumps(
            {
                "error": (
                    "training_runs_manifest.json not found. "
                    "Set Training Models Folder to the Train output folder (multi-profile run)."
                ),
                "inspected_folders": manifest_lookup["inspected_folders"],
            }
        )

    payload = {
        "source": manifest_lookup["source"],
        "folder_id": manifest_lookup["folder_id"],
        "manifest_path": manifest_lookup["manifest_path"],
        "manifest": manifest_lookup["manifest_data"],
    }
    return json.dumps(payload)
