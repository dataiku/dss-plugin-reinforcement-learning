import dataiku

from dataiku.customrecipe import *

from stable_baselines_testing_wrapper import *
from managed_folder_io import path_exists, read_json

def _load_manifest(saved_models_folder):
    manifest_path = "training_runs_manifest.json"
    if not path_exists(saved_models_folder, manifest_path):
        raise ValueError(
            "training_runs_manifest.json not found in input folder. Run Train in multi-profile mode first, "
            "or switch test selection mode to Manual."
        )
    manifest = read_json(saved_models_folder, manifest_path)
    runs = manifest.get("runs", [])
    if not isinstance(runs, list) or len(runs) == 0:
        raise ValueError("training_runs_manifest.json does not contain any runs")
    return runs


def _parse_slugs(raw_text):
    if raw_text is None:
        return []
    tokens = str(raw_text).replace("\n", ",").split(",")
    slugs = [token.strip() for token in tokens if token.strip()]
    return slugs


def _select_manifest_runs(manifest_runs, recipe_config):
    selection_mode = recipe_config.get("model_selection_mode", "manual")
    if selection_mode == "manifest_batch_all":
        return manifest_runs

    if selection_mode == "manifest_single":
        selected_slug = recipe_config.get("selected_profile_slug", "").strip()
        if not selected_slug:
            raise ValueError("selected_profile_slug must be provided for single-profile manifest testing")
        selected = [run for run in manifest_runs if run.get("profile_slug") == selected_slug]
        if len(selected) == 0:
            raise ValueError("Profile slug '{}' was not found in training manifest".format(selected_slug))
        return selected

    if selection_mode == "manifest_batch_selected":
        selected_slugs = _parse_slugs(recipe_config.get("selected_profile_slugs_csv", ""))
        if len(selected_slugs) == 0:
            raise ValueError("Provide at least one profile slug for manifest batch-selected mode")
        selected_set = set(selected_slugs)
        selected = [run for run in manifest_runs if run.get("profile_slug") in selected_set]
        if len(selected) == 0:
            raise ValueError("None of the provided slugs matched training manifest entries")
        return selected

    raise ValueError("Unsupported manifest selection mode '{}'".format(selection_mode))


def _normalize_model_name_for_profile(run):
    model_name = run.get("model_name")
    if not model_name:
        return None
    if model_name.endswith(".pickle"):
        return model_name[:-7]
    if model_name.endswith(".zip"):
        return model_name[:-4]
    return model_name


def _build_profile(saved_models_folder, run):
    training_infos_file = run.get("training_infos_file", "training_infos.json")
    if not path_exists(saved_models_folder, training_infos_file):
        raise ValueError("Training infos file '{}' not found for profile".format(training_infos_file))

    training_infos = read_json(saved_models_folder, training_infos_file)

    return {
        "agent": run.get("agent") or training_infos.get("agent"),
        "profile_slug": run.get("profile_slug"),
        "model_name": _normalize_model_name_for_profile(run),
        "training_infos_file": training_infos_file,
        "training_infos": training_infos,
    }


recipe_config = get_recipe_config()
selection_mode = recipe_config.get("model_selection_mode", "manual")

if selection_mode == "manual":
    agent_var = recipe_config["agent"]
    if agent_var == "q":
        from q_testing_wrapper import test_q_agent

        test_q_agent(agent_var)
    else:
        test_agent(agent_var)
else:
    saved_models = dataiku.Folder(get_input_names_for_role("main_input")[0])

    manifest_runs = _load_manifest(saved_models)
    runs_to_test = _select_manifest_runs(manifest_runs, recipe_config)

    for run in runs_to_test:
        profile = _build_profile(saved_models, run)
        profile_agent = (profile.get("agent") or "").lower()
        if profile_agent == "q":
            from q_testing_wrapper import test_q_agent

            test_q_agent("q", profile=profile)
        elif profile_agent == "dqn":
            test_agent("dqn", profile=profile)
        else:
            raise ValueError("Unsupported profile agent '{}' in training manifest".format(profile.get("agent")))



