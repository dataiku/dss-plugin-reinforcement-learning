import numpy as np
import time

import dataiku
from dataiku.customrecipe import *
import pickle

from managed_folder_io import first_existing_path, read_bytes, read_json, write_json
from rl_compat import (
    make_env,
    parse_environment_kwargs,
    reset_env,
    resolve_environment_config,
    safe_artifact_name,
    step_env,
)


def _extract_environment_name(training_infos):
    environment_name = training_infos.get("environmentName")
    if environment_name:
        return environment_name
    raw_environment_value = training_infos.get("environment")
    if isinstance(raw_environment_value, str) and raw_environment_value.strip():
        return raw_environment_value.strip().split(" ")[0]
    return None


def _load_training_infos(saved_models_folder, profile):
    if profile and isinstance(profile.get("training_infos"), dict):
        return profile["training_infos"]
    training_infos_file = "training_infos.json"
    if profile and profile.get("training_infos_file"):
        training_infos_file = profile["training_infos_file"]
    return read_json(saved_models_folder, training_infos_file)


def _manual_runtime_settings():
    recipe_config = get_recipe_config()
    environment_config = resolve_environment_config(recipe_config)
    return {
        "agent_var": recipe_config["agent"],
        "environment_var": environment_config["environment_id"],
        "raw_environment_var": environment_config["raw_environment_id"],
        "environment_kwargs": environment_config["environment_kwargs"],
        "custom_environment_module": environment_config["custom_environment_module"],
        "model_name_override": None,
        "output_suffix": None,
    }


def _profile_runtime_settings(profile, training_infos):
    environment_var = _extract_environment_name(training_infos) or profile.get("environment")
    if not environment_var:
        raise ValueError("Cannot determine environment for profile testing")

    raw_environment_var = training_infos.get("environmentRawName", environment_var)
    environment_kwargs = parse_environment_kwargs(training_infos.get("environment_kwargs", {}))
    custom_environment_module = training_infos.get("custom_environment_module", "")
    agent_var = profile.get("agent") or training_infos.get("agent") or "q"
    model_name_override = profile.get("model_name")
    if model_name_override and model_name_override.endswith(".pickle"):
        model_name_override = model_name_override[:-7]

    return {
        "agent_var": agent_var,
        "environment_var": environment_var,
        "raw_environment_var": raw_environment_var,
        "environment_kwargs": environment_kwargs,
        "custom_environment_module": custom_environment_module,
        "model_name_override": model_name_override,
        "output_suffix": profile.get("profile_slug"),
    }


def test_q_agent(agent, profile=None):
    del agent

    saved_models = dataiku.Folder(get_input_names_for_role("main_input")[0])

    saved_replays = dataiku.Folder(get_output_names_for_role("main_output")[0])

    if profile:
        training_infos = _load_training_infos(saved_models, profile)
        runtime_settings = _profile_runtime_settings(profile, training_infos)
    else:
        runtime_settings = _manual_runtime_settings()
        training_infos = _load_training_infos(saved_models, None)

    agent_var = runtime_settings["agent_var"]
    environment_var = runtime_settings["environment_var"]
    raw_environment_var = runtime_settings["raw_environment_var"]
    model_name_override = runtime_settings["model_name_override"]
    model_candidates = []
    if model_name_override:
        normalized_override = model_name_override.rstrip("/")
        model_candidates.append(normalized_override)
        if not normalized_override.endswith(".pickle"):
            model_candidates.append(normalized_override + ".pickle")
    else:
        normalized_model_name = agent_var + "_" + safe_artifact_name(environment_var) + ".pickle"
        model_candidates.append(normalized_model_name)
        if raw_environment_var != environment_var:
            model_candidates.append(agent_var + "_" + safe_artifact_name(raw_environment_var) + ".pickle")

    model_path = first_existing_path(saved_models, model_candidates)
    if not model_path:
        raise ValueError("Could not find Q model object for candidates: {}".format(model_candidates))

    qtable = pickle.loads(read_bytes(saved_models, model_path))

    env = make_env(
        environment_var,
        environment_kwargs=runtime_settings["environment_kwargs"],
        custom_environment_module=runtime_settings["custom_environment_module"],
    )
    scores = []

    for episode in range(100):
        state = int(reset_env(env))
        episode_reward = 0.0
        done = False

        for _ in range(99):
            action = int(np.argmax(qtable[state, :]))
            new_state, reward, done, _ = step_env(env, action)
            episode_reward += float(reward)

            if done:
                break
            state = int(new_state)

        scores.append(episode_reward)

    env.close()

    mean_score = sum(scores) / float(len(scores))
    average_score = int(mean_score * 100)
    percentage_score = mean_score * 100

    testing_infos = {
        "agent_name": "Q-Learning Agent",
        "score": scores,
        "average_score": average_score,
        "percentage_score": percentage_score,
    }

    infos = dict(training_infos, **testing_infos)

    output_suffix = runtime_settings["output_suffix"]
    output_json_name = (
        safe_artifact_name(training_infos["environment"])
        + "_"
        + safe_artifact_name(training_infos["agent"])
        + ("_" + safe_artifact_name(output_suffix) if output_suffix else "")
        + "_"
        + str(time.time())
        + ".json"
    )

    write_json(saved_replays, output_json_name, infos)
