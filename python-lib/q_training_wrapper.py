import dataiku

import datetime

import pickle

from dataiku.customrecipe import *

from q import train_q
from rl_compat import build_training_profile_configs, resolve_environment_config, safe_artifact_name
from managed_folder_io import write_bytes, write_json


SUPPORTED_Q_ENVIRONMENTS = {"FrozenLake-v1", "Taxi-v3"}


def _unique_profile_slug(profile_config, profile_position, used_slugs):
    raw_name = profile_config.get("_profile_name") or ("profile_{}".format(profile_position + 1))
    base_slug = safe_artifact_name(raw_name)
    slug = base_slug
    suffix = 2
    while slug in used_slugs:
        slug = "{}_{}".format(base_slug, suffix)
        suffix += 1
    used_slugs.add(slug)
    return slug


def train_q_agent(agent):
    recipe_config = get_recipe_config()
    profile_configs = build_training_profile_configs(recipe_config)
    is_multi_profile_mode = recipe_config.get("training_profiles_mode", "single") == "multi"

    saved_models = dataiku.Folder(get_output_names_for_role("main_output")[0])

    training_manifest = []
    used_profile_slugs = set()
    for profile_position, profile_config in enumerate(profile_configs):
        agent_var = profile_config["agent"]
        if agent_var != agent:
            raise ValueError("Profile agent '{}' does not match selected agent '{}'".format(agent_var, agent))

        environment_config = resolve_environment_config(profile_config)
        environment_var = environment_config["environment_id"]
        raw_environment_var = environment_config["raw_environment_id"]
        environment_kwargs = environment_config["environment_kwargs"]
        custom_environment_module = environment_config["custom_environment_module"]
        environment_mode = environment_config["environment_mode"]
        environment_library_var = profile_config["environment_library"]

        gamma_var = float(profile_config["gamma"])
        lr_var = float(profile_config["learning_rate"])
        total_episodes_var = int(profile_config["total_episodes"])

        if environment_mode != "custom" and environment_var not in SUPPORTED_Q_ENVIRONMENTS:
            raise ValueError(
                "Q-learning built-in mode supports only FrozenLake-v1 and Taxi-v3. "
                "Use custom mode for your own discrete environments."
            )

        profile_slug = _unique_profile_slug(profile_config, profile_position, used_profile_slugs)

        training_infos = {
            'environment': environment_var + " " + environment_library_var,
            'environmentName': environment_var,
            'environmentRawName': raw_environment_var,
            'agent': agent_var,
            'gamma': gamma_var,
            'policy': "Q Learning",
            'lr': lr_var,
            'environment_kwargs': environment_kwargs,
            'custom_environment_module': custom_environment_module,
            'trainingdate': str(datetime.datetime.now()),
            'total_timesteps': str(total_episodes_var) + " " + "episodes",
            'profile_name': profile_config.get("_profile_name", "profile_{}".format(profile_position + 1)),
            'profile_slug': profile_slug,
            'profile_index': profile_position + 1,
        }

        q_max_steps_var = int(profile_config["q_max_steps"])
        q_epsilon_var = float(profile_config["q_epsilon"])
        q_max_epsilon_var = float(profile_config["q_max_epsilon"])
        q_min_epsilon_var = float(profile_config["q_min_epsilon"])
        q_decay_rate_var = float(profile_config["q_decay_rate"])

        q_model = train_q(
            environment_var,
            agent_var,
            gamma_var,
            lr_var,
            total_episodes_var,
            q_max_steps_var,
            q_epsilon_var,
            q_max_epsilon_var,
            q_min_epsilon_var,
            q_decay_rate_var,
            environment_kwargs,
            custom_environment_module,
        )

        compatibility_model_name = agent_var + "_" + safe_artifact_name(environment_var)
        if is_multi_profile_mode:
            model_name = compatibility_model_name + "__" + profile_slug
            info_filename = "training_infos__{}.json".format(profile_slug)
        else:
            model_name = compatibility_model_name
            info_filename = "training_infos.json"

        serialized_model = pickle.dumps(q_model, protocol=pickle.HIGHEST_PROTOCOL)
        write_bytes(saved_models, model_name + ".pickle", serialized_model)
        write_json(saved_models, info_filename, training_infos)

        if profile_position == 0:
            if info_filename != "training_infos.json":
                write_json(saved_models, "training_infos.json", training_infos)
            if model_name != compatibility_model_name:
                write_bytes(saved_models, compatibility_model_name + ".pickle", serialized_model)

        training_manifest.append(
            {
                "profile_name": training_infos["profile_name"],
                "profile_slug": profile_slug,
                "profile_index": profile_position + 1,
                "model_name": model_name + ".pickle",
                "compatibility_model_name": compatibility_model_name + ".pickle",
                "training_infos_file": info_filename,
                "environment": environment_var,
                "agent": agent_var,
                "policy": "Q Learning",
            }
        )

    if is_multi_profile_mode:
        manifest = {
            "generated_at": str(datetime.datetime.now()),
            "total_profiles": len(training_manifest),
            "runs": training_manifest,
        }
        write_json(saved_models, "training_runs_manifest.json", manifest)
