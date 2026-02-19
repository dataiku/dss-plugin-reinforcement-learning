import dataiku

import datetime

from dataiku.customrecipe import *

from rl_compat import build_training_profile_configs, resolve_environment_config, safe_artifact_name
from managed_folder_io import write_json


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


def train_agent(agent):
    recipe_config = get_recipe_config()
    profile_configs = build_training_profile_configs(recipe_config)
    is_multi_profile_mode = recipe_config.get("training_profiles_mode", "single") == "multi"

    saved_models = dataiku.Folder(get_output_names_for_role("main_output")[0])
    if agent != "dqn":
        raise ValueError("Unsupported agent '{}' for this training wrapper".format(agent))

    from dqn import train_dqn, save_dqn_model_to_folder

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
        environment_library_var = profile_config["environment_library"]
        policy_var = profile_config["policy"]

        gamma_var = float(profile_config["gamma"])
        lr_var = float(profile_config["learning_rate"])
        total_timesteps_var = int(profile_config["total_timesteps"])

        profile_slug = _unique_profile_slug(profile_config, profile_position, used_profile_slugs)

        training_infos = {
            'environment': environment_var + " " + environment_library_var,
            'environmentName': environment_var,
            'environmentRawName': raw_environment_var,
            'agent': agent_var,
            'policy': policy_var,
            'gamma': gamma_var,
            'lr': lr_var,
            'environment_kwargs': environment_kwargs,
            'custom_environment_module': custom_environment_module,
            'trainingdate': str(datetime.datetime.now()),
            'total_timesteps': str(total_timesteps_var) + " " + "timesteps",
            'profile_name': profile_config.get("_profile_name", "profile_{}".format(profile_position + 1)),
            'profile_slug': profile_slug,
            'profile_index': profile_position + 1,
        }

        dqn_exploration_fraction_var = profile_config["dqn_exploration_fraction"]
        dqn_exploration_final_eps_var = profile_config["dqn_exploration_final_eps"]
        dqn_buffer_size_var = int(profile_config["dqn_buffer_size"])
        dqn_prioritized_replay_var = profile_config["dqn_prioritized_replay"]

        dqn_double_q_var = profile_config["dqn_double_q"]
        dqn_target_network_update_freq_var = int(profile_config["dqn_target_network_update_freq"])
        dqn_train_freq_var = int(profile_config["dqn_train_freq"])
        dqn_batch_size_var = int(profile_config["dqn_batch_size"])

        model = train_dqn(
            total_timesteps_var,
            policy_var,
            environment_var,
            gamma_var,
            lr_var,
            dqn_buffer_size_var,
            dqn_exploration_fraction_var,
            dqn_exploration_final_eps_var,
            dqn_train_freq_var,
            dqn_batch_size_var,
            dqn_double_q_var,
            dqn_target_network_update_freq_var,
            dqn_prioritized_replay_var,
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

        save_dqn_model_to_folder(model, saved_models, model_name)
        write_json(saved_models, info_filename, training_infos)

        if profile_position == 0:
            if info_filename != "training_infos.json":
                write_json(saved_models, "training_infos.json", training_infos)
            if model_name != compatibility_model_name:
                save_dqn_model_to_folder(model, saved_models, compatibility_model_name)

        training_manifest.append(
            {
                "profile_name": training_infos["profile_name"],
                "profile_slug": profile_slug,
                "profile_index": profile_position + 1,
                "model_name": model_name,
                "compatibility_model_name": compatibility_model_name,
                "training_infos_file": info_filename,
                "environment": environment_var,
                "agent": agent_var,
                "policy": policy_var,
            }
        )

    if is_multi_profile_mode:
        manifest = {
            "generated_at": str(datetime.datetime.now()),
            "total_profiles": len(training_manifest),
            "runs": training_manifest,
        }
        write_json(saved_models, "training_runs_manifest.json", manifest)
