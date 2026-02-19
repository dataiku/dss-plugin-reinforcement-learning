"""Compatibility helpers for Gym/Gymnasium API differences."""

import importlib
import json
import re

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym


ENV_ALIASES = {
    "FrozenLake-v0": "FrozenLake-v1",
    "FrozenLake8x8-v0": "FrozenLake-v1",
    "Pendulum-v0": "Pendulum-v1",
}


def normalize_environment_id(environment_id):
    return ENV_ALIASES.get(environment_id, environment_id)


def parse_environment_kwargs(raw_environment_kwargs):
    if raw_environment_kwargs is None:
        return {}
    if isinstance(raw_environment_kwargs, dict):
        return raw_environment_kwargs
    if isinstance(raw_environment_kwargs, str):
        stripped_value = raw_environment_kwargs.strip()
        if not stripped_value:
            return {}
        parsed_value = json.loads(stripped_value)
        if not isinstance(parsed_value, dict):
            raise ValueError("Environment kwargs must be a JSON object")
        return parsed_value
    raise ValueError("Unsupported type for environment kwargs")


def build_training_profile_configs(recipe_config):
    profile_mode = recipe_config.get("training_profiles_mode", "single")
    if profile_mode != "multi":
        return [dict(recipe_config)]

    raw_profiles_json = recipe_config.get("training_profiles_json", "").strip()
    if not raw_profiles_json:
        raise ValueError("training_profiles_json must be provided when training_profiles_mode is multi")

    parsed_profiles = json.loads(raw_profiles_json)
    if not isinstance(parsed_profiles, list) or len(parsed_profiles) == 0:
        raise ValueError("training_profiles_json must be a non-empty JSON array")

    profile_configs = []
    for index, profile in enumerate(parsed_profiles):
        if not isinstance(profile, dict):
            raise ValueError("Each training profile must be a JSON object")

        merged_profile_config = dict(recipe_config)
        merged_profile_config.update(profile)

        profile_name = profile.get("name") or profile.get("profile_name") or ("profile_{}".format(index + 1))
        merged_profile_config["_profile_name"] = str(profile_name)
        merged_profile_config["_profile_index"] = index + 1

        if "environment_kwargs" in profile and "custom_environment_kwargs_json" not in merged_profile_config:
            merged_profile_config["custom_environment_kwargs_json"] = json.dumps(profile["environment_kwargs"])

        raw_kwargs = merged_profile_config.get("custom_environment_kwargs_json")
        if isinstance(raw_kwargs, dict):
            merged_profile_config["custom_environment_kwargs_json"] = json.dumps(raw_kwargs)

        profile_configs.append(merged_profile_config)

    return profile_configs


def resolve_environment_config(recipe_config):
    environment_mode = recipe_config.get("environment_mode", "builtin")
    custom_environment_module = recipe_config.get("custom_environment_module", "").strip()

    if environment_mode == "custom":
        raw_environment_id = recipe_config.get("custom_environment_id", "").strip()
        if not raw_environment_id:
            raise ValueError("custom_environment_id must be provided when environment_mode is custom")
        raw_environment_kwargs = recipe_config.get("custom_environment_kwargs_json", "{}")
    else:
        raw_environment_id = recipe_config["environment"]
        raw_environment_kwargs = {}

    return {
        "environment_mode": environment_mode,
        "raw_environment_id": raw_environment_id,
        "environment_id": normalize_environment_id(raw_environment_id),
        "environment_kwargs": parse_environment_kwargs(raw_environment_kwargs),
        "custom_environment_module": custom_environment_module,
    }


def maybe_import_custom_environment_module(custom_environment_module):
    if not custom_environment_module:
        return

    module_name = str(custom_environment_module).strip()
    if not module_name:
        return

    # Accept user-friendly inputs from DSS project libraries:
    # - "my_env.py"
    # - "subfolder/my_env.py"
    # - "subfolder.my_env"
    if module_name.endswith(".py"):
        module_name = module_name[:-3]
    module_name = module_name.replace("\\", "/").strip("/")
    module_name = module_name.replace("/", ".")

    try:
        importlib.import_module(module_name)
    except Exception as err:
        raise ValueError(
            "Could not import custom environment module '{}'. "
            "Use a module path available in the project library/code env, for example "
            "'my_env' or 'package.subpackage.my_env'.".format(custom_environment_module)
        ) from err


def safe_artifact_name(value):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    sanitized = sanitized.strip("._")
    return sanitized or "artifact"


def make_env(environment_id, environment_kwargs=None, custom_environment_module=None):
    maybe_import_custom_environment_module(custom_environment_module)

    normalized_environment_id = normalize_environment_id(environment_id)
    environment_kwargs = dict(environment_kwargs or {})
    if environment_id == "FrozenLake8x8-v0":
        environment_kwargs.setdefault("map_name", "8x8")
    return gym.make(normalized_environment_id, **environment_kwargs)


def reset_env(env):
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def step_env(env, action):
    step_result = env.step(action)
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
        return observation, reward, done, info
    observation, reward, done, info = step_result
    return observation, reward, bool(done), info
