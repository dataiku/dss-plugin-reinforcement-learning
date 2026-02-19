from pathlib import Path
import tempfile
from stable_baselines3 import DQN

from rl_compat import make_env, safe_artifact_name
from managed_folder_io import first_existing_path, read_bytes, write_bytes


SUPPORTED_POLICIES = {
    "MlpPolicy": "MlpPolicy",
    "CnnPolicy": "CnnPolicy",
    "MlpLstmPolicy": "MlpPolicy",
    "MlpLnLstmPolicy": "MlpPolicy",
    "CnnLstmPolicy": "MlpPolicy",
    "CnnLnLstmPolicy": "MlpPolicy",
}


def train_dqn(
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
    environment_kwargs=None,
    custom_environment_module=None,
):
    del dqn_double_q_var
    del dqn_prioritized_replay_var

    env = make_env(
        environment_var,
        environment_kwargs=environment_kwargs,
        custom_environment_module=custom_environment_module,
    )
    resolved_policy = SUPPORTED_POLICIES.get(policy_var, "MlpPolicy")
    model = DQN(
        policy=resolved_policy,
        env=env,
        gamma=float(gamma_var),
        learning_rate=float(lr_var),
        buffer_size=int(dqn_buffer_size_var),
        exploration_fraction=float(dqn_exploration_fraction_var),
        exploration_final_eps=float(dqn_exploration_final_eps_var),
        train_freq=int(dqn_train_freq_var),
        batch_size=int(dqn_batch_size_var),
        target_update_interval=int(dqn_target_network_update_freq_var),
        verbose=0,
    )

    model.learn(total_timesteps=int(total_timesteps_var))
    env.close()
    return model


def save_dqn_model_to_folder(model, folder, model_name):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_prefix = Path(tmp_dir) / "dqn_model"
        model.save(str(tmp_prefix))
        model_zip_path = tmp_prefix.with_suffix(".zip")
        model_bytes = model_zip_path.read_bytes()
        write_bytes(folder, model_name + ".zip", model_bytes)


def _resolve_model_object_path(saved_models_folder, agent_var, environment_var, raw_environment_var=None, model_name_override=None):
    candidate_paths = []
    if model_name_override:
        normalized_override = str(model_name_override).rstrip("/")
        candidate_paths.append(normalized_override)
        candidate_paths.append(normalized_override + ".zip")
    else:
        normalized_model_name = agent_var + "_" + safe_artifact_name(environment_var)
        raw_model_name = normalized_model_name
        if raw_environment_var and raw_environment_var != environment_var:
            raw_model_name = agent_var + "_" + safe_artifact_name(raw_environment_var)

        candidate_paths.extend(
            [
                normalized_model_name + ".zip",
                normalized_model_name,
            ]
        )
        if raw_model_name != normalized_model_name:
            candidate_paths.extend([raw_model_name + ".zip", raw_model_name])

    resolved_path = first_existing_path(saved_models_folder, candidate_paths)
    if not resolved_path:
        raise ValueError("Could not find DQN model object in managed folder for candidates: {}".format(candidate_paths))
    return resolved_path


def test_dqn(saved_models_folder, agent_var, environment_var, raw_environment_var=None, model_name_override=None):
    model_object_path = _resolve_model_object_path(
        saved_models_folder,
        agent_var,
        environment_var,
        raw_environment_var,
        model_name_override,
    )
    model_bytes = read_bytes(saved_models_folder, model_object_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_model_path = Path(tmp_dir) / "model.zip"
        local_model_path.write_bytes(model_bytes)
        model = DQN.load(str(local_model_path))
    return model
