import numpy as np
import random

from rl_compat import make_env, reset_env, step_env


def train_q(
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
    environment_kwargs=None,
    custom_environment_module=None,
):
    del agent_var

    env = make_env(
        environment_var,
        environment_kwargs=environment_kwargs,
        custom_environment_module=custom_environment_module,
    )

    if not hasattr(env.action_space, "n") or not hasattr(env.observation_space, "n"):
        raise ValueError(
            "Q-learning in this plugin requires a discrete action space and a discrete observation space. "
            "Use FrozenLake-v1 or Taxi-v3."
        )

    action_size = env.action_space.n
    state_size = env.observation_space.n
    qtable = np.zeros((state_size, action_size))

    total_episodes = int(total_episodes_var)
    learning_rate = float(lr_var)
    max_steps = int(q_max_steps_var)
    gamma = float(gamma_var)

    epsilon = float(q_epsilon_var)
    max_epsilon = float(q_max_epsilon_var)
    min_epsilon = float(q_min_epsilon_var)
    decay_rate = float(q_decay_rate_var)

    for episode in range(total_episodes):
        state = int(reset_env(env))
        done = False

        for _ in range(max_steps):
            exp_exp_tradeoff = random.uniform(0, 1)
            if exp_exp_tradeoff > epsilon:
                action = int(np.argmax(qtable[state, :]))
            else:
                action = env.action_space.sample()

            new_state, reward, done, _ = step_env(env, action)
            new_state = int(new_state)

            qtable[state, action] = qtable[state, action] + learning_rate * (
                reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action]
            )
            state = new_state

            if done:
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    env.close()
    return qtable


            
