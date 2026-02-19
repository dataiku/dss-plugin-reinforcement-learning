"""Example business environment with discrete states/actions for Q-learning."""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    USE_GYMNASIUM_API = True
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

    USE_GYMNASIUM_API = False


class CallCenterStaffingEnv(gym.Env):
    """Discrete staffing control simulator for tabular Q-learning."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_queue=20,
        max_agents=10,
        initial_queue=5,
        initial_agents=4,
        episode_length=40,
        demand_lambda=6.0,
        calls_per_agent=2,
        wait_cost=1.5,
        staffing_cost=0.4,
        adjustment_cost=0.2,
        seed=0,
    ):
        super(CallCenterStaffingEnv, self).__init__()

        self.max_queue = int(max_queue)
        self.max_agents = int(max_agents)
        self.initial_queue = int(initial_queue)
        self.initial_agents = int(initial_agents)
        self.episode_length = int(episode_length)
        self.demand_lambda = float(demand_lambda)
        self.calls_per_agent = int(calls_per_agent)
        self.wait_cost = float(wait_cost)
        self.staffing_cost = float(staffing_cost)
        self.adjustment_cost = float(adjustment_cost)

        self._seed = int(seed)
        self._rng = np.random.RandomState(self._seed)

        self.action_space = spaces.Discrete(3)

        self.time_buckets = 4
        self.n_states = (self.max_queue + 1) * (self.max_agents + 1) * self.time_buckets
        self.observation_space = spaces.Discrete(self.n_states)

        self.current_queue = 0
        self.current_agents = 0
        self.current_step = 0

    def _time_bucket(self):
        if self.episode_length <= 1:
            return 0
        return min(
            self.time_buckets - 1,
            int((self.current_step / float(self.episode_length - 1)) * self.time_buckets),
        )

    def _state_index(self):
        queue_component = int(np.clip(self.current_queue, 0, self.max_queue))
        agent_component = int(np.clip(self.current_agents, 0, self.max_agents))
        time_component = int(np.clip(self._time_bucket(), 0, self.time_buckets - 1))
        return (
            queue_component * (self.max_agents + 1) * self.time_buckets
            + agent_component * self.time_buckets
            + time_component
        )

    def reset(self, seed=None, options=None):
        del options
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.RandomState(self._seed)

        self.current_queue = int(np.clip(self.initial_queue, 0, self.max_queue))
        self.current_agents = int(np.clip(self.initial_agents, 0, self.max_agents))
        self.current_step = 0

        observation = self._state_index()
        if USE_GYMNASIUM_API:
            return observation, {}
        return observation

    def step(self, action):
        action = int(action)
        delta_staff = action - 1
        previous_agents = self.current_agents
        self.current_agents = int(np.clip(self.current_agents + delta_staff, 0, self.max_agents))

        incoming_calls = int(self._rng.poisson(lam=max(0.1, self.demand_lambda)))
        service_capacity = int(self.current_agents * self.calls_per_agent)

        self.current_queue = int(np.clip(self.current_queue + incoming_calls - service_capacity, 0, self.max_queue))
        served_calls = min(service_capacity, self.current_queue + service_capacity)

        queue_penalty = self.wait_cost * float(self.current_queue)
        staffing_penalty = self.staffing_cost * float(self.current_agents)
        adjustment_penalty = self.adjustment_cost * float(abs(self.current_agents - previous_agents))
        reward = -(queue_penalty + staffing_penalty + adjustment_penalty)

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False

        observation = self._state_index()
        info = {
            "incoming_calls": incoming_calls,
            "served_calls": served_calls,
            "queue_size": self.current_queue,
            "agents": self.current_agents,
        }

        if USE_GYMNASIUM_API:
            return observation, reward, terminated, truncated, info
        done = bool(terminated or truncated)
        return observation, reward, done, info


ENV_ID = "business/CallCenterStaffing-v0"

try:
    gym.spec(ENV_ID)
except Exception:
    gym.register(id=ENV_ID, entry_point="call_center_staffing_env:CallCenterStaffingEnv")
