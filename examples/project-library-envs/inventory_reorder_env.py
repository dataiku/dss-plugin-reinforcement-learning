"""Example business environment for inventory replenishment decisions."""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    USE_GYMNASIUM_API = True
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

    USE_GYMNASIUM_API = False


class InventoryReorderEnv(gym.Env):
    """Inventory-control simulator for reinforcement learning experiments."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_stock=100,
        episode_length=30,
        demand_lambda=20.0,
        holding_cost=0.1,
        stockout_cost=2.0,
        order_cost=0.05,
        seed=0,
    ):
        super(InventoryReorderEnv, self).__init__()

        self.max_stock = int(max_stock)
        self.episode_length = int(episode_length)
        self.demand_lambda = float(demand_lambda)
        self.holding_cost = float(holding_cost)
        self.stockout_cost = float(stockout_cost)
        self.order_cost = float(order_cost)

        self._seed = int(seed)
        self._rng = np.random.RandomState(self._seed)

        self.action_levels = np.array([0, 10, 20, 30, 40, 50], dtype=np.int32)
        self.action_space = spaces.Discrete(len(self.action_levels))

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        self.current_stock = 0
        self.current_day = 0

    def _get_observation(self):
        return np.array(
            [
                float(self.current_stock) / float(self.max_stock),
                float(self.current_day) / float(max(self.episode_length - 1, 1)),
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        del options
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.RandomState(self._seed)

        self.current_stock = self.max_stock // 2
        self.current_day = 0
        observation = self._get_observation()

        if USE_GYMNASIUM_API:
            return observation, {}
        return observation

    def step(self, action):
        action = int(action)
        ordered_units = int(self.action_levels[action])

        self.current_stock = min(self.max_stock, self.current_stock + ordered_units)
        realized_demand = int(self._rng.poisson(lam=self.demand_lambda))

        sold_units = min(self.current_stock, realized_demand)
        stockout_units = max(0, realized_demand - self.current_stock)
        self.current_stock -= sold_units

        holding_penalty = self.holding_cost * float(self.current_stock)
        stockout_penalty = self.stockout_cost * float(stockout_units)
        order_penalty = self.order_cost * float(ordered_units)

        reward = -(holding_penalty + stockout_penalty + order_penalty)

        self.current_day += 1
        terminated = self.current_day >= self.episode_length
        truncated = False

        observation = self._get_observation()
        info = {
            "demand": realized_demand,
            "ordered_units": ordered_units,
            "stockout_units": stockout_units,
            "stock_level": self.current_stock,
        }

        if USE_GYMNASIUM_API:
            return observation, reward, terminated, truncated, info
        done = bool(terminated or truncated)
        return observation, reward, done, info


ENV_ID = "business/InventoryReorder-v0"

try:
    gym.spec(ENV_ID)
except Exception:
    gym.register(id=ENV_ID, entry_point="inventory_reorder_env:InventoryReorderEnv")
