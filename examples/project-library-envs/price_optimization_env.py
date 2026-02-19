"""Example business environment for price optimization decisions."""

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    USE_GYMNASIUM_API = True
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

    USE_GYMNASIUM_API = False


class PriceOptimizationEnv(gym.Env):
    """Pricing simulator with demand elasticity and inventory effects."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_price=10.0,
        unit_cost=4.0,
        max_stock=300,
        initial_stock=220,
        episode_length=30,
        base_daily_demand=40.0,
        demand_volatility=0.15,
        elasticity=2.0,
        holding_cost=0.03,
        stockout_penalty=1.5,
        restock_amount=10,
        seed=0,
    ):
        super(PriceOptimizationEnv, self).__init__()

        self.base_price = float(base_price)
        self.unit_cost = float(unit_cost)
        self.max_stock = int(max_stock)
        self.initial_stock = int(initial_stock)
        self.episode_length = int(episode_length)
        self.base_daily_demand = float(base_daily_demand)
        self.demand_volatility = float(demand_volatility)
        self.elasticity = float(elasticity)
        self.holding_cost = float(holding_cost)
        self.stockout_penalty = float(stockout_penalty)
        self.restock_amount = int(restock_amount)

        self._seed = int(seed)
        self._rng = np.random.RandomState(self._seed)

        self.price_multipliers = np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.price_multipliers))

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 3.0, 1.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

        self.current_stock = 0
        self.current_day = 0
        self.current_demand_forecast = self.base_daily_demand

    def _sample_forecast(self):
        day_cycle = np.sin((2.0 * np.pi * self.current_day) / max(self.episode_length, 1))
        baseline = self.base_daily_demand * (1.0 + 0.2 * day_cycle)
        noise_factor = self._rng.normal(loc=1.0, scale=self.demand_volatility)
        return max(1.0, baseline * max(0.1, noise_factor))

    def _get_observation(self):
        return np.array(
            [
                float(self.current_stock) / float(max(self.max_stock, 1)),
                float(self.current_demand_forecast) / float(max(self.base_daily_demand, 1.0)),
                float(self.current_day) / float(max(self.episode_length - 1, 1)),
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        del options
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.RandomState(self._seed)

        self.current_day = 0
        self.current_stock = min(self.max_stock, max(0, self.initial_stock))
        self.current_demand_forecast = self._sample_forecast()

        observation = self._get_observation()
        if USE_GYMNASIUM_API:
            return observation, {}
        return observation

    def step(self, action):
        action = int(action)
        multiplier = float(self.price_multipliers[action])
        price = self.base_price * multiplier

        demand_mean = self.current_demand_forecast * (multiplier ** (-self.elasticity))
        realized_demand = int(self._rng.poisson(lam=max(0.1, demand_mean)))

        units_sold = min(self.current_stock, realized_demand)
        stockout_units = max(0, realized_demand - self.current_stock)
        self.current_stock -= units_sold

        revenue = float(units_sold) * price
        cogs = float(units_sold) * self.unit_cost
        holding_penalty = self.holding_cost * float(self.current_stock)
        stockout_penalty = self.stockout_penalty * float(stockout_units)
        reward = revenue - cogs - holding_penalty - stockout_penalty

        self.current_stock = min(self.max_stock, self.current_stock + self.restock_amount)
        self.current_day += 1
        self.current_demand_forecast = self._sample_forecast()

        terminated = self.current_day >= self.episode_length
        truncated = False
        observation = self._get_observation()
        info = {
            "price": price,
            "realized_demand": realized_demand,
            "units_sold": units_sold,
            "stockout_units": stockout_units,
            "stock_level": self.current_stock,
            "profit": reward,
        }

        if USE_GYMNASIUM_API:
            return observation, reward, terminated, truncated, info
        done = bool(terminated or truncated)
        return observation, reward, done, info


ENV_ID = "business/PriceOptimization-v0"

try:
    gym.spec(ENV_ID)
except Exception:
    gym.register(id=ENV_ID, entry_point="price_optimization_env:PriceOptimizationEnv")
