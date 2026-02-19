# Reinforcement Learning in Dataiku

Train, evaluate, and visualize reinforcement learning agents in Dataiku.

## Plugin Scope
- Algorithms:
  - `dqn` (Stable-Baselines3 DQN)
  - `q` (tabular Q-learning, discrete state/action spaces)
- Environment sources:
  - built-in Gym/Gymnasium environment ID
  - custom environment ID + optional module import
- Storage:
  - DSS managed folders via object APIs (local filesystem and cloud/object storage backends)

## Components
- Train recipe: `reinforcement-learning-train`
- Test recipe: `reinforcement-learning-test`
- Webapp template: `vizualisation`

## Train Recipe
Input:
- none

Output:
- managed folder containing model artifacts and training metadata

Key parameters:
- `agent`: `dqn` or `q`
- `environment_mode`: `builtin` or `custom`
- `custom_environment_id` / `custom_environment_module` / `custom_environment_kwargs_json` (for custom mode)
- hyperparameters (`gamma`, learning rate, timesteps/episodes, agent-specific settings)

### Multi-Profile Train Runs
Use one recipe run to train multiple configurations:
- `training_profiles_mode = multi`
- `training_profiles_json = [...]`

Example:
```json
[
  {
    "name": "baseline",
    "agent": "dqn",
    "environment_mode": "custom",
    "custom_environment_id": "business/InventoryReorder-v0",
    "custom_environment_module": "inventory_reorder_env",
    "custom_environment_kwargs_json": "{\"episode_length\": 30, \"demand_lambda\": 20, \"seed\": 42}",
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "total_timesteps": 50000
  },
  {
    "name": "high_service",
    "agent": "dqn",
    "environment_mode": "custom",
    "custom_environment_id": "business/InventoryReorder-v0",
    "custom_environment_module": "inventory_reorder_env",
    "custom_environment_kwargs_json": "{\"episode_length\": 30, \"demand_lambda\": 24, \"stockout_cost\": 4.0, \"seed\": 42}",
    "learning_rate": 0.0003,
    "gamma": 0.995,
    "total_timesteps": 70000
  }
]
```

Typical output files:
- `training_infos.json`
- `training_infos__<profile_slug>.json` (multi-profile)
- `training_runs_manifest.json` (multi-profile)
- model files (`.zip` for DQN, `.pickle` for Q-learning)

## Test Recipe
Input:
- managed folder with trained models and training info

Output:
- managed folder with testing result JSON files

Model selection modes:
- `manual`
- `manifest_single`
- `manifest_batch_selected` (comma/newline slug list)
- `manifest_batch_all`

For manifest modes, the recipe reads `training_runs_manifest.json` and each profile’s training info file.

## Webapp
Use the `vizualisation` webapp template to inspect test outputs:
- score trend
- score distribution
- run comparison table
- per-run metadata

Webapp parameters:
- `replay_folder`: test output folder (required)
- `training_models_folder`: train output folder (optional, enables manifest profile helper)

## Custom Environment Usage
Provide a Gym/Gymnasium environment and register an ID.

Workflow:
1. Add your env module to Dataiku project library or plugin code env.
2. Ensure module import registers the environment ID.
3. In Train/Test recipe:
   - set `environment_mode = custom`
   - set `custom_environment_id` to the registered ID
   - set `custom_environment_module` if registration happens on module import
   - optionally pass `custom_environment_kwargs_json`

Accepted module formats:
- `my_env`
- `my_env.py`
- `package.subpackage.my_env`
- `subfolder/my_env.py`

## Included Example Custom Environments
- `inventory_reorder_env` -> `business/InventoryReorder-v0`
- `price_optimization_env` -> `business/PriceOptimization-v0`
- `call_center_staffing_env` -> `business/CallCenterStaffing-v0` (fully discrete, suitable for `q`)

Separate, project-library-ready copies of these files are available in `examples/project-library-envs/`.

## Notes
- Use `q` only with discrete observation and action spaces.
- Use `dqn` for broader observation spaces with neural-network policies.

## Source and Issues
- Source: `https://github.com/dataiku/dss-plugin-reinforcement-learning`
- Issues: `https://github.com/dataiku/dss-plugin-reinforcement-learning/issues`
