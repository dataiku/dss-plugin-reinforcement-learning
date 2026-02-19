# Project Library Environment Examples

These files are standalone Gym/Gymnasium environments intended for upload to a Dataiku project library.

Included modules:
- `inventory_reorder_env.py` -> `business/InventoryReorder-v0`
- `price_optimization_env.py` -> `business/PriceOptimization-v0`
- `call_center_staffing_env.py` -> `business/CallCenterStaffing-v0`

How to use:
1. Upload one or more files to the Dataiku project library.
2. In Train/Test recipe:
   - set `Environment source` to `Custom environment ID`
   - set `Custom Environment ID` to the ID above
   - set `Module to import` to the module name (for example `inventory_reorder_env` or `inventory_reorder_env.py`)
3. Optionally set `Environment kwargs` as JSON.
