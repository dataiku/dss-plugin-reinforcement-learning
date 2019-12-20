import dataiku

from dataiku.customrecipe import *

import pandas as pd, numpy as np
from dataiku import pandasutils as pdu

import datetime

from stable_baselines_training_wrapper import *

# Get the parameters
agent_var = get_recipe_config()['agent']

# Train the agent
if agent_var == "q":
    from q_training_wrapper import train_q_agent
    train_q_agent(agent_var)
else:
    train_agent(agent_var)
