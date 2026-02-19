from dataiku.customrecipe import *

from stable_baselines_training_wrapper import *

agent_var = get_recipe_config()['agent']

if agent_var == "q":
    from q_training_wrapper import train_q_agent
    train_q_agent(agent_var)
else:
    train_agent(agent_var)
