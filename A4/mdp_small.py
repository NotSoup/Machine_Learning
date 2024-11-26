import gymnasium as gym
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
from bettermdptools.envs.cartpole_wrapper import CartpoleWrapper
from bettermdptools.utils.test_env import TestEnv
import numpy as np
from bettermdptools.algorithms.rl import RL



# # make gym environment 
# frozen_lake = gym.make('FrozenLake8x8-v1', render_mode="rgb_array")

# # Init plotting stuff
# fl_actions = {
#     0: "←", 
#     1: "↓", 
#     2: "→", 
#     3: "↑"
# }
# fl_map_size=(8,8)

# # Probability Transition Matitrix & Rewards
# #   env.P -> dict{states: 
# #                   {
# #               action_1: [ (probability to s2, s2, reward, terminal?) ,
# #                           (probability to s3, s3, reward, terminal?) ,
# #                           (probability to sN, sN, reward, terminal?)
# #                          ]
# #               action_N: [ (probability to s2, s2, reward, terminal?) ,
# #                           (probability to s3, s3, reward, terminal?) ,
# #                           (probability to sN, sN, reward, terminal?)
# #                          ]
# #                   }
# #               }

# # run VI
# V, V_track, pi = Planner(frozen_lake.P).value_iteration()

# # run PI

# val_max, policy_map = Plots.get_policy_map(pi, V, fl_actions, fl_map_size)
# Plots.plot_policy(
#     val_max=val_max, 
#     directions=policy_map, 
#     map_size=fl_map_size, 
#     title="FL Mapped Policy"
# )

# #plot state values
# fl_map_size=(8,8)
# Plots.values_heat_map(V, "Frozen Lake\nValue Iteration State Values", fl_map_size)


############################################################################################################
############################################################################################################


base_env = gym.make('Blackjack-v1', render_mode=None)
blackjack = BlackjackWrapper(base_env)

# run VI
V, V_track, pi = Planner(blackjack.P).value_iteration()

# test policy
test_scores = TestEnv.test_env(env=blackjack, n_iters=100, render=False, pi=pi, user_input=False)
print(np.mean(test_scores))

# Q-learning
Q, V, pi, Q_track, pi_track = RL(blackjack).q_learning()

#test policy
test_scores = TestEnv.test_env(env=blackjack, n_iters=100, render=False, pi=pi, user_input=False)
print(np.mean(test_scores))


############################################################################################################
############################################################################################################


import mdptoolbox.example

P, R = mdptoolbox.example.forest()

import matrix_mdp

env = gym.make('matrix_mdp/MatrixMDP-v0')