import gymnasium as gym
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from bettermdptools.envs.blackjack_wrapper import BlackjackWrapper
from bettermdptools.envs.cartpole_wrapper import CartpoleWrapper
from bettermdptools.utils.test_env import TestEnv
import numpy as np
from bettermdptools.algorithms.rl import RL
import matrix_mdp
import mdptoolbox.example
from gymnasium.utils.env_checker import check_env


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


# base_env = gym.make('Blackjack-v1', render_mode=None)
# blackjack = BlackjackWrapper(base_env)

# # run VI
# V, V_track, pi = Planner(blackjack.P).value_iteration()

# # test policy
# test_scores = TestEnv.test_env(env=blackjack, n_iters=100, render=False, pi=pi, user_input=False)
# print(np.mean(test_scores))

# # Q-learning
# Q, V, pi, Q_track, pi_track = RL(blackjack).q_learning()

# #test policy
# test_scores = TestEnv.test_env(env=blackjack, n_iters=100, render=False, pi=pi, user_input=False)
# print(np.mean(test_scores))


############################################################################################################
############################################################################################################


# P: 
# a1   s1 |
#      s2 |
#      s3 | _   _   _
#           s1' s2' s3'

# a2   s1 |
#      s2 |
#      s3 | _   _   _
#           s1' s2' s3'

# R:
#     s1 |
#     s2 |
#     s3 | _   _ 
#         a1  a2

num_states  = 3
num_actions = 2
probability_of_fire = 0.1
P, R = mdptoolbox.example.forest(S=num_states, p=probability_of_fire)


s_0 = np.ones((num_states, )) / num_states
# rew = np.array(
#     [[[0,0,0],
#       [0,0,0],
#       [4,4,4]],

#      [[0,0,0],
#       [1,1,1],
#       [2,2,2]]]
# )
r_temp= np.array([
    [[0.0, 0],
     [0.0 ,1],
     [0.0 ,2]],

    [[0.0, 0],
    [0.0, 0],
    [0.0, 0]],

    [[0.0 , 0. ],
    [0.0 , 0. ],
    [4.0 , 0. ]]
])
p_env = np.array([[
        [0.1, 1],
        [0.1 ,1],
        [0.1 ,1]],

       [[0.9, 0],
        [0.0, 0],
        [0.0, 0]],

       [[0.0 , 0. ],
        [0.9 , 0. ],
        [0.9 , 0. ]]])

env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=s_0, p=p_env, r=r_temp, render_mode=None)

mdp_dict = {}
for s_1 in range(num_states):
    action_dict = {}

    for a in range(num_actions):
        action_dict[a] = []

        for s_2 in range(num_states):
            action_dict[a].append((P[a][s_1][s_2], s_2, R[s_1][a], False))

    mdp_dict[s_1] = action_dict

# run VI
V, V_track, pi = Planner(mdp_dict).value_iteration()

# test policy
test_scores = TestEnv.test_env(env=env, n_iters=100, render=False, pi=pi, user_input=False)
# print(np.mean(test_scores))

# # Q-learning
Q, V, pi, Q_track, pi_track = RL(env).q_learning()

# Plot setup
forest_actions = {
    0: "Wait", 
    1: "Cut", 
}
forest_map_size = (3, 1)
val_max, policy_map = Plots.get_policy_map(pi, V, forest_actions, forest_map_size)

Plots.plot_policy(
    val_max    = val_max, 
    directions = policy_map, 
    map_size   = forest_map_size, 
    title      = "Forest Mapped Policy"
)

Plots.values_heat_map(
    V, 
    "Forest\nValue Iteration State Values", 
    forest_map_size
)