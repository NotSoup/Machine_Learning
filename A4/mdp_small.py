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


def plotting_func(num_states, pi, V):
    # Setup
    forest_actions = {
        0: "Wait", 
        1: "Cut", 
    }
    forest_map_size = (num_states, 1)

    # Calculate Plot 1
    val_max, policy_map = Plots.get_policy_map(pi, V, forest_actions, forest_map_size)
    Plots.plot_policy(
        val_max    = val_max, 
        directions = policy_map, 
        map_size   = forest_map_size, 
        title      = "Forest Mapped Policy"
    )

    # Calculate Plot 2
    Plots.values_heat_map(
        V, 
        "Forest\nValue Iteration State Values", 
        forest_map_size
    )


# def run_small_mdp():
"""
Probability Transition Matitrix & Rewards
env.P -> dict{states: 
                    {
                    action_1: [
                                (probability to s2, s2, reward, terminal?) ,
                                (probability to s3, s3, reward, terminal?) ,
                                (probability to sN, sN, reward, terminal?)
                            ],
                    action_N: [
                                (probability to s2, s2, reward, terminal?) ,
                                (probability to s3, s3, reward, terminal?) ,
                                (probability to sN, sN, reward, terminal?)
                            ]
                    }
                }

P: 
    a1  s1 |
        s2 |
        s3 | _   _   _
            s1' s2' s3'

    a2  s1 |
        s2 |
        s3 | _   _   _
            s1' s2' s3'

R:
    s1 |
    s2 |
    s3 | _   _ 
        a1  a2
"""

# 
# Setup MDPToolbox to create Forest Management MDP and:
#   P(s'|s,a)
#   R(s,a)
#
num_actions = 2
r1 = 4
r2 = 2
r3 = 1
probability_of_fire = 0.1

for num_states in range(100,500, 50):
    # num_states  = 3
    P, R = mdptoolbox.example.forest(S=num_states, r1=4, r2=2, p=probability_of_fire)

    # 
    # Reshape values from P() and R() to cast MDP into a Gym Env
    # 
    s_0    = np.ones((num_states, )) / num_states

    # r_temp = np.array([
    #     [[0.0, 0],
    #     [0.0 ,1],
    #     [0.0 ,2]],

    #     [[0.0, 0],
    #     [0.0, 0],
    #     [0.0, 0]],

    #     [[0.0 , 0. ],
    #     [0.0 , 0. ],
    #     [4.0 , 0. ]]
    # ])

    r_temp = np.zeros((num_states,num_states,num_actions))
    r_temp[-1][-1][0] = r1
    r_temp[0][-1][-1] = r2
    r_temp[0, 1:-1, -1] = r3

    # done_arr = np.array([
    #     [[True, True],
    #     [True ,True],
    #     [True ,True]],

    #     [[False, True],
    #     [False, True],
    #     [False, True]],

    #     [[False , True ],
    #     [False , True ],
    #     [True , True ]]
    # ])

    done_arr=np.zeros((num_states,num_states,num_actions))
    done_arr[-1, -1, 0] = True
    done_arr[:, :, -1] = True
    done_arr[0, :, :] = True

    # p_env  = np.array([
    #     [[0.1, 1],
    #     [0.1 ,1],
    #     [0.1 ,1]],

    #     [[0.9, 0],
    #     [0.0, 0],
    #     [0.0, 0]],

    #     [[0.0 , 0. ],
    #     [0.9 , 0. ],
    #     [0.9 , 0. ]]
    # ])

    p_env = np.swapaxes(P, 2, 0)


    env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=s_0, p=p_env, r=r_temp, render_mode=None)
    # check_env(env)
    env.reset()

    # 
    # Reshape P() and R() to Dict to comply with Planner(env.P : dict).[policy or value]_iteration() 
    # 
    mdp_dict = {}

    for s_1 in range(num_states):
        action_dict = {}

        for a in range(num_actions):
            action_dict[a] = []

            for s_2 in range(num_states):
                action_dict[a].append((P[a][s_1][s_2], s_2, r_temp[s_2][s_1][a], done_arr[s_2][s_1][a]))

        mdp_dict[s_1] = action_dict


    # 
    # Run Value Iteration and test it
    # 
    V, V_track, pi = Planner(mdp_dict).value_iteration()
    test_scores = TestEnv.test_env(env=env, n_iters=100, render=False, pi=pi, user_input=False)
    env.reset()
    print(f"Mean of test scores: {np.mean(test_scores)}")
    print(f"Num of States: \t\t {num_states}")
    print(f"Iterations: \t\t {len(np.trim_zeros(np.mean(V_track, axis=1), 'b'))}")
    max_value_per_iter = np.trim_zeros(np.mean(V_track, axis=1), 'b')
Plots.v_iters_plot(max_value_per_iter, "titties lol")
# plotting_func(num_states, pi, V)


# 
# Run Policy Iteration and test it
# 
V, V_track, pi = Planner(mdp_dict).policy_iteration()
test_scores = TestEnv.test_env(env=env, n_iters=100, render=False, pi=pi, user_input=False)
env.reset()
print(np.mean(test_scores))
plotting_func(num_states, pi, V)

# # 
# # Q-learning  (runtime: ~5mins)
# # 
Q, V, pi, Q_track, pi_track = RL(env).q_learning()
test_scores = TestEnv.test_env(env=env, n_iters=100, render=False, pi=pi, user_input=False)
print(np.mean(test_scores))
plotting_func(num_states, pi, V)
