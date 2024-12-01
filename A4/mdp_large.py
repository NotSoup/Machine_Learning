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


def plotting_func(pi, V):
    # Setup
    forest_actions = {
        0: "Wait", 
        1: "Cut", 
    }
    forest_map_size = (3, 1)

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

def run_large_mdp():

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
    num_states  = 3
    num_actions = 2
    probability_of_fire = 0.1
    P, R = mdptoolbox.example.forest(S=num_states, p=probability_of_fire)

    # 
    # Reshape values from P() and R() to cast MDP into a Gym Env
    # 
    s_0    = np.ones((num_states, )) / num_states
    r_temp = np.array([
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

    done_arr = np.array([
        [[True, True],
        [True ,True],
        [True ,True]],

        [[False, True],
        [False, True],
        [False, True]],

        [[False , True ],
        [False , True ],
        [True , True ]]
    ])

    p_env  = np.array([[
        [0.1, 1],
        [0.1 ,1],
        [0.1 ,1]],

        [[0.9, 0],
        [0.0, 0],
        [0.0, 0]],

        [[0.0 , 0. ],
        [0.9 , 0. ],
        [0.9 , 0. ]]
    ])

    env = gym.make('matrix_mdp/MatrixMDP-v0', p_0=s_0, p=p_env, r=r_temp, render_mode=None)
    check_env(env)
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
    print(np.mean(test_scores))
    plotting_func(pi, V)

    # 
    # Run Policy Iteration and test it
    # 
    V, V_track, pi = Planner(mdp_dict).policy_iteration(theta=1e-5)
    test_scores = TestEnv.test_env(env=env, n_iters=100, render=False, pi=pi, user_input=False)
    env.reset()
    print(np.mean(test_scores))
    plotting_func(pi, V)

    # # 
    # # Q-learning
    # # 
    Q, V, pi, Q_track, pi_track = RL(env).q_learning()
    test_scores = TestEnv.test_env(env=env, n_iters=100, render=False, pi=pi, user_input=False)
    print(np.mean(test_scores))
    plotting_func(pi, V)
