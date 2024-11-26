import gymnasium as gym
from bettermdptools.algorithms.planner import Planner
from bettermdptools.utils.plots import Plots
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


# make gym environment 
frozen_lake = gym.make('FrozenLake8x8-v1', render_mode="rgb_array")

# Init plotting stuff
fl_actions = {
    0: "←", 
    1: "↓", 
    2: "→", 
    3: "↑"
}
fl_map_size=(8,8)

# Probability Transition Matitrix & Rewards
#   env.P -> dict{states: 
#                   {
#               action_1: [ (probability to s2, s2, reward, terminal?) ,
#                           (probability to s3, s3, reward, terminal?) ,
#                           (probability to sN, sN, reward, terminal?)
#                          ]
#               action_N: [ (probability to s2, s2, reward, terminal?) ,
#                           (probability to s3, s3, reward, terminal?) ,
#                           (probability to sN, sN, reward, terminal?)
#                          ]
#                   }
#               }

# run VI
V, V_track, pi = Planner(frozen_lake.P).value_iteration()

# run PI

val_max, policy_map = Plots.get_policy_map(pi, V, fl_actions, fl_map_size)
Plots.plot_policy(
    val_max=val_max, 
    directions=policy_map, 
    map_size=fl_map_size, 
    title="FL Mapped Policy"
)

#plot state values
fl_map_size=(8,8)
Plots.values_heat_map(V, "Frozen Lake\nValue Iteration State Values", fl_map_size)

# num_eval_episodes = 4

# env = gym.make("FrozenLake8x8-v1", render_mode="rgb_array")  # replace with your environment
# env = RecordVideo(env, video_folder="frozenLake-agent", name_prefix="eval",
#                   episode_trigger=lambda x: True)
# env = RecordEpisodeStatistics(env, deque_size=num_eval_episodes)

# for episode_num in range(num_eval_episodes):
#     obs, info = env.reset()

#     episode_over = False
#     while not episode_over:
#         action = env.action_space.sample()  # replace with actual agent
#         obs, reward, terminated, truncated, info = env.step(action)

#         episode_over = terminated or truncated
# env.close()

# print(f'Episode time taken: {env.time_queue}')
# print(f'Episode total rewards: {env.return_queue}')
# print(f'Episode lengths: {env.length_queue}')