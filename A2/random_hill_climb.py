import mlrose_ky as mlrose
import numpy as np
import matplotlib.pyplot as plt


# loop over seeds:
#   loop over problem sizes:
#     tune
#     run tuned models
#     save curves
#  average curves across seeds

seed = 90210
# np.random.seed = 90210

# 1) Define fitness function
fitness = mlrose.FourPeaks(t_pct=0.15)
    # fitness.evaluate(state) >>>> F(x) = Reward
        # so this thing just takes a state and spits out a score. NO OPTIMIZATION IS HAPPENING HERE 

# Graph with x-axis as 'Problem Size'
fitness_scores = []

for size in range(1, 20):
    # 2) Define optimization problem
    problem = mlrose.DiscreteOpt(
        length=size,                       # Length of state vector
        fitness_fn=fitness,             # Fitness function
        max_val=2                       # Possible different states
    )

    # just init a state [random seed here]
    # init_state = np.array([1, 1, 0, 1])
    init_state = np.random.randint(2, size=size)

    # 3) Run optimization algorithm (Simulated Annealing)
    best_state, best_fitness, curve = mlrose.random_hill_climb(
        problem,                        # Made in step 2
        # schedule = schedule,          # Decay scheduler (?)
        max_attempts = 10,              # on the tin
        max_iters = 1000,               # on the tin
        init_state = init_state,        # made this above
        random_state = seed,            # Random seed here
        # curve=True                      # Makes 3rd return into plottable curve
    )

    fitness_scores.append(best_fitness)

# # 3) Use Runner as a method for finding hyperparameters (gridsearch)
# rhc = mlrose.RHCRunner(
#     problem=problem,
#     experiment_name="rhc_4peak",
#     seed=seed,
#     iteration_list = 2 ** np.arange(11),
#     restart_list=[25,75,100],
#     max_attempts=5000,              # 24 mins
# )
# df_run_stats, df_run_curves = rhc.run()

# Plot
# plt.plot(df_run_stats, label="stats")
plt.plot(np.arange(1,20), fitness_scores)
plt.show()

# Score
print(fitness.evaluate(best_state)) # same as print(best_fitness)  ???????