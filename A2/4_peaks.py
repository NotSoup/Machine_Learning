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

# 1) Define fitness function
fitness = mlrose.FourPeaks(t_pct=0.1)
    # fitness.evaluate(state) >>>> F(x) = Reward
        # so this thing just takes a state and spits out a score. NO OPTIMIZATION IS HAPPENING HERE 

# Graph with x-axis as 'Problem Size'
rhc_fitness_scores = []
sa_fitness_scores = []
ga_fitness_scores = []

problem_range = range(2, 10)

for size in problem_range:

    ############# 2) Define optimization problem #############
    problem = mlrose.DiscreteOpt(
        length=size,                    # Length of state vector
        fitness_fn=fitness,             # Fitness function
        max_val=2                       # Possible different states
    )

    # just init a state [random seed here]
    # init_state = np.array([1, 1, 0, 1])
    init_state = np.random.randint(2, size=size)

    ############# 3) Run optimization algorithms #############
    # (Restart-Random Hill Climb)
    rhc_best_state, rhc_best_fitness, rhc_curve = mlrose.random_hill_climb(
        problem,                        # Made in step 2
        max_attempts = 10,              # on the tin
        max_iters = 7,             # on the tin
        # restarts=3,                     # ?
        init_state = init_state,        # made this above
        curve=True,                     # Makes 3rd return into plottable curve
        random_state = seed,            # Random seed here
    )

    # (Simulated Annealing)
    sa_best_state, sa_best_fitness, sa_curve = mlrose.simulated_annealing(
        problem,                        # Made in step 2
        schedule = mlrose.GeomDecay(),  # Decay scheduler
        max_attempts = 10,              # on the tin
        max_iters = 7,             # on the tin
        init_state = init_state,        # made this above
        curve=True,                     # Makes 3rd return into plottable curve
        random_state = seed,            # Random seed here
    )

    # (Genetic Algorithm)
    ga_best_state, ga_best_fitness, ga_curve = mlrose.genetic_alg(
        problem,                        # Made in step 2
        pop_size = 200,                 #
        pop_breed_percent = 0.75,       #
        elite_dreg_ratio = 0.99,        #
        minimum_elites = 0,             #
        minimum_dregs = 0,              #
        mutation_prob = 0.1,            #
        max_attempts = 10,              # on the tin
        max_iters = 7,           # on the tin
                # ?? No Init_State ?? 
        curve=True,                     # Makes 3rd return into plottable curve
        random_state = seed,            # Random seed here
    )

    rhc_fitness_scores.append(rhc_best_fitness)
    sa_fitness_scores.append(sa_best_fitness)
    ga_fitness_scores.append(ga_best_fitness)

    # plt.plot(rhc_curve, label="RHC")
    # plt.plot(sa_curve, label="SA")
    # plt.plot(ga_curve, label="GA")
    # plt.show()
# # Use Runner as a method for finding hyperparameters (gridsearch)
# rhc = mlrose.RHCRunner(
#     problem=problem,
#     experiment_name="rhc_4peak",
#     seed=seed,
#     iteration_list = 2 ** np.arange(11),
#     restart_list=[25,75,100],
#     max_attempts=5000,              # 24 mins
# )
# df_run_stats, df_run_curves = rhc.run()
# plt.plot(df_run_stats, label="stats")

# Plot
plt.plot(problem_range, rhc_fitness_scores, label="RHC")
plt.plot(problem_range, sa_fitness_scores, label="SA")
plt.plot(problem_range, ga_fitness_scores, label="GA")
plt.legend(loc='best')
plt.show()

# Score
# print(fitness.evaluate(best_state)) # same as print(best_fitness)  ???????