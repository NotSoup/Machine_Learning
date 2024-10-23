import mlrose_ky as mlrose
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from timeit import default_timer as timer


# seeds = [42, 69, 408, 420, 300, 12345, 54321, 90210, 101010, 343434]
seeds = [90210]

# 1) Define fitness function
fitness = mlrose.FourPeaks(t_pct=0.1)

# edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
# fitness = mlrose.MaxKColor(edges)

# Graph with x-axis as 'Problem Size'
rhc_df = pd.DataFrame()
sa_df = pd.DataFrame()
ga_df = pd.DataFrame()
mc_df = pd.DataFrame()

problem_range = range(2, 200, 10)

start = timer()

for seed in seeds:

    rhc_fitness_scores = []
    sa_fitness_scores = []
    ga_fitness_scores = []
    mc_fitness_scores = []

    for size in problem_range:

        ############# 2) Define optimization problem #############
        problem = mlrose.DiscreteOpt(
            length=size,                    # Length of state vector
            fitness_fn=fitness,             # Fitness function
            max_val=5                       # Possible different states
        )

        # Init a state [random seed here]
        np.random.seed(seed)
        init_state = np.random.randint(2, size=size)

        ############# 3) Run optimization algorithms #############
        # (Restart-Random Hill Climb)
        rhc_best_state, rhc_best_fitness, rhc_curve = mlrose.random_hill_climb(
            problem,                        # Made in step 2
            max_attempts = 200,              # on the tin
            max_iters = np.inf,             # on the tin
            restarts=3,                     # on the tin
            init_state = init_state,        # made this above
            curve=True,                     # Makes 3rd return into plottable curve
            random_state = seed,            # Random seed here
        )

        # (Simulated Annealing)
        sa_best_state, sa_best_fitness, sa_curve = mlrose.simulated_annealing(
            problem,                        # Made in step 2
            schedule = mlrose.ExpDecay(),  # Decay scheduler
            max_attempts = 200,              # on the tin
            max_iters = np.inf,             # on the tin
            init_state = init_state,        # made this above
            curve=True,                     # Makes 3rd return into plottable curve
            random_state = seed,            # Random seed here
        )

        # # (Genetic Algorithm)
        # ga_best_state, ga_best_fitness, ga_curve = mlrose.genetic_alg(
        #     problem,                        # Made in step 2
        #     pop_size = 200,                 #
        #     pop_breed_percent = 0.75,       #
        #     elite_dreg_ratio = 0.99,        #
        #     minimum_elites = 0,             #
        #     minimum_dregs = 0,              #
        #     mutation_prob = 0.1,            #
        #     max_attempts = 200,              # on the tin
        #     max_iters = np.inf,             # on the tin
        #             # ?? No Init_State ?? 
        #     curve=True,                     # Makes 3rd return into plottable curve
        #     random_state = seed,            # Random seed here
        # )

        # # (MIMIC Algorithm)
        # mc_best_state, mc_best_fitness, mc_curve = mlrose.mimic(
        #     problem,                        # Made in step 2
        #     pop_size = 200,                 #
        #     keep_pct = 0.75,       #
        #     noise = 0.1,            #
        #     max_attempts = 200,              # on the tin
        #     max_iters = np.inf,             # on the tin
        #             # ?? No Init_State ?? 
        #     # curve=True,                     # Makes 3rd return into plottable curve
        #     random_state = seed,            # Random seed here
        # )

        rhc_fitness_scores.append(rhc_best_fitness)
        sa_fitness_scores.append(sa_best_fitness)
        # ga_fitness_scores.append(ga_best_fitness)
        # mc_fitness_scores.append(mc_best_fitness)

        rhc_fitness_scores.append(rhc_best_fitness)
        sa_fitness_scores.append(sa_best_fitness)
        # ga_fitness_scores.append(ga_best_fitness)
        # mc_fitness_scores.append(mc_best_fitness)

    rhc_df = pd.concat((rhc_df, pd.DataFrame(rhc_fitness_scores)), axis=1)
    sa_df = pd.concat((sa_df, pd.DataFrame(sa_fitness_scores)), axis=1)
    ga_df = pd.concat((ga_df, pd.DataFrame(ga_fitness_scores)), axis=1)
    # mc_df = pd.concat((mc_df, pd.DataFrame(mc_fitness_scores)), axis=1)

print(f"Runtime: {timer() - start}")

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

# Prep data for plotting
rhc_seed_avg = rhc_df.mean(axis=1)
sa_seed_avg = sa_df.mean(axis=1)
ga_seed_avg = ga_df.mean(axis=1)
# mc_seed_avg = mc_df.mean(axis=1)

rhc_std = rhc_df.std(axis=1)
sa_std = sa_df.std(axis=1)
ga_std = ga_df.std(axis=1)
# mc_std = mc_df.std(axis=1)

# Plot
# plt.plot(problem_range, rhc_fitness_scores, label="RHC")
plt.plot(problem_range, rhc_seed_avg, label="RHC")
plt.fill_between(problem_range, rhc_seed_avg-rhc_std, rhc_seed_avg+rhc_std, alpha=0.3)
# plt.plot(problem_range, sa_fitness_scores, label="SA")
plt.plot(problem_range, sa_seed_avg, label="SA")
plt.fill_between(problem_range, sa_seed_avg-sa_std, sa_seed_avg+sa_std, alpha=0.3)
# plt.plot(problem_range, ga_fitness_scores, label="GA")
# plt.plot(problem_range, ga_seed_avg, label="GA")
# plt.fill_between(problem_range, ga_seed_avg-ga_std, ga_seed_avg+ga_std, alpha=0.3)
plt.title("Four-Peaks: Fitness vs Problem Size")
plt.xlabel('Problem Size')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.grid()
# plt.show()
plt.savefig("fitness-4-peaks-init.png")

# Score
# print(fitness.evaluate(best_state)) # same as print(best_fitness)  ???????