import mlrose_ky as mlrose
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from timeit import default_timer as timer


seeds = [42, 69, 408, 420, 300, 12345, 54321, 90210, 101010]
fitness = mlrose.FourPeaks(t_pct=0.1)

rhc_df = pd.DataFrame()
time_df = pd.DataFrame()
iter_df = pd.DataFrame()

problem_size = 128

# restart_sweep = [0, 5, 10, 20, 50, 100, 200]

problem_range = range(2, 200, 10)

for seed in seeds:

    rhc_fitness_scores = []
    rhc_time = []
    rhc_iterations = []

    for size in problem_range:      ### <<<<<<

        np.random.seed(seed)
        init_state = np.random.randint(2, size=128)

        problem = mlrose.DiscreteOpt(
            length=128,             ### <<<<<<
            fitness_fn=fitness,
            max_val=2
        )

        start = timer()

        rhc_best_state, rhc_best_fitness, rhc_curve = mlrose.random_hill_climb(
            problem,                        # Made in step 2
            max_attempts = 200,           # on the tin
            max_iters = np.inf,             # on the tin
            restarts=5,                     # on the tin
            init_state = init_state,        # made this above
            curve=True,                     # Makes 3rd return into plottable curve
            random_state = seed,            # Random seed here
        )

        print(f"Runtime: {timer() - start}")
        clock_time = timer() - start

        rhc_time.append(clock_time)
        rhc_fitness_scores.append(rhc_best_fitness)
        rhc_iterations.append(len(rhc_curve))
    
    time_df = pd.concat((time_df, pd.DataFrame(rhc_time)), axis=1)
    rhc_df = pd.concat((rhc_df, pd.DataFrame(rhc_fitness_scores)), axis=1)
    iter_df = pd.concat((iter_df, pd.DataFrame(rhc_iterations)), axis=1)

plt.clf()
plt.plot(problem_range, rhc_df.mean(axis=1), label="RHC")
plt.fill_between(problem_range, rhc_df.min(axis=1), rhc_df.max(axis=1), alpha=0.3)
plt.title("RHC (Four-Peaks): Fitness vs Problem Size")
plt.xlabel('Problem Size')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.grid()
plt.show()

# plt.plot(problem_range, iter_seed_avg, label="RHC")
# plt.fill_between(problem_range, iter_seed_avg-iter_std, iter_seed_avg+iter_std, alpha=0.3)
# plt.title("Four-Peaks: Fitness vs Iterations")
# plt.xlabel('Iterations')
# plt.ylabel('Fitness')
# plt.legend(loc='best')
# plt.grid()
# plt.show()

# plt.plot(problem_range, rhc_time_seed_avg, label="RHC")
# plt.fill_between(problem_range, rhc_time_seed_avg-rhc_time_seed_std, rhc_time_seed_avg+rhc_time_seed_std, alpha=0.3)
# plt.title("Four-Peaks: Fitness vs Time")
# plt.xlabel('Problem Size')
# plt.ylabel('Time')
# plt.legend(loc='best')
# plt.grid()
# # plt.show()