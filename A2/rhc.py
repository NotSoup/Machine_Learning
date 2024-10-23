import mlrose_ky as mlrose
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from timeit import default_timer as timer


seeds = [42, 69, 408, 420, 300, 12345, 54321, 90210, 101010, 343434]

# 1) Define fitness function
fitness = mlrose.FourPeaks(t_pct=0.1)

# Graph with x-axis as 'Problem Size'
rhc_df = pd.DataFrame()
time_df = pd.DataFrame()

size = 128

problem = mlrose.DiscreteOpt(
    length=size,                    # Length of state vector
    fitness_fn=fitness,             # Fitness function
    max_val=2                       # zaraPossible different states
)

atmpt_sweep = np.arange(1, 200, 10)

for seed in seeds:

    # Init a state [random seed here]
    np.random.seed(seed)
    init_state = np.random.randint(2, size=size)

    rhc_fitness_scores = []
    rhc_fitness_scores2 = []
    rhc_time = []

    for max_atmpt in atmpt_sweep:

        start = timer()

        rhc_best_state, rhc_best_fitness, rhc_curve = mlrose.random_hill_climb(
            problem,                        # Made in step 2
            max_attempts = max_atmpt,           # on the tin
            max_iters = np.inf,             # on the tin
            restarts=3,                     # on the tin
            init_state = init_state,        # made this above
            curve=False,                     # Makes 3rd return into plottable curve
            random_state = seed,            # Random seed here
        )

        print(f"Runtime: {timer() - start}")
        clock_time = timer() - start

        rhc_time.append(clock_time)
        rhc_fitness_scores.append(rhc_best_fitness)
    
    rhc_df = pd.concat((rhc_df, pd.DataFrame(rhc_fitness_scores)), axis=1)
    time_df = pd.concat((time_df, pd.DataFrame(rhc_time)), axis=1)

# Prep data for plotting
rhc_seed_avg = rhc_df.mean(axis=1)
rhc_time_seed_avg = time_df.mean(axis=1)

rhc_std = rhc_df.std(axis=1)
rhc_time_seed_std = time_df.mean(axis=1)

# Plot
plt.plot(atmpt_sweep, rhc_seed_avg, label="RHC")
plt.fill_between(atmpt_sweep, rhc_seed_avg-rhc_std, rhc_seed_avg+rhc_std, alpha=0.3)
plt.title("RHC Tuning (Four-Peaks): Fitness vs Max Attempts")
plt.xlabel('Max Attempts')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.grid()
# plt.show()
plt.savefig("./A2/plots/4p-RCH-tuning-atmp.png")

######################################################################################################

# 1) Define fitness function
fitness = mlrose.FourPeaks(t_pct=0.1)

# Graph with x-axis as 'Problem Size'
rhc_df = pd.DataFrame()
time_df = pd.DataFrame()

size = 128

restart_sweep = [0, 5, 10, 20, 50, 100, 200]

for seed in seeds:

    # Init a state [random seed here]
    np.random.seed(seed)
    init_state = np.random.randint(2, size=size)

    rhc_fitness_scores = []
    rhc_time = []

    for num_restarts in restart_sweep:

        start = timer()

        rhc_best_state, rhc_best_fitness, rhc_curve = mlrose.random_hill_climb(
            problem,                        # Made in step 2
            max_attempts = 200,           # on the tin
            max_iters = np.inf,             # on the tin
            restarts=num_restarts,                     # on the tin
            init_state = init_state,        # made this above
            curve=False,                     # Makes 3rd return into plottable curve
            random_state = seed,            # Random seed here
        )

        print(f"Runtime: {timer() - start}")
        # clock_time = timer() - start

        # rhc_time.append(clock_time)
        rhc_fitness_scores.append(rhc_best_fitness)
    

    rhc_df = pd.concat((rhc_df, pd.DataFrame(rhc_fitness_scores)), axis=1)
    # time_df = pd.concat((time_df, pd.DataFrame(rhc_time)), axis=1)

# Prep data for plotting
rhc_seed_avg = rhc_df.mean(axis=1)
# rhc_time_seed_avg = time_df.mean(axis=1)

rhc_std = rhc_df.std(axis=1)
# rhc_time_seed_std = time_df.mean(axis=1)

plt.clf()
plt.plot(restart_sweep, rhc_seed_avg, label="RHC")
plt.fill_between(restart_sweep, rhc_seed_avg-rhc_std, rhc_seed_avg+rhc_std, alpha=0.3)
plt.title("RHC Tuning (Four-Peaks): Fitness vs Restarts")
plt.xlabel('Restarts')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.grid()
# plt.show()
plt.savefig("./A2/plots/4p-RCH-tuning-rstrt.png")

######################################################################################################

# Graph with x-axis as 'Problem Size'
rhc_df = pd.DataFrame()
time_df = pd.DataFrame()
iter_df = pd.DataFrame()

# size = 128

# restart_sweep = [0, 5, 10, 20, 50, 100, 200]

problem_range = range(2, 200, 10)

for seed in seeds:

    # Init a state [random seed here]
    np.random.seed(seed)
    init_state = np.random.randint(2, size=size)

    rhc_fitness_scores = []
    rhc_time = []
    rhc_iterations = []

    for size in problem_range:

        problem = mlrose.DiscreteOpt(
            length=size,                    # Length of state vector
            fitness_fn=fitness,             # Fitness function
            max_val=2                       # zaraPossible different states
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

rhc_seed_avg = rhc_df.mean(axis=1)
iter_seed_avg = iter_df.mean(axis=1)
rhc_time_seed_avg = time_df.mean(axis=1)

rhc_std = rhc_df.std(axis=1)
iter_std = iter_df.std(axis=1)
rhc_time_seed_std = time_df.std(axis=1)

plt.clf()
plt.plot(problem_range, rhc_seed_avg, label="RHC")
plt.fill_between(problem_range, rhc_seed_avg-rhc_std, rhc_seed_avg+rhc_std, alpha=0.3)
plt.title("RHC (Four-Peaks): Fitness vs Iterations")
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.grid()
# plt.show()
plt.savefig("./A2/plots/4p-RCH-FitVsIter.png")

plt.plot(problem_range, iter_seed_avg, label="RHC")
plt.fill_between(problem_range, iter_seed_avg-iter_std, iter_seed_avg+iter_std, alpha=0.3)
plt.title("Four-Peaks: Fitness vs Problem Size")
plt.xlabel('Problem Size')
plt.ylabel('Fitness')
plt.legend(loc='best')
plt.grid()
# plt.show()
plt.savefig("./A2/plots/4p-RCH-FitVsSize.png")

plt.plot(problem_range, rhc_time_seed_avg, label="RHC")
plt.fill_between(problem_range, rhc_time_seed_avg-rhc_time_seed_std, rhc_time_seed_avg+rhc_time_seed_std, alpha=0.3)
plt.title("Four-Peaks: Fitness vs Time")
plt.xlabel('Problem Size')
plt.ylabel('Time')
plt.legend(loc='best')
plt.grid()
# plt.show()
plt.savefig("./A2/plots/4p-RCH-FitVsTime.png")