import mlrose_ky as mlrose
import numpy as np
import matplotlib.pyplot as plt


# 1) Define fitness function
fitness = mlrose.FourPeaks(t_pct=0.15)
    # fitness.evaluate(state) >>>> F(x) = Reward
        # so this thing just takes a state and spits out a score. NO OPTIMIZATION IS HAPPENING HERE 

# 2) Define optimization problem
problem = mlrose.DiscreteOpt(
    length=4,                       # Length of state vector
    fitness_fn=fitness              # Fitness function
    # max_val=
)

# just init a state [random seed here]
init_state = np.array([1, 1, 0, 1])

# 3) Run optimization algorithm (Simulated Annealing)
best_state, best_fitness, _ = mlrose.simulated_annealing(
    problem,                        # Made in step 2
    # schedule = schedule,          # Decay scheduler (?)
    max_attempts = 10,              # on the tin
    max_iters = 1000,               # on the tin
    init_state = init_state,        # made this above
    random_state = 1                # Random seed here
    # curve=True                      # Makes 3rd return into plottable curve
)

# Score
print(fitness.evaluate(best_state)) # same as print(best_fitness)  ???????