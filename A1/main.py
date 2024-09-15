from KNN import KNN
from Boost_DT import BoostDT
from SVM import SVM
from NN import NN
import pandas as pd
import statistics
from timeit import default_timer as timer


start = timer()
data_dir = "./../data/"

# # Std Dev: 6.21
# # Runtime: 0.9
# # Instances: 2044
# dataset_name = data_dir + "data.csv"
# dt = pd.read_csv(dataset_name)
# X = dt.drop(["verification.result", "verification.time"], axis=1)
# y = dt["verification.result"]

# # Std Dev: 13.63
# # Runtime: 21
# # Instances: 4425
# dataset_name = data_dir + "student_dropout_success.csv"
# dt = pd.read_csv(dataset_name, sep=";")
# X = dt.drop(["Target"], axis=1)
# y = dt["Target"]

# Std Dev: 2.32
# Runtime: 1
# Instances: 6287
dataset_name = data_dir + "NHANES_age_prediction.csv"
dt = pd.read_csv(dataset_name)
dt = dt.drop(["SEQN"], axis=1)
X = dt.drop(["age_group"], axis=1)
y = dt["age_group"]

dataset_diff = []
dataset_diff.append(100 * KNN(X, y))
dataset_diff.append(100 * SVM(X, y))
dataset_diff.append(100 * NN(X, y))
dataset_diff.append(100 * BoostDT(X, y))

print(f"Std Dev of dataset: {statistics.stdev(dataset_diff)}")
end = timer()
print(f"Runtime: {end - start}")