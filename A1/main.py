from KNN import KNN
from Boost_DT import BoostDT
from SVM import SVM
from NN import NN
import pandas as pd
import statistics
from timeit import default_timer as timer


start = timer()
data_dir = "./../data/"

# # Std Dev: 5.4
# # Runtime: 3.6
# # Instances: 2044
# dataset_name = data_dir + "data.csv"
# dt = pd.read_csv(dataset_name)
# X = dt.drop(["verification.result", "verification.time"], axis=1)
# y = dt["verification.result"]

# # Std Dev: 9.3
# # Runtime: 70
# # Instances: 4425
# dataset_name = data_dir + "student_dropout_success.csv"
# dt = pd.read_csv(dataset_name, sep=";")
# X = dt.drop(["Target"], axis=1)
# y = dt["Target"]

# # Std Dev: 1.9
# # Runtime: 4.7
# # Instances: 6287
# dataset_name = data_dir + "NHANES_age_prediction.csv"
# dt = pd.read_csv(dataset_name)
# dt = dt.drop(["SEQN"], axis=1)
# X = dt.drop(["age_group"], axis=1)
# y = dt["age_group"]

# # Std Dev: 3.8
# # Runtime: 4
# # Instances: 303(299)
# dataset_name = data_dir + "heart/processed.cleveland.data"
# dt = pd.read_csv(dataset_name)
# dt = dt.drop(["thal"], axis=1)
# dt = dt.drop([166,192,287,302])
# X = dt.drop(["num"], axis=1)
# y = dt["num"]

# # Std Dev: 2.3
# # Runtime: 749
# # Instances: 70,000
# dataset_name = data_dir + "diabetes/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
# dt = pd.read_csv(dataset_name)
# X = dt.drop(["Diabetes_binary"], axis=1)
# y = dt["Diabetes_binary"]

# Std Dev: 9.5
# Runtime: 10
# Instances: 3810
dataset_name = data_dir + "Rice_Cammeo_Osmancik.arff"
dt = pd.read_csv(dataset_name)
X = dt.drop(["Class"], axis=1)
y = dt["Class"]

#####
# Performance metrics (BESIDES ERROR): precision, recall, F1 score
#####

dataset_diff = []
dataset_diff.append(100 * KNN(X, y))
dataset_diff.append(100 * SVM(X, y))
dataset_diff.append(100 * NN(X, y))
dataset_diff.append(100 * BoostDT(X, y))

print(f"Std Dev of dataset: {statistics.stdev(dataset_diff)}")
end = timer()
print(f"Runtime: {end - start}")