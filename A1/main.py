from KNN import KNN
from Boost_DT import BoostDT
from SVM import SVM
from MLP import NN
import pandas as pd
import statistics
from timeit import default_timer as timer


start = timer()
data_dir = "./data/"

# # Std Dev: 5.4      [46, 58, 56, 54]
# # Runtime: 3.6
# # Instances: 2044
# dataset_name = data_dir + "/heart/processed.cleveland.data"
# df = pd.read_csv(dataset_name)
# df = df.drop([87, 166, 192, 287, 302, 266])    # 'thal' & 'ca' with ? values
# X = df.drop(["num"], axis=1)
# y = df["num"]

# # Std Dev: 9.3    [61, 76, 61, 77]
# # Runtime: 70
# # Instances: 4425
# # Features: 36
# dataset_name = data_dir + "student_dropout_success.csv"
# df = pd.read_csv(dataset_name, sep=";")
# X = df.drop(["Target"], axis=1)
# y = df["Target"]

# # Std Dev: 9.3    [24, 24, 27, 26]
# # Runtime: 70
# # Instances: 4425
# # Features: 36
# dataset_name = data_dir + "abalone.data"
# df = pd.read_csv(dataset_name)
# X = df.drop(["Rings"], axis=1)
# X = pd.get_dummies(X)
# y = df["Rings"]

# # Std Dev: 
# # Runtime: 70
# # Instances: 4425
# # Features: 36
# dataset_name = data_dir + "housing.csv"
# df = pd.read_csv(dataset_name)
# X = df.drop(["Rings"], axis=1)
# X = pd.get_dummies(X)
# y = df["Rings"]

# # Std Dev: 6.8
# # Runtime: 463
# # Instances: 4600
# # Features: 58
# dataset_name = data_dir + "spambase.data"
# df = pd.read_csv(dataset_name)
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]

# # Std Dev: 8.1      [58, 56, 42, 61]
# # Runtime: 15
# # Instances: 1483
# # Features: 10
# dataset_name = data_dir + "yeast.data"
# df = pd.read_csv(dataset_name, delim_whitespace=True)
# X = df.iloc[:, 1:-1]
# y = df.iloc[:, -1]


######################## NUMERICAL / BINARY Y ############################
# Std Dev: 9.5     [88, 94, 73, 93]
# Runtime: 10
# Instances: 3810
# Features: 7
dataset_name = data_dir + "Rice_Cammeo_Osmancik.arff"
df = pd.read_csv(dataset_name)
X = df.drop(["Class"], axis=1)
y = df["Class"]

# https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
####################################################################

# # Std Dev: 5.9
# # Runtime: 16
# # Instances: 2111
# dataset_name = data_dir + "ObesityDataSet_raw_and_data_sinthetic.csv"
# df = pd.read_csv(dataset_name)
# X = df.drop(["NObeyesdad"], axis=1)
# X = pd.get_dummies(X)
# y = df["NObeyesdad"]

# # Std Dev: 
# # Runtime: 
# # Instances: 
# dataset_name = data_dir + "optdigits.tra"
# df = pd.read_csv(dataset_name)
# dataset_name = data_dir + "optdigits.tes"
# df2 = pd.read_csv(dataset_name)
# df.columns = [None] * len(df.columns)
# df2.columns = [None] * len(df2.columns)
# df = pd.concat([df, df2])
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]

# # Std Dev: 4.6
# # Runtime: 27
# # Instances: 2310
# # Features: 19
# dataset_name = data_dir + "segmentation.data"
# df = pd.read_csv(dataset_name)
# y = df.index
# X = df.reset_index(drop=True)

# # Std Dev: 4.6
# # Runtime: 27
# # Instances: 2310
# # Features: 19
# dataset_name = data_dir + "agaricus-lepiota.data"
# df = pd.read_csv(dataset_name)
# X = df.iloc[:, 1:]
# y = df.iloc[:, 0]
# X = pd.get_dummies(X)

# # Std Dev: 4.6
# # Runtime: 27
# # Instances: 2310
# # Features: 19
# dataset_name = data_dir + "sat.trn"
# df = pd.read_csv(dataset_name, delimiter=" ")
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]

# from datasets import load_dataset

# df = load_dataset("mstz/heart_failure", split="train")
# df = df.to_pandas()
# X = df.drop(["is_dead"], axis=1)
# y = df["is_dead"]

# # Std Dev: 4.3
# # Runtime: 5.2
# # Instances: 1729
# dataset_name = data_dir + "car.data"
# df = pd.read_csv(dataset_name)
# X = df.drop(["class"], axis=1)
# y = df["class"]
# X = pd.get_dummies(X)

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