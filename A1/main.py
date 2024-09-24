from KNN import KNN
from Boost_DT import BoostDT
from SVM import SVM
from MLP import NN
import pandas as pd
import statistics
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


start = timer()
data_dir = "..\data\\"

######################## OBESITY DATASET ##########################
dataset_name = "ObesityDataSet_raw_and_data_sinthetic"
df = pd.read_csv(data_dir + dataset_name + ".csv")

df_dumm = pd.get_dummies(
    df, 
    columns=['Gender', 
             'family_history_with_overweight', 
             'FAVC', 
             'CAEC', 
             'SMOKE', 
             'SCC', 
             'CALC', 
             'MTRANS' 
    ]
)

X = df_dumm.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

######################## RICE DATASET ##########################
# dataset_name = "Rice_Cammeo_Osmancik"
# df = pd.read_csv(data_dir + dataset_name + ".csv")
# X = df.drop(["Class"], axis=1)
# y = df["Class"]

# # Convert rice class strings to numeric
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)

######################## PRE-PROCESS ##########################
# Metrics (precision, recall, F1 score)
# metric = 'accuracy'
# metric = 'f1'
metric = 'recall_macro'

# # Normalize/scale features (for stable gradient descent, etc)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Split dataset for test/train
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=90210,
)

fig, ax = plt.subplots(2, 4, figsize=(23,14))

######################## LEARN ##########################
KNN(X_train, X_test, y_train, y_test, ax, metric)
SVM(X_train, X_test, y_train, y_test, ax, metric)
NN(X_train, X_test, y_train, y_test, ax, metric)
# BoostDT(X_train, X_test, y_train, y_test, ax, metric)

print(f"Runtime: {timer() - start}")
# plt.savefig(f'./plots/full_suite_{dataset_name}.png', dpi=300)