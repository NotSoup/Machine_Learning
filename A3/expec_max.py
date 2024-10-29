import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


data_dir = "../data/"

dataset_name = "ObesityDataSet_raw_and_data_sinthetic"
df = pd.read_csv(data_dir + dataset_name + ".csv")

data_encoded = df.copy()

label_encoders = {}
for column in data_encoded.select_dtypes(include='object').columns:
    label_encoders[column] = LabelEncoder()
    data_encoded[column] = label_encoders[column].fit_transform(data_encoded[column])

# Initialize and fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=90210)  # Set to 3 clusters arbitrarily
gmm.fit(data_encoded)

# Predict cluster labels
cluster_labels = gmm.predict(data_encoded)

# Calculate the silhouette score
silhouette_avg = silhouette_score(data_encoded, cluster_labels)
print(silhouette_avg)

# Add cluster labels to the data and display the results
data_encoded['Cluster'] = cluster_labels
print(data_encoded)