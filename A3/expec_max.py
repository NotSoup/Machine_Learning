import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt


data_dir = "../data/"

dataset_name = "ObesityDataSet_raw_and_data_sinthetic"
df = pd.read_csv(data_dir + dataset_name + ".csv")
data_encoded = df.copy()

label_encoders = {}
for column in data_encoded.select_dtypes(include='object').columns:
    label_encoders[column] = LabelEncoder()
    data_encoded[column] = label_encoders[column].fit_transform(data_encoded[column])


###############################################################################
##                      Step 1
###############################################################################

#
#                               Expectation Maximization
#
silhouette_scores = []
k_sweep = range(2, 4)

for k in k_sweep:

    gmm = GaussianMixture(n_components=k, random_state=90210)
    gmm.fit(data_encoded)

    cluster_labels = gmm.predict(data_encoded)

    silhouette_avg = silhouette_score(data_encoded, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(silhouette_avg)

    data_encoded['Cluster'] = cluster_labels
    # print(data_encoded)

plt.plot(k_sweep, silhouette_scores)
plt.grid()
plt.show()

# #
# #                               K-Means Clustering
# #
# kmeans = KMeans(n_clusters=3, random_state=90210)
# kmeans.fit(data_encoded)

# kmeans_labels = kmeans.predict(data_encoded)

# data_encoded['KMeans_Cluster'] = kmeans_labels


# ###############################################################################
# ##                      Step 2
# ###############################################################################

# #
# # Randomized Projections
# #
# random_proj = GaussianRandomProjection(n_components=5, random_state=90210)
# data_projected = random_proj.fit_transform(data_encoded)


# #
# # Principle Component Analysis (PCA)
# #
pca = PCA(n_components=5, random_state=90210)
data_pca = pca.fit_transform(data_encoded)
print(pca.explained_variance_ratio_)

# #
# # Independent Component Analysis (ICA)
# #
# ica = FastICA(n_components=5, random_state=90210)
# data_ica = ica.fit_transform(data_encoded)


# ##############################################################################
# #                      Step 3
# ##############################################################################


# Randomized Projections                    > Expectation Maximization



# Principle Component Analysis (PCA)        > Expectation Maximization



# Independent Component Analysis (ICA)      > Expectation Maximization



# Randomized Projections                    > K-Means Clustering



# Principle Component Analysis (PCA)        > K-Means Clustering



# Independent Component Analysis (ICA)      > K-Means Clustering



# ##############################################################################
# #                      Step 4
# ##############################################################################

# ##############################################################################
# #                      Step 5
# ##############################################################################