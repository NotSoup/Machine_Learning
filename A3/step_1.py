import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt


def run_step_1():

    data_dir = "./data/"

    # dataset_name = "ObesityDataSet_raw_and_data_sinthetic"
    # name = "Obesity"
    # letter = "o"

    dataset_name = "Rice_Cammeo_Osmancik"
    name = "Rice"
    letter = "r"

    df = pd.read_csv(data_dir + dataset_name + ".csv")
    data_encoded = df.copy()

    label_encoders = {}
    for column in data_encoded.select_dtypes(include='object').columns:
        label_encoders[column] = LabelEncoder()
        data_encoded[column] = label_encoders[column].fit_transform(data_encoded[column])

    scaler = StandardScaler()
    data_encoded = pd.DataFrame(scaler.fit_transform(data_encoded), columns=data_encoded.columns)


    ###############################################################################

    #
    #                               Expectation Maximization
    #
    silhouette_scores = []
    bic_scores = []

    k_sweep = range(2, 50)

    # for k in k_sweep:

    #     gmm = GaussianMixture(n_components=k, random_state=90210)
    #     gmm.fit(data_encoded)
    #     cluster_labels = gmm.predict(data_encoded)

    #     silhouette_scores.append(silhouette_score(data_encoded, cluster_labels))
    #     bic_scores.append(gmm.bic(data_encoded))

    #     data_encoded['Cluster'] = cluster_labels

    # print(k_sweep[np.argmin(bic_scores)])
    # print(k_sweep[np.argmin(silhouette_scores)])

    # plt.figure(figsize=(7,6)) # or (7,6)
    # plt.title(f'Component Sweep of {name} Dataset')
    # plt.plot(k_sweep, bic_scores)
    # plt.xlabel('# of Components')
    # plt.ylabel("Bayesian Information Criterion (BIC)")
    # plt.grid()
    # plt.savefig(f"./plots/EM_bic_{letter}.png", dpi=300)
    # # plt.show()

    # plt.clf()
    # plt.figure(figsize=(7, 6)) # or (7,6)
    # plt.title(f'Component Sweep of {name} Dataset')
    # plt.plot(k_sweep, silhouette_scores)
    # plt.xlabel('# of Components')
    # plt.ylabel("Silhouette Score")
    # plt.grid()
    # plt.savefig(f"./plots/EM_sil_{letter}.png", dpi=300)

    ###############################################################################

    # #
    # #                               K-Means Clustering
    # #
    inertia_scores = []

    for k in k_sweep:

        kmeans = KMeans(n_clusters=k, random_state=90210)
        kmeans.fit(data_encoded)
        cluster_labels = kmeans.predict(data_encoded)

        silhouette_scores.append(silhouette_score(data_encoded, cluster_labels))
        inertia_scores.append(kmeans.inertia_)

        data_encoded['Cluster'] = cluster_labels

    print(k_sweep[np.argmin(inertia_scores)])
    print(k_sweep[np.argmin(silhouette_scores)])

    plt.figure(figsize=(7,6)) # or (7,6)
    plt.title(f'Cluster Sweep of {name} Dataset')
    plt.plot(k_sweep, inertia_scores)
    plt.xlabel('# of Components')
    plt.ylabel("Inertia")
    plt.grid()
    plt.savefig(f"./plots/KM_inert_{letter}.png", dpi=300)
    # plt.show()

    plt.clf()
    plt.figure(figsize=(7, 6)) # or (7,6)
    plt.title(f'Cluster Sweep of {name} Dataset')
    plt.plot(k_sweep, silhouette_scores)
    plt.xlabel('# of Components')
    plt.ylabel("Silhouette Score")
    plt.grid()
    plt.savefig(f"./plots/KM_sil_{letter}.png", dpi=300)