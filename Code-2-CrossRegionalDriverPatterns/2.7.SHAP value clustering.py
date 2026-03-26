import sys
import csv
from contextlib import contextmanager
import shap
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import shutil
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
import time

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

features = ['Elevation', 'Climate', 'EcologicalZones', 'SoilZones', 'Landform',
            'PopulationBaseNumber', 'PopulationVariation', 'GDPDensityBaseNumber', 'GDPDensityVariation', 'NighttimeLightBaseNumber',
            'NighttimeLightVariation', 'BaselineWaterStress', 'SoilMoisture', 'Aridity', 'TemperatureBaseNumber',
            'TemperatureVariation', 'PrecipitationBaseNumber', 'PrecipitationVariation', 'GlaciersMeltingElevationChange', 'GlaciersMeltingMassChange',
            'SDGRegions', 'AreaTypeNumber']

def Process(fileName):
    df = pd.read_csv(f"..\\10.SHAPValueSeparatePosNegCompMean\\{fileName}.csv")
    print(f"Number of {fileName}: {len(df)}")

    silhouette_scores = [np.nan] * len(df)
    cluster_results = [[np.nan] * len(features) for _ in range(len(df))]

    for index, row in df.iterrows():
        print(f"{index}")
        X = row[features].values.reshape(-1, 1)
        non_zero_indices = np.nonzero(X)[0]
        X_non_zero = X[non_zero_indices]
        if len(X_non_zero) == 0:
            print("len(X_non_zero) == 0")
            continue

        sorted_indices = np.argsort(X_non_zero.flatten())
        X_non_zero_sorted = X_non_zero[sorted_indices]

        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
        kmeans.fit(X_non_zero_sorted)
        score = silhouette_score(X_non_zero_sorted, kmeans.labels_)
        print(f"number of clusters: {N_CLUSTERS}, silhouette_score: {score}")
        silhouette_scores[index] = score

        labels = kmeans.labels_
        sorted_labels = np.empty_like(labels)
        sorted_labels[sorted_indices] = labels

        cluster_means = {cluster: np.mean(X[non_zero_indices][sorted_labels == cluster]) for cluster in range(N_CLUSTERS)}
        sorted_clusters = sorted(cluster_means, key=cluster_means.get, reverse=True)
        cluster_mapping = {old_cluster: new_cluster + 1 for new_cluster, old_cluster in enumerate(sorted_clusters)}
        new_labels = np.full(len(features), np.nan)
        for idx in non_zero_indices:
            new_labels[idx] = cluster_mapping[sorted_labels[np.where(non_zero_indices == idx)[0][0]]]
        cluster_results[index] = new_labels

    df['SilhouetteScore'] = silhouette_scores
    for i, feature in enumerate(features):
        df[f'C-{feature}'] = [cluster_result[i] for cluster_result in cluster_results]

    print(f"Mean of all silhouette_score: {np.nanmean(silhouette_scores)}")

    df = df.loc[:, ~df.columns.str.startswith('Nor')]
    df[features] = df[features].replace(0, np.nan)

    df.to_csv(f"{fileName}-Cluster：{N_CLUSTERS}.csv", index=False)

class TeeOutput:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
    def __enter__(self):
        sys.stdout = self
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)

# N_CLUSTERS = 4

for N_CLUSTERS in range(2, 7):
    with TeeOutput(f'11.SHAPValueAllAndClusteringLog-np-shap-POS-Cluster：{N_CLUSTERS}.txt'):
        Process('np-shap-POS')
    with TeeOutput(f'11.SHAPValueAllAndClusteringLog-np-shap-NEG-Cluster：{N_CLUSTERS}.txt'):
        Process('np-shap-NEG')
    with TeeOutput(f'11.SHAPValueAllAndClusteringLog-areamn-shap-POS-Cluster：{N_CLUSTERS}.txt'):
        Process('areamn-shap-POS')
    with TeeOutput(f'11.SHAPValueAllAndClusteringLog-areamn-shap-NEG-Cluster：{N_CLUSTERS}.txt'):
        Process('areamn-shap-NEG')
    with TeeOutput(f'11.SHAPValueAllAndClusteringLog-aggregation-shap-POS-Cluster：{N_CLUSTERS}.txt'):
        Process('aggregation-shap-POS')
    with TeeOutput(f'11.SHAPValueAllAndClusteringLog-aggregation-shap-NEG-Cluster：{N_CLUSTERS}.txt'):
        Process('aggregation-shap-NEG')
    with TeeOutput(f'11.SHAPValueAllAndClusteringLog-connectivity-shap-POS-Cluster：{N_CLUSTERS}.txt'):
        Process('connectivity-shap-POS')
    with TeeOutput(f'11.SHAPValueAllAndClusteringLog-connectivity-shap-NEG-Cluster：{N_CLUSTERS}.txt'):
        Process('connectivity-shap-NEG')