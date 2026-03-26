import sys
import csv
from contextlib import contextmanager
import shap
import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.ensemble import IsolationForest
import random
import copy
import matplotlib.pyplot as plt
import os
import re
import shutil
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

target = "slope1"
totalCount = 10

metrics = ["3-4", "4-5", "5-6", "6-7", "7-8"]

mapping = {
    1: '1Sub-SaharanAfrica', 2: '2NorthernAfrica', 3: '3WesternAsia', 4: '4CentralAsia', 5: '5SouthernAsia',
    6: '6EasternAsia', 7: '7South-easternAsia', 8: '8NorthernAsia', 9: '9Caribbean', 10: '10CentralAmerica',
    11: '11SouthAmerica', 12: '12AustraliaAndNewZealand', 13: '13OceaniaExcludingAUSAndNZ', 14: '14ContiguousUnitedStates', 15: '15Alaska' ,
    16: '16CanadaProvinces', 17: '17CanadaTerritories', 18: '18Greenland', 19: '19EasternEurope', 20: '20NorthernEurope',
    21: '21SouthernEurope', 22: '22WesternEurope'
}

def FirstProcess(fileName):
    Process(fileName, f"0World-{fileName}-all")
    for metric in metrics:
        Process(fileName, f"0World-{fileName}-{metric}")
    print()

    for i in range(1, 23):
        Process(fileName, f"{mapping[i]}-{fileName}-all")
        for metric in metrics:
            Process(fileName, f"{mapping[i]}-{fileName}-{metric}")
        print()

def Process(fileName, fileNameFull):
    print(fileNameFull)
    fileIn = f"..\\3.CheckOutlierForNextStep\\{fileName}\\{fileNameFull}_remained.csv"

    if not os.path.exists(fileIn):
        print(f"File does not exist. {fileIn}")
        return

    df = pd.read_csv(fileIn)
    df.dropna(how="all", inplace=True)

    if len(df) < 5:
        print(f"len(df) < 5. Not enough data in {fileIn}")
        return

    y = df.loc[:, df.columns == target].copy().values.ravel()
    x = df.loc[:, df.columns != target].copy()
    for v in ["SOURCE_ID", "COType", "FID", "fileName", "X", "Y", "slope", "DamConstruction-value", "AreaType"]:
        x = x.loc[:, x.columns != v].copy()

    feature_importances_result = pd.DataFrame({'Feature': x.columns,
                                               'Importance': 0.0,
                                               'Total count of records': 0,
                                               'Count of positive records': 0,
                                               'Mean of positive records': 0.0,
                                               'Rank for Mean of positive records': 0,
                                               'Median of positive records': 0.0,
                                               'Rank for Median of positive records': 0,
                                               'Count of negetive records': 0,
                                               'Mean of negetive records': 0.0,
                                               'Rank for Mean of negetive records': 0,
                                               'Median of negetive records': 0.0,
                                               'Rank for Median of negetive records': 0})
    avg_oob_score_= 0.0
    avg_mse_= 0.0
    avg_r2_= 0.0
    number_of_records = len(y)

    model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=None)

    for i in range(totalCount):
        print(i)
        model.fit(x, y)

        print(f'Out-of-Bag Score: {model.oob_score_}')
        y_pred = model.predict(x)
        mse = mean_squared_error(y, y_pred)
        print(f'Mean Squared Error: {mse}')
        r2 = r2_score(y, y_pred)
        print(f'R-squared: {r2}')
        avg_oob_score_ += model.oob_score_
        avg_mse_ += mse
        avg_r2_ += r2

        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(x)
        feature_importances_result['Importance'] += np.abs(shap_values).mean(axis=0)
        feature_importances_result['Total count of records'] += shap_values.shape[0]

        positive_values = np.where(shap_values >= 0, shap_values, 0)
        negative_values = np.where(shap_values < 0, shap_values, 0)
        feature_importances_result['Count of positive records'] += np.sum(positive_values != 0, axis=0)
        feature_importances_result['Mean of positive records'] += np.mean(positive_values, axis=0)
        feature_importances_result['Median of positive records'] += np.median(positive_values, axis=0)
        feature_importances_result['Count of negetive records'] += np.sum(negative_values != 0, axis=0)
        feature_importances_result['Mean of negetive records'] += np.mean(negative_values, axis=0)
        feature_importances_result['Median of negetive records'] += np.median(negative_values, axis=0)

    feature_importances_result['Importance'] /= totalCount
    feature_importances_result['Total count of records'] /= totalCount
    feature_importances_result['Count of positive records'] /= totalCount
    feature_importances_result['Mean of positive records'] /= totalCount
    feature_importances_result['Median of positive records'] /= totalCount
    feature_importances_result['Count of negetive records'] /= totalCount
    feature_importances_result['Mean of negetive records'] /= totalCount
    feature_importances_result['Median of negetive records'] /= totalCount

    for column in ['Mean of positive records', 'Median of positive records', 'Mean of negetive records',
                   'Median of negetive records']:
        rank_column = 'Rank for ' + column
        ascending = False if 'positive' in column else True
        feature_importances_result[rank_column] = feature_importances_result[column].rank(ascending=ascending,
                                                                                          method='min').astype(int)

    feature_importances_result = feature_importances_result.sort_values(by='Importance', ascending=False)
    print(feature_importances_result)
    feature_importances_result.to_csv(f"{fileName}\\{fileNameFull}_FeatureImportance.csv", index=True)

    avg_oob_score_ = avg_oob_score_ / totalCount
    avg_mse_ = avg_mse_ / totalCount
    avg_r2_ = avg_r2_ / totalCount
    performance = ("number_of_records: " + str(number_of_records) + "\navg_oob_score_: " + str(
        avg_oob_score_) + "\navg_mse_: " + str(avg_mse_) + "\navg_r2_: " + str(avg_r2_))
    print("performance: " + performance)
    with open(f"{fileName}\\{fileNameFull}_performance.txt", 'w') as file:
        file.write(performance)

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

with TeeOutput('4.AnalyseOverallFeatureImportanceLog_np.txt'):
    FirstProcess("np")
with TeeOutput('4.AnalyseOverallFeatureImportanceLog_areamn.txt'):
    FirstProcess("areamn")
with TeeOutput('4.AnalyseOverallFeatureImportanceLog_aggregation.txt'):
    FirstProcess("aggregation")
with TeeOutput('4.AnalyseOverallFeatureImportanceLog_connectivity.txt'):
    FirstProcess("connectivity")