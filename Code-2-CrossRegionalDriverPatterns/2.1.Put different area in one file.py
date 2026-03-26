import sys
import csv
from contextlib import contextmanager
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
import random
import copy
import matplotlib.pyplot as plt
import os
import shutil
import warnings

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

metrics = ["3-4", "4-5", "5-6", "6-7", "7-8"]

def Process(fileName):
    print(f"{fileName}")
    dfs = {}

    for metric in metrics:
        filePath = (f"..\\..\\Experiment-GSWE-resultAnalysis5.20\\2.VerifyCodeValueAndRemoveNAN\\{fileName}-1-{metric}_VerifyCodeValue.csv")
        dfs[metric] = pd.read_csv(filePath)
        dfs[metric].dropna(how = "all", inplace = True)
        dfs[metric]['AreaType'] = metric
        dfs[metric]['AreaTypeNumber'] = sum(map(int, metric.split("-"))) / 2

    combined_df = pd.concat(dfs.values(), ignore_index=True)

    print(f"Number of records before removing invalid Climate value: {len(combined_df)}")
    combined_df = combined_df[combined_df['Climate-value'] != 0]

    print(f"Number of records before removing invalid EcologicalZones value: {len(combined_df)}")
    combined_df = combined_df[combined_df['EcologicalZones-value'] != 901]
    combined_df = combined_df[combined_df['EcologicalZones-value'] != 0]

    print(f"Number of records before removing invalid SoilZones value: {len(combined_df)}")
    combined_df = combined_df[combined_df['SoilZones-value'] != 0]

    print(f"Number of records before removing invalid Landform value: {len(combined_df)}")
    combined_df = combined_df[combined_df['Landform-value'] != 0]

    print(f"Number of records before removing invalid BaselineWaterStress value: {len(combined_df)}")
    combined_df = combined_df[combined_df['BaselineWaterStress-value'] != 0]

    print(f"Number of records before removing invalid SoilMoisture value: {len(combined_df)}")
    combined_df = combined_df[combined_df['SoilMoisture-value'] != 0]

    print(f"Number of records before removing invalid Aridity value: {len(combined_df)}")
    combined_df = combined_df[combined_df['Aridity-value'] != 0]

    print(f"Number of records before removing invalid TemperatureBaseNumber value: {len(combined_df)}")
    combined_df = combined_df[combined_df['TemperatureBaseNumber-value'] != 0]

    print(f"Number of records before removing invalid TemperatureVariation value: {len(combined_df)}")
    combined_df = combined_df[combined_df['TemperatureVariation-value'] != 0]

    print(f"Number of records before removing invalid PrecipitationBaseNumber value: {len(combined_df)}")
    combined_df = combined_df[combined_df['PrecipitationBaseNumber-value'] != 0]

    print(f"Number of records before removing invalid PrecipitationVariation value: {len(combined_df)}")
    combined_df = combined_df[combined_df['PrecipitationVariation-value'] != 0]

    print(f"Number of records before removing invalid SDGRegions value: {len(combined_df)}")
    combined_df = combined_df[combined_df['SDGRegions-value'] != 0]

    print(f"Finally number of records: {len(combined_df)}")
    combined_df.to_csv(f"{fileName}-3to8InOneFile.csv", index = False)

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

with TeeOutput('2.ThenPutDifferentAreaInOneFileAndRemoveInvalidValuesLog.txt'):
    Process("np")
    Process("areamn")
    Process("aggregation")
    Process("connectivity")
