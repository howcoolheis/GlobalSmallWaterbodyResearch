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

climate_mapping = {
    1: 101, 2: 102, 3: 103, 6: 201, 7: 202, 4: 211, 5: 212,
    13: 301, 14: 302, 15: 303, 8: 311, 9: 312, 10: 321,
    11: 322, 12: 323, 24: 401, 25: 402, 26: 403, 27: 404,
    16: 411, 17: 412, 18: 413, 19: 414, 20: 421, 21: 422,
    22: 423, 23: 424, 28: 501, 29: 502
}
ecological_zones_mapping = {
    11: 101, 12: 102, 13: 103, 14: 104, 15: 105, 16: 106,
    21: 201, 22: 202, 23: 203, 24: 204, 25: 206, 31: 301,
    32: 302, 33: 303, 34: 304, 35: 305, 41: 401, 42: 402,
    43: 403, 50: 501, 90: 901
}
soil_zones_mapping = {
    5: 101, 2: 201, 3: 202, 6: 203, 10: 204, 11: 205, 8: 206,
    9: 301, 7: 401, 1: 501
}
landform_mapping = {
    4: 101, 5: 102, 1: 201, 2: 202, 3: 203, 6: 301, 7: 302,
    8: 303, 9: 304, 10: 305, 11: 401, 12: 501, 13: 502, 14: 503,
    15: 504
}

def Process(fileName):
    print(fileName)
    fileIn = "..\\1.HHAndLLDataExport\\" + fileName + ".csv"
    df = pd.read_csv(fileIn)
    df.dropna(how="all", inplace=True)
    print(f"Number of records before map: {len(df)}")

    df['Climate-value'] = df['Climate-value'].map(climate_mapping)
    df['EcologicalZones-value'] = df['EcologicalZones-value'].map(ecological_zones_mapping)
    df['SoilZones-value'] = df['SoilZones-value'].map(soil_zones_mapping)
    df['Landform-value'] = df['Landform-value'].map(landform_mapping)

    df.dropna(inplace=True)

    df.to_csv(fileName + "_VerifyCodeValue.csv", index=False)
    print(f"Number of records after map: {len(df)}")

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

with TeeOutput('2.VerifyCodeValueAndRemoveNANLog.txt'):
    Process("np-3-4")
    Process("np-4-5")
    Process("np-5-6")
    Process("np-6-7")
    Process("np-7-8")

    Process("areamn-3-4")
    Process("areamn-4-5")
    Process("areamn-5-6")
    Process("areamn-6-7")
    Process("areamn-7-8")

    Process("aggregation-3-4")
    Process("aggregation-4-5")
    Process("aggregation-5-6")
    Process("aggregation-6-7")
    Process("aggregation-7-8")

    Process("connectivity-3-4")
    Process("connectivity-4-5")
    Process("connectivity-5-6")
    Process("connectivity-6-7")
    Process("connectivity-7-8")