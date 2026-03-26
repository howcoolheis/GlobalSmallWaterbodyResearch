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
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

thresholdCOR = 0.8
thresholdVIF = 10.0

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

    for i in range(1, 23):
        Process(fileName, f"{mapping[i]}-{fileName}-all")
        for metric in metrics:
            Process(fileName, f"{mapping[i]}-{fileName}-{metric}")

def Process(fileName, fileNameFull):
    print(fileNameFull)
    fileIn = f"..\\2.ThenSeparateIntoStudyRegions\\{fileName}\\{fileNameFull}.csv"
    df = pd.read_csv(fileIn)
    df.dropna(how="all", inplace=True)

    if len(df) < 2:
        print(f"len(df) < 2. Not enough data in {fileIn}")
        return

    columns_to_drop = ["SOURCE_ID", "slope", "slope1", "COType", "FID", "fileName", "X", "Y", "AreaType"]
    df.drop(columns=columns_to_drop, inplace=True)

    #corr
    correlation_matrix = df.corr()
    print("corr Correlation Matrix:")
    print(correlation_matrix)

    highly_correlated_columns = set()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > thresholdCOR:
                colname = correlation_matrix.columns[i]
                highly_correlated_columns.add(colname)

    print("Highly corr Correlated Columns to be Removed:")
    print(highly_correlated_columns)
    result = "threshold: " + str(thresholdCOR) + "\nHighly corr Correlated Columns to be Removed: " + str(highly_correlated_columns)
    output_file_path = f"{fileName}\\{fileNameFull}_corrResult.txt"
    with open(output_file_path, 'w') as file:
        file.write(result)

    correlation_matrix.columns = correlation_matrix.columns.astype(str)
    correlation_matrix.reset_index(inplace=True)
    correlation_matrix.to_csv(f"{fileName}\\{fileNameFull}_corrColumnMatrix.csv", index=False)
    # corr

    #vif
    result = calculate_vif(df)
    print("Highly vif Correlated Columns to be Removed by order:")
    print(result)
    result = "threshold: " + str(thresholdVIF) + "\nHighly vif Correlated Columns to be Removed by order: " + str(result)
    output_file_path = f"{fileName}\\{fileNameFull}_vifResult.txt"
    with open(output_file_path, 'w') as file:
        file.write(result)
    #vif
    print()

def calculate_vif(X, thresh=thresholdVIF):
    X = X.assign(ManuallyAddedConst=1)  # faster than add_constant from statsmodels
    variables = list(range(X.shape[1]))

    # Initial VIF values
    initial_vif = [variance_inflation_factor(X.values, ix) for ix in range(X.shape[1])]
    for i in range(0, len(X.columns), 4):
        print('Initial Column names:', X.columns[i:i + 4])
        print('Initial VIF values:', initial_vif[i:i + 4])

    deleted_columns = []  # List to store deleted column names

    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]
        vif = vif[:-1]  # don't let the constant be removed in the loop.
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('Dropping a column with VIF value larger than thresh')
            # Output column names and VIF values in rows of 4
            for i in range(0, len(variables), 4):
                print('Column names:', X.iloc[:, variables].columns[i:i + 4])
                print('VIF values:', vif[i:i+4])
            deleted_column = X.iloc[:, variables].columns[maxloc]
            print('dropping \'' + deleted_column + '\' at index: ' + str(maxloc))
            deleted_columns.append(deleted_column)
            del variables[maxloc]
            dropped = True

    return deleted_columns

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

with TeeOutput('3.CheckColumnCorrelationLog-np.txt'):
    FirstProcess("np")
with TeeOutput('3.CheckColumnCorrelationLog-areamn.txt'):
    FirstProcess("areamn")
with TeeOutput('3.CheckColumnCorrelationLog-aggregation.txt'):
    FirstProcess("aggregation")
with TeeOutput('3.CheckColumnCorrelationLog-connectivity.txt'):
    FirstProcess("connectivity")