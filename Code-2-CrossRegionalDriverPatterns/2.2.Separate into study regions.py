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

mapping = {
    1: '1Sub-SaharanAfrica', 2: '2NorthernAfrica', 3: '3WesternAsia', 4: '4CentralAsia', 5: '5SouthernAsia',
    6: '6EasternAsia', 7: '7South-easternAsia', 8: '8NorthernAsia', 9: '9Caribbean', 10: '10CentralAmerica',
    11: '11SouthAmerica', 12: '12AustraliaAndNewZealand', 13: '13OceaniaExcludingAUSAndNZ', 14: '14ContiguousUnitedStates', 15: '15Alaska' ,
    16: '16CanadaProvinces', 17: '17CanadaTerritories', 18: '18Greenland', 19: '19EasternEurope', 20: '20NorthernEurope',
    21: '21SouthernEurope', 22: '22WesternEurope'
}

def Process(fileName):
    print(f"{fileName}, number of records:")

    filePath = (f"..\\2.ThenPutDifferentAreaInOneFileAndRemoveInvalidValues\\{fileName}-3to8InOneFile.csv")
    pdAll = pd.read_csv(filePath)
    pdAll.dropna(how = "all", inplace = True)

    print(f"0World-{fileName}-all: {len(pdAll)}")
    pdAll.to_csv(f"{fileName}\\0World-{fileName}-all.csv", index = False)
    for metric in metrics:
        pdTarget = pdAll[pdAll['AreaType'] == metric]
        print(f"0World-{fileName}-{metric}: {len(pdTarget)}")
        pdTarget.to_csv(f"{fileName}\\0World-{fileName}-{metric}.csv", index=False)
    print()

    for i in range(1, 23):
        pdTarget = pdAll[pdAll['SDGRegions-value'] == i]
        print(f"{mapping[i]}-{fileName}-all: {len(pdTarget)}")
        pdTarget.to_csv(f"{fileName}\\{mapping[i]}-{fileName}-all.csv", index=False)
        for metric in metrics:
            pdTarget2 = pdTarget[pdTarget['AreaType'] == metric]
            print(f"{mapping[i]}-{fileName}-{metric}: {len(pdTarget2)}")
            pdTarget2.to_csv(f"{fileName}\\{mapping[i]}-{fileName}-{metric}.csv", index=False)
        print()

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

with TeeOutput('2.ThenSeparateIntoStudyRegionsLog-np.txt'):
    Process("np")
with TeeOutput('2.ThenSeparateIntoStudyRegionsLog-area_mn.txt'):
    Process("areamn")
with TeeOutput('2.ThenSeparateIntoStudyRegionsLog-ai.txt'):
    Process("aggregation")
with TeeOutput('2.ThenSeparateIntoStudyRegionsLog-cohesion.txt'):
    Process("connectivity")

