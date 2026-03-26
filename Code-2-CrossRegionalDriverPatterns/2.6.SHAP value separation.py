import sys
import csv
from contextlib import contextmanager
import shap
import pandas as pd
import numpy as np
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
import os
import re
import shutil
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

features = ['Elevation', 'Climate', 'EcologicalZones', 'SoilZones', 'Landform',
            'PopulationBaseNumber', 'PopulationVariation', 'GDPDensityBaseNumber', 'GDPDensityVariation', 'NighttimeLightBaseNumber',
            'NighttimeLightVariation', 'BaselineWaterStress', 'SoilMoisture', 'Aridity', 'TemperatureBaseNumber',
            'TemperatureVariation', 'PrecipitationBaseNumber', 'PrecipitationVariation', 'GlaciersMeltingElevationChange', 'GlaciersMeltingMassChange',
            'SDGRegions', 'AreaTypeNumber']

def Plot(fileName, normalized_df):
    plt.figure(figsize=(10, 6))
    plt.bar(normalized_df.columns, normalized_df.loc[0], color='blue')
    plt.xlabel('Fields')
    plt.ylabel('Normalized Values')
    plt.title(fileName)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)  
    plt.tight_layout()
    plt.savefig(os.path.join("shap-plot", f"{fileName}.png"))
    plt.close()

def Process_NEG(fileName, fileNameFull, mean_values_NEG):
    print(fileNameFull)
    fileIn = f"..\\9.AnalysisSHAPDependencePlot\\{fileName}\\{fileNameFull}-YXShapValuesOthers.csv"
    if(not os.path.exists(fileIn)):
        print("no such file")
        return
    df = pd.read_csv(fileIn)
    df.dropna(how="all", inplace=True)

    df_lt_0 = df[df['target'] < 0]
    print(f"Number of Neg records: {len(df_lt_0)}")
    if(len(df_lt_0) < 5):
        print(f"len(df_lt_0) < 5")
        return
    df_lt_0.to_csv(f"{fileName}\\{fileNameFull}-YXShapValuesOthers-Neg.csv", index=False)

    current_mean_values = {feature: None for feature in features}
    for feature in features:
        columns_matching_value = [col for col in df_lt_0.columns if col == feature or (col.endswith('-value') and col[:-6] == feature)]
        if len(columns_matching_value) == 1:
            next_col = df_lt_0.columns[df_lt_0.columns.get_loc(columns_matching_value[0]) + 1]
            current_mean_values[feature] = df_lt_0[next_col].mean() if not df_lt_0[next_col].empty else 0
        else:
            print("error")

    current_mean_values_df = pd.DataFrame(current_mean_values, index=[0])
    zero_mask = current_mean_values_df == 0  #
    max_value = current_mean_values_df.loc[:, ~zero_mask.iloc[0]].max(axis=1).values[0] #
    min_value = current_mean_values_df.loc[:, ~zero_mask.iloc[0]].min(axis=1).values[0]
    normalized_df = (current_mean_values_df.loc[:, ~zero_mask.iloc[0]] - min_value) / (max_value - min_value)   #
    normalized_df = normalized_df.reindex(columns=current_mean_values_df.columns, fill_value=0)  #
    normalized_df.columns = [f'Nor{col}' for col in normalized_df.columns]

    Plot(f"{fileName}-{fileNameFull}-Neg", normalized_df)
    for col in normalized_df.columns:
        current_mean_values[col] = normalized_df[col].iloc[0]

    current_mean_values['Index'] = fileName
    current_mean_values['SDGRegionsCode'] = re.search(r'\d+', fileNameFull).group()
    current_mean_values['SDGName'] = fileNameFull.split('-', 1)[0]
    current_mean_values['AreaType'] = fileNameFull.split('-', 2)[-1]
    if current_mean_values['SDGName'] == '1Sub':
        current_mean_values['SDGName'] = '1Sub-SaharanAfrica'
        current_mean_values['AreaType'] = current_mean_values['AreaType'].split('-', 1)[1]
    if current_mean_values['SDGName'] == '7South':
        current_mean_values['SDGName'] = '7South-easternAsia'
        current_mean_values['AreaType'] = current_mean_values['AreaType'].split('-', 1)[1]
    current_mean_values['PosNeg'] = 'Neg'
    current_mean_values['NumberOfRecordsForThisMean'] = len(df_lt_0)

    current_mean_values['Number_Of_Records_For_Performance'], current_mean_values['Avg_Oob_Score'], current_mean_values['Avg_Mse'], current_mean_values['Avg_R2'] = GetPerformance(fileName, fileNameFull)

    mean_values_NEG[f"{fileNameFull}-Neg"] = current_mean_values

def Process_POS(fileName, fileNameFull, mean_values_POS):
    print(fileNameFull)
    fileIn = f"..\\9.AnalysisSHAPDependencePlot\\{fileName}\\{fileNameFull}-YXShapValuesOthers.csv"
    if(not os.path.exists(fileIn)):
        print("no such file")
        return
    df = pd.read_csv(fileIn)
    df.dropna(how="all", inplace=True)

    df_ge_0 = df[df['target'] >= 0]
    print(f"Number of Pos records: {len(df_ge_0)}")
    if(len(df_ge_0) < 5):
        print(f"len(df_ge_0) < 5")
        return
    df_ge_0.to_csv(f"{fileName}\\{fileNameFull}-YXShapValuesOthers-Pos.csv", index=False)

    current_mean_values = {feature: None for feature in features}
    for feature in features:
        columns_matching_value = [col for col in df_ge_0.columns if col == feature or (col.endswith('-value') and col[:-6] == feature)]
        if len(columns_matching_value) == 1:
            next_col = df_ge_0.columns[df_ge_0.columns.get_loc(columns_matching_value[0]) + 1]
            current_mean_values[feature] = df_ge_0[next_col].mean() if not df_ge_0[next_col].empty else 0
        else:
            print("error")

    current_mean_values_df = pd.DataFrame(current_mean_values, index=[0])
    zero_mask = current_mean_values_df == 0 #
    max_value = current_mean_values_df.loc[:, ~zero_mask.iloc[0]].max(axis=1).values[0] #
    min_value = current_mean_values_df.loc[:, ~zero_mask.iloc[0]].min(axis=1).values[0]
    normalized_df = (current_mean_values_df.loc[:, ~zero_mask.iloc[0]] - min_value) / (max_value - min_value)   #
    normalized_df = normalized_df.reindex(columns=current_mean_values_df.columns, fill_value=0) #
    normalized_df.columns = [f'Nor{col}' for col in normalized_df.columns]

    Plot(f"{fileName}-{fileNameFull}-Pos", normalized_df)
    for col in normalized_df.columns:
        current_mean_values[col] = normalized_df[col].iloc[0]

    current_mean_values['Index'] = fileName
    current_mean_values['SDGRegionsCode'] = re.search(r'\d+', fileNameFull).group()
    current_mean_values['SDGName'] = fileNameFull.split('-', 1)[0]
    current_mean_values['AreaType'] = fileNameFull.split('-', 2)[-1]
    if current_mean_values['SDGName'] == '1Sub':
        current_mean_values['SDGName'] = '1Sub-SaharanAfrica'
        current_mean_values['AreaType'] = current_mean_values['AreaType'].split('-', 1)[1]
    if current_mean_values['SDGName'] == '7South':
        current_mean_values['SDGName'] = '7South-easternAsia'
        current_mean_values['AreaType'] = current_mean_values['AreaType'].split('-', 1)[1]
    current_mean_values['PosNeg'] = 'Pos'
    current_mean_values['NumberOfRecordsForThisMean'] = len(df_ge_0)

    current_mean_values['Number_Of_Records_For_Performance'], current_mean_values['Avg_Oob_Score'], current_mean_values['Avg_Mse'], current_mean_values['Avg_R2'] = GetPerformance(fileName, fileNameFull)

    mean_values_POS[f"{fileNameFull}-Pos"] = current_mean_values

def GetPerformance(fileName, fileNameFull):
    fileIn = f"..\\4.AnalyseOverallFeatureImportance\\{fileName}\\{fileNameFull}_performance.txt"
    if(not os.path.exists(fileIn)):
        print("no such file")
        return

    with open(fileIn, 'r') as file:
        lines = file.readlines()

    data_dict = {}
    for line in lines:
        key, value = line.strip().split(': ', 1)
        data_dict[key] = value

    return data_dict.get('number_of_records'), data_dict.get('avg_oob_score_'), data_dict.get('avg_mse_'), data_dict.get('avg_r2_')

def FirstProcess(fileName):
    mean_values_POS = {}
    mean_values_NEG = {}

    Process_POS(fileName, f"0World-{fileName}-all", mean_values_POS)
    Process_NEG(fileName, f"0World-{fileName}-all", mean_values_NEG)
    for metric in metrics:
        Process_POS(fileName, f"0World-{fileName}-{metric}", mean_values_POS)
        Process_NEG(fileName, f"0World-{fileName}-{metric}", mean_values_NEG)
    print()

    for i in range(1, 23):
        Process_POS(fileName, f"{mapping[i]}-{fileName}-all", mean_values_POS)
        Process_NEG(fileName, f"{mapping[i]}-{fileName}-all", mean_values_NEG)
        for metric in metrics:
            Process_POS(fileName, f"{mapping[i]}-{fileName}-{metric}", mean_values_POS)
            Process_NEG(fileName, f"{mapping[i]}-{fileName}-{metric}", mean_values_NEG)
        print()

    df_mean_values_POS = pd.DataFrame.from_dict(mean_values_POS, orient='index')
    priority_columns = ['Index', 'SDGRegionsCode', 'SDGName', 'AreaType', 'PosNeg', 'NumberOfRecordsForThisMean',
                        'Number_Of_Records_For_Performance', 'Avg_Oob_Score', 'Avg_Mse', 'Avg_R2']
    other_columns = [col for col in df_mean_values_POS.columns if col not in priority_columns]
    df_mean_values_POS = df_mean_values_POS[priority_columns + other_columns]
    df_mean_values_POS.to_csv(f"{fileName}-shap-POS.csv", index_label='fileName')

    df_mean_values_NEG = pd.DataFrame.from_dict(mean_values_NEG, orient='index')
    priority_columns = ['Index', 'SDGRegionsCode', 'SDGName', 'AreaType', 'PosNeg', 'NumberOfRecordsForThisMean',
                        'Number_Of_Records_For_Performance', 'Avg_Oob_Score', 'Avg_Mse', 'Avg_R2']
    other_columns = [col for col in df_mean_values_NEG.columns if col not in priority_columns]
    df_mean_values_NEG = df_mean_values_NEG[priority_columns + other_columns]
    df_mean_values_NEG.to_csv(f"{fileName}-shap-NEG.csv", index_label='fileName')

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

with TeeOutput('10.SHAPValueSeparatePosNegCompMean-np.txt'):
    FirstProcess("np")
with TeeOutput('10.SHAPValueSeparatePosNegCompMean-areamn.txt'):
    FirstProcess("areamn")
with TeeOutput('10.SHAPValueSeparatePosNegCompMean-aggregation.txt'):
    FirstProcess("aggregation")
with TeeOutput('10.SHAPValueSeparatePosNegCompMean-connectivity.txt'):
    FirstProcess("connectivity")


