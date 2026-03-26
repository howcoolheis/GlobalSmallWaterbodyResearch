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
import argparse

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

clusterNo = 6

Landscapes = ["np", "areamn", "aggregation", "connectivity"]
PosNegs = ["Pos", "Neg"]
SDGNames = ['0World', '8NorthernAsia', '4CentralAsia', '6EasternAsia', '3WesternAsia',
            '5SouthernAsia', '7South-easternAsia', '20NorthernEurope', '19EasternEurope', '22WesternEurope',
            '21SouthernEurope', '2NorthernAfrica', '1Sub-SaharanAfrica', '13Oceania(excludingAUSandNZ)', '12AustraliaAndNewZealand',
            '18Greenland', '15Alaska', '17CanadaTerritories', '16CanadaProvinces', '14ContiguousUnitedStates',
            '10CentralAmerica', '11SouthAmerica', '9Caribbean']
AreaTypes = ['all', '3-4', '4-5', '5-6', '6-7', '7-8']

Features = ['1Elevation', '2Landform', '3EcologicalZones', '4Climate', '5SoilMoisture', '6BaselineWaterStress', '7Aridity',
            '8TemperatureBaseNumber', '9TemperatureVariation',
            '10PrecipitationBaseNumber', '11PrecipitationVariation',
            '12GlaciersMeltingElevationChange', '13GlaciersMeltingMassChange',
            '14PopulationBaseNumber', '15PopulationVariation',
            '16GDPDensityBaseNumber', '17GDPDensityVariation',
            '18SDGRegions', '19AreaTypeNumber']

class TeeOutput:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        try:
            self.file.write(data)
            self.file.flush()
        except Exception:
            pass
        try:
            self.stdout.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self.file.flush()
        except:
            pass
        try:
            self.stdout.flush()
        except:
            pass

def merge_shap_files(input_folder=None):
    cwd = input_folder if input_folder else "."
    all_df = pd.DataFrame()
    candidate_dirs = [cwd, os.path.join(cwd, "11.SHAPValueAllAndClustering"), "..\\11.SHAPValueAllAndClustering", "../11.SHAPValueAllAndClustering"]
    found_any = False

    for Landscape in Landscapes:
        for PosNeg in PosNegs:
            filename = f"{Landscape}-shap-{PosNeg}-Cluster：{clusterNo}.csv"
            file_found = None
            for d in candidate_dirs:
                path = os.path.join(d, filename)
                if os.path.exists(path):
                    file_found = path
                    break
            if file_found is None:
                if os.path.exists(filename):
                    file_found = filename
            if file_found:
                try:
                    df = pd.read_csv(file_found)
                    print(f"Loaded {file_found} rows: {len(df)}")
                    all_df = pd.concat([all_df, df], ignore_index=True)
                    found_any = True
                except Exception as e:
                    print(f"Failed to read {file_found}: {e}")
            else:
                print(f"Warning: file not found: {filename} (searched dirs: {candidate_dirs})")

    if not found_any:
        print("No SHAP files found. Exiting merge_shap_files()")
        return None

    out_origin = f"all-shap-origin-Cluster：{clusterNo}.csv"
    all_df.to_csv(out_origin, index=False)
    print(f"Wrote merged origin CSV: {out_origin}")

    all_df_sorted_and_full = pd.DataFrame(columns=all_df.columns)
    for SDGName in SDGNames:
        for Landscape in Landscapes:
            for PosNeg in PosNegs:
                for AreaType in AreaTypes:
                    filtered_df = all_df[(all_df.get('Index') == Landscape) & (all_df.get('PosNeg') == PosNeg) & (all_df.get('SDGName') == SDGName) & (all_df.get('AreaType') == AreaType)]
                    if not filtered_df.empty:
                        all_df_sorted_and_full = pd.concat([all_df_sorted_and_full, filtered_df], ignore_index=True)
                    else:
                        new_row = {col: np.nan for col in all_df.columns}
                        new_row['fileName'] = f"{SDGName}-{Landscape}-{AreaType}-{PosNeg}"
                        new_row['Index'] = Landscape
                        new_row['PosNeg'] = PosNeg
                        new_row['SDGName'] = SDGName
                        new_row['AreaType'] = AreaType
                        all_df_sorted_and_full = pd.concat([all_df_sorted_and_full, pd.DataFrame([new_row])], ignore_index=True)

    out_sorted = f"all-shap-sorted-and-full-Cluster：{clusterNo}.csv"
    all_df_sorted_and_full.to_csv(out_sorted, index=False)
    print(f"Wrote sorted & full CSV: {out_sorted}")

    return out_origin, out_sorted

results_df = pd.DataFrame(columns=['SDGName', 'Index', 'PosNeg', 'AreaType', '2', '1', '*', '-2', '-1', '-*'])

def GetValues(filtered_df, PosNeg, AreaType):
    count_for_all_for_2 = {feature: 0 for feature in Features}
    count_for_all_for_1 = {feature: 0 for feature in Features}
    count_for_all_for_star = {feature: 0 for feature in Features}

    if filtered_df is None or filtered_df.empty:
        return count_for_all_for_2, count_for_all_for_1, count_for_all_for_star

    if PosNeg == PosNegs[0]:  # 'Pos'
        if AreaType == AreaTypes[0]:
            row = filtered_df.iloc[0] if len(filtered_df) > 0 else None
            if row is not None:
                for Feature in Features:
                    val = row.get(Feature)
                    if val == '2':
                        count_for_all_for_2[Feature] += 1
                    elif val == '1':
                        count_for_all_for_1[Feature] += 1
                    elif val == '*':
                        count_for_all_for_star[Feature] += 1
        else:
            for _, row in filtered_df.iterrows():
                for Feature in Features:
                    val = row.get(Feature)
                    if val == '2':
                        count_for_all_for_2[Feature] += 1
                    elif val == '1':
                        count_for_all_for_1[Feature] += 1
                    elif val == '*':
                        count_for_all_for_star[Feature] += 1
    elif PosNeg == PosNegs[1]:  # 'Neg'
        if AreaType == AreaTypes[0]:
            row = filtered_df.iloc[0] if len(filtered_df) > 0 else None
            if row is not None:
                for Feature in Features:
                    val = row.get(Feature)
                    if val == '-2':
                        count_for_all_for_2[Feature] += 1
                    elif val == '-1':
                        count_for_all_for_1[Feature] += 1
                    elif val == '*':
                        count_for_all_for_star[Feature] += 1
        else:
            for _, row in filtered_df.iterrows():
                for Feature in Features:
                    val = row.get(Feature)
                    if val == '-2':
                        count_for_all_for_2[Feature] += 1
                    elif val == '-1':
                        count_for_all_for_1[Feature] += 1
                    elif val == '*':
                        count_for_all_for_star[Feature] += 1

    return count_for_all_for_2, count_for_all_for_1, count_for_all_for_star

def Parse(data):
    result = ''
    for Feature in Features:
        if data.get(Feature, 0) > 0:
            result += f'{Feature}: {data[Feature]}; '
    if result:
        result = result.rstrip('; ')
    return result

def Output(SDGName, Landscape, PosNeg, AreaType, count_for_2, count_for_1, count_for_star):
    global results_df
    record = {}
    if PosNeg == PosNegs[0]:  # Pos
        record = {'SDGName': SDGName, 'Index': Landscape, 'PosNeg': PosNeg, 'AreaType': AreaType,
                  '2': Parse(count_for_2), '1': Parse(count_for_1), '*': Parse(count_for_star)}
    elif PosNeg == PosNegs[1]:  # Neg
        record = {'SDGName': SDGName, 'Index': Landscape, 'PosNeg': PosNeg, 'AreaType': AreaType,
                  '-2': Parse(count_for_2), '-1': Parse(count_for_1), '-*': Parse(count_for_star)}
    results_df = pd.concat([results_df, pd.DataFrame([record])], ignore_index=True)

def analyze_shap_file(fileName):
    global results_df
    results_df = pd.DataFrame(columns=['SDGName', 'Index', 'PosNeg', 'AreaType', '2', '1', '*', '-2', '-1', '-*'])

    if not os.path.exists(fileName):
        print(f"Input file not found: {fileName}")
        return None

    try:
        dfAll = pd.read_csv(f"{fileName}")
    except Exception as e:
        print(f"Failed to read {fileName}: {e}")
        return None

    for SDGName in SDGNames:
        for Landscape in Landscapes:
            filtered_df = dfAll[(dfAll.get('PosNeg') == PosNegs[0]) &
                                (dfAll.get('SDGName') == SDGName) &
                                (dfAll.get('Index') == Landscape) &
                                (dfAll.get('AreaType') == AreaTypes[0])]
            pos_for_all_for_2, pos_for_all_for_1, pos_for_all_for_star = GetValues(filtered_df, PosNegs[0], AreaTypes[0])
            Output(SDGName, Landscape, PosNegs[0], AreaTypes[0],
                   pos_for_all_for_2, pos_for_all_for_1, pos_for_all_for_star)

            filtered_df = dfAll[(dfAll.get('PosNeg') == PosNegs[0]) &
                                (dfAll.get('SDGName') == SDGName) &
                                (dfAll.get('Index') == Landscape) &
                                (dfAll.get('AreaType').isin([AreaTypes[1], AreaTypes[2], AreaTypes[3], AreaTypes[4], AreaTypes[5]]))]
            pos_23456_for_2, pos_23456_for_1, pos_23456_for_star = GetValues(filtered_df, PosNegs[0],
                                                                                                      AreaTypes[1]+AreaTypes[2]+AreaTypes[3]+AreaTypes[4]+AreaTypes[5])
            Output(SDGName, Landscape, PosNegs[0], f"{AreaTypes[1]}-{AreaTypes[2]}-{AreaTypes[3]}-{AreaTypes[4]}-{AreaTypes[5]}",
                   pos_23456_for_2, pos_23456_for_1, pos_23456_for_star)

            filtered_df = dfAll[(dfAll.get('PosNeg') == PosNegs[1]) &
                                (dfAll.get('SDGName') == SDGName) &
                                (dfAll.get('Index') == Landscape) &
                                (dfAll.get('AreaType') == AreaTypes[0])]
            neg_for_all_for_2, neg_for_all_for_1, neg_for_all_for_star = GetValues(filtered_df, PosNegs[1], AreaTypes[0])
            Output(SDGName, Landscape, PosNegs[1], AreaTypes[0],
                   neg_for_all_for_2, neg_for_all_for_1, neg_for_all_for_star)

            filtered_df = dfAll[(dfAll.get('PosNeg') == PosNegs[1]) &
                                (dfAll.get('SDGName') == SDGName) &
                                (dfAll.get('Index') == Landscape) &
                                (dfAll.get('AreaType').isin([AreaTypes[1], AreaTypes[2], AreaTypes[3], AreaTypes[4], AreaTypes[5]]))]
            neg_23456_for_2, neg_23456_for_1, neg_23456_for_star = GetValues(filtered_df, PosNegs[1],
                                                                                                      AreaTypes[1]+AreaTypes[2]+AreaTypes[3]+AreaTypes[4]+AreaTypes[5])
            Output(SDGName, Landscape, PosNegs[1], f"{AreaTypes[1]}-{AreaTypes[2]}-{AreaTypes[3]}-{AreaTypes[4]}-{AreaTypes[5]}",
                   neg_23456_for_2, neg_23456_for_1, neg_23456_for_star)

    out_all = "Result_All.csv"
    results_df.to_csv(out_all, index=False)
    print(f"Wrote analysis results: {out_all}")

    results_df_cleaned = results_df.copy()
    try:
        results_df_cleaned[['2', '1', '*', '-2', '-1', '-*']] = results_df_cleaned[['2', '1', '*', '-2', '-1', '-*']].replace(['', '<NA>'], pd.NA)
    except Exception:
        for c in ['2', '1', '*', '-2', '-1', '-*']:
            if c not in results_df_cleaned.columns:
                results_df_cleaned[c] = pd.NA

    results_df_cleaned = results_df_cleaned.dropna(subset=['2', '1', '*', '-2', '-1', '-*'], how='all')
    out_clean = "Result_Cleaned.csv"
    results_df_cleaned.to_csv(out_clean, index=False)
    print(f"Wrote cleaned results: {out_clean}")

    return out_all, out_clean

def main():
    parser = argparse.ArgumentParser(description='Merged SHAP merge & analysis script.')
    parser.add_argument('--merge', action='store_true', help='Run merge_shap_files() to merge SHAP CSVs.')
    parser.add_argument('--analyze', action='store_true', help='Run analyze_shap_file() on a provided input CSV.')
    parser.add_argument('--input', type=str, default='13.ExportData-Cluster：4Edit20240805.csv', help='Input filename for analysis (CSV).')
    parser.add_argument('--input_folder', type=str, default='.', help='Input folder to search for SHAP files when merging.')
    parser.add_argument('--log', type=str, default='MergedScriptLog.txt', help='Log file name.')
    args = parser.parse_args()

    with TeeOutput(args.log):
        print('===== Merged Script Start =====')
        print('Timestamp:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if args.merge:
            print('--- Running merge_shap_files ---')
            try:
                out_origin, out_sorted = merge_shap_files(input_folder=args.input_folder)
                print('merge_shap_files completed. Outputs:', out_origin, out_sorted)
            except Exception as e:
                print('merge_shap_files error:', e)

        if args.analyze:
            print('--- Running analyze_shap_file ---')
            try:
                out_all, out_clean = analyze_shap_file(args.input)
                print('analyze_shap_file completed. Outputs:', out_all, out_clean)
            except Exception as e:
                print('analyze_shap_file error:', e)

        if not args.merge and not args.analyze:
            print('No action requested. Use --merge and/or --analyze. Example:')
            print('  python merged_analysis.py --merge')
            print('  python merged_analysis.py --analyze --input "13.ExportData-Cluster：4Edit20240805.csv"')

        print('===== Merged Script End =====')

if __name__ == '__main__':
    main()