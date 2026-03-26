import pandas as pd
import numpy as np
import pymannkendall as mk
from sklearn.linear_model import LinearRegression
import os
from tqdm import tqdm

CSV_PATH = 'SW_metrics_all_years.csv'
YEAR_START = 2001
YEAR_END = 2021
YEARS = list(range(YEAR_START, YEAR_END + 1))
CENTRAL_START = YEAR_START + 3
CENTRAL_END = YEAR_END - 3
MIN_CENTRAL_VALID = 8
MK_P_THRESHOLD = 0.01

NP_cols = [f'NP_c{i}' for i in range(5)]
AREAMN_cols = [f'AREAMN_c{i}' for i in range(5)]
AGGREGATION_cols = [f'AGGREGATION_c{i}' for i in range(5)]
CONNECTIVITY_cols = [f'CONNECTIVITY_c{i}' for i in range(5)]

metric_groups = {
    'NP': NP_cols,
    'AREAMN': AREAMN_cols,
    'AGGREGATION': AGGREGATION_cols,
    'CONNECTIVITY': CONNECTIVITY_cols
}

df = pd.read_csv(CSV_PATH)

print('Rows in CSV:', len(df))
print('Columns:', df.columns.tolist()[:50])

if not set(['lon0', 'lat0', 'year']).issubset(set(df.columns)):
    raise ValueError('CSV must contain lon0, lat0, year columns.')

# Output columns for trend analysis
out_cols = ['lon0', 'lat0', 'area_class', 'metric', 'mk_trend', 'mk_p', 'slope', 'intercept', 'n_valid',
            'n_central_valid']
results = []

# Function to get time series for each grid cell and metric column
def get_timeseries_for_cell_col(df_cell, colname):
    ser = df_cell.set_index('year')[colname].reindex(YEARS).astype(float)  # Use reindex to directly align with YEARS
    return ser

unique_cells = df[['lon0', 'lat0']].drop_duplicates()
print('Unique cells:', len(unique_cells))

# Iterate over each unique grid cell
for _, cell in tqdm(unique_cells.iterrows(), total=len(unique_cells)):
    lon0 = cell['lon0']
    lat0 = cell['lat0']
    df_cell = df[(df['lon0'] == lon0) & (df['lat0'] == lat0)]

    for area_idx in range(5):
        for metric_name, cols in metric_groups.items():
            colname = cols[area_idx]
            if colname not in df.columns:
                continue
            ser = get_timeseries_for_cell_col(df_cell, colname)

            # Create valid mask and count valid data points
            valid_mask = ~ser.isna()
            n_valid = valid_mask.sum()

            # Check the criteria for first 3 and last 3 years, and central years
            first3 = YEARS[:3]
            last3 = YEARS[-3:]
            has_first3 = ser[first3].notna().any()
            has_last3 = ser[last3].notna().any()
            central_years = list(range(CENTRAL_START, CENTRAL_END + 1))
            n_central_valid = ser[central_years].notna().sum()

            meets_criteria = has_first3 and has_last3 and (n_central_valid >= MIN_CENTRAL_VALID)
            if not meets_criteria:
                continue

            values = ser.values
            if n_valid < 3:
                continue

            # Mann-Kendall trend test
            try:
                mk_result = mk.original_test(values)
                mk_p = mk_result.p
                mk_trend = mk_result.trend
            except Exception as e:
                mk_p = np.nan
                mk_trend = 'no_result'

            # If no valid Mann-Kendall result, continue
            if mk_p is None or np.isnan(mk_p):
                continue
            if mk_p >= MK_P_THRESHOLD:
                continue

            # Perform linear regression to calculate slope and intercept
            try:
                yrs = np.array([y for y in YEARS if not np.isnan(ser.at[y])]).reshape(-1, 1)
                vals = np.array([ser.at[y] for y in YEARS if not np.isnan(ser.at[y])]).reshape(-1, 1)
                lr = LinearRegression()
                lr.fit(yrs, vals)
                slope = float(lr.coef_[0][0])
                intercept = float(lr.intercept_[0])

                # Append results to list
                results.append(
                    [lon0, lat0, area_idx, metric_name, mk_trend, float(mk_p), slope, intercept, int(n_valid),
                     int(n_central_valid)])
            except Exception as e:
                print(f"Error with linear regression for {lon0}, {lat0}, {metric_name}, area {area_idx}: {e}")
                continue

# Create DataFrame from results
res_df = pd.DataFrame(results, columns=out_cols)
outpath = 'SW_metrics_trends_results.csv'
res_df.to_csv(outpath, index=False)
print('Trend analysis finished. Results saved to', outpath)
