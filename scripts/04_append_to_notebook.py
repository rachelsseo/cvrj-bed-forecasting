
import json
import os

NOTEBOOK_PATH = "02_initial_model_draft.ipynb"

# The code to append (Cleaned version of run_forecast.py)
CODE_TO_APPEND = r"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Update paths to be relative to the notebook or absolute
# Assuming notebook is in scripts/
DATA_DIR = "../data/raw" 
CVRJ_FILE = "../data/processed/cvrj_dataset_v2.csv" 
# Check specific file location:
if not os.path.exists(CVRJ_FILE):
    # Try absolute path if relative fails, or parent
    CVRJ_FILE = "../data/processed/cvrj_dataset_v2.csv"
    DATA_DIR = "../data/raw"

print(f"Using CVRJ File: {CVRJ_FILE}")

# --- 1. Load and Process CVRJ Data ---
df_cvrj = pd.read_csv(CVRJ_FILE)
df_cvrj['Book Date'] = pd.to_datetime(df_cvrj['Book Date'], errors='coerce')
df_cvrj['Release Date'] = pd.to_datetime(df_cvrj['Release Date'], errors='coerce')

# Filter for relevant years (2012-2025)
df_cvrj = df_cvrj[df_cvrj['Book Date'].dt.year >= 2012]

# Calculate Daily Census (ADP)
min_date = df_cvrj['Book Date'].min()
max_date = df_cvrj['Release Date'].max()
if pd.isnull(max_date) or max_date.year > 2030: 
    max_date = pd.Timestamp('2025-12-31')

date_range = pd.date_range(start=min_date.normalize(), end=max_date.normalize(), freq='D')

events = []
for _, row in df_cvrj.iterrows():
    start = row['Book Date']
    end = row['Release Date']
    if pd.isnull(start): continue
    if pd.isnull(end): end = max_date
    start_date = start.normalize()
    end_date = end.normalize()
    if end_date < start_date: continue
    events.append((start_date, 1))
    events.append((end_date + pd.Timedelta(days=1), -1))

evt_df = pd.DataFrame(events, columns=['Date', 'Change'])
if not evt_df.empty:
    evt_df = evt_df.groupby('Date')['Change'].sum().sort_index()
    daily_census = evt_df.reindex(date_range, fill_value=0).cumsum()
else:
    daily_census = pd.Series(0, index=date_range)

# Aggregating to Annual ADP
annual_cvrj_adp = daily_census.resample('Y').mean()
print("CVRJ Annual ADP (Head):")
print(annual_cvrj_adp.head())

# --- 2. Load Population Data ---
counties = ['Fluvanna', 'Greene', 'Louisa', 'Madison', 'Orange']
pop_data = {}

for county in counties:
    fpath = os.path.join(DATA_DIR, f"{county}Population.csv")
    try:
        if os.path.exists(fpath):
            temp_df = pd.read_csv(fpath, header=4)
            target_row = temp_df.iloc[1]
            years = [str(y) for y in range(2012, 2025)]
            pop_series = target_row[years].str.replace(',', '').astype(float)
            pop_series.index = pd.to_datetime(pop_series.index, format='%Y')
            pop_data[county] = pop_series
        else:
            print(f"Warning: {fpath} not found.")
    except Exception as e:
        print(f"Error loading {county}: {e}")

if pop_data:
    total_cvrj_pop = pd.DataFrame(pop_data).sum(axis=1)
    total_cvrj_pop.index = total_cvrj_pop.index + pd.offsets.YearEnd()
else:
    total_cvrj_pop = pd.Series()

# --- 3. Load Culpeper Data ---
culpeper_data = {
    '2020': 148.93, '2021': 185.75, '2022': 210.82, 
    '2023': 200.24, '2024': 312.33, '2025': 227.81
}
culpeper_adp = pd.Series(culpeper_data)
culpeper_adp.index = pd.to_datetime(culpeper_adp.index, format='%Y') + pd.offsets.YearEnd()

# Culpeper Census
culp_pop_file = os.path.join(DATA_DIR, "CulpeperPopulation.csv")
culp_pop = pd.Series()
if os.path.exists(culp_pop_file):
    try:
        temp_culp = pd.read_csv(culp_pop_file, header=4)
        culp_pop_row = temp_culp.iloc[1]
        years = [str(y) for y in range(2012, 2025)]
        culp_pop = culp_pop_row[years].str.replace(',', '').astype(float)
        culp_pop.index = pd.to_datetime(culp_pop.index, format='%Y') + pd.offsets.YearEnd()
    except Exception as e:
        print(f"Error loading Culpeper Census: {e}")

# --- 4. SARIMA Modeling & Forecasting ---

def fit_forecast(adp_series, exog_series, name, steps=10):
    print(f"\nModeling {name}...")
    combined = pd.concat([adp_series, exog_series], axis=1).dropna()
    combined.columns = ['ADP', 'Pop']
    
    if combined.empty:
        print(f"Empty training data for {name}")
        return None, None
        
    try:
        model = SARIMAX(combined['ADP'], exog=combined['Pop'], order=(1,1,1), seasonal_order=(0,0,0,0))
        results = model.fit(disp=False)
        print(results.summary())
        
        # Forecast Population (Linear Trend)
        pop_y = combined['Pop'].values
        pop_x = np.arange(len(pop_y))
        z = np.polyfit(pop_x, pop_y, 1)
        p = np.poly1d(z)
        future_x = np.arange(len(pop_y), len(pop_y) + steps)
        future_pop = p(future_x)
        
        forecast = results.get_forecast(steps=steps, exog=future_pop.reshape(-1,1))
        forecast_vals = forecast.predicted_mean
        return forecast_vals, future_pop
    except Exception as e:
        print(f"Error modeling {name}: {e}")
        return None, None

# Model CVRJ
if not total_cvrj_pop.empty:
    # Extrapolate Pop 2025
    last_pop = total_cvrj_pop.iloc[-1]
    prev_pop = total_cvrj_pop.iloc[-2]
    total_cvrj_pop.loc[pd.Timestamp('2025-12-31')] = last_pop + (last_pop - prev_pop)

cvrj_forecast, cvrj_fut_pop = fit_forecast(annual_cvrj_adp, total_cvrj_pop, "CVRJ")

# Model Culpeper
if not culp_pop.empty:
    last_c_pop = culp_pop.iloc[-1]
    prev_c_pop = culp_pop.iloc[-2]
    culp_pop.loc[pd.Timestamp('2025-12-31')] = last_c_pop + (last_c_pop - prev_c_pop)

culpeper_clean = culpeper_adp[~culpeper_adp.index.year.isin([2020, 2021])]
culpeper_forecast, culp_fut_pop = fit_forecast(culpeper_clean, culp_pop, "Culpeper")

# --- 5. Visualization ---
if cvrj_forecast is not None and culpeper_forecast is not None:
    years_future = pd.date_range(start='2026-12-31', periods=10, freq='Y')
    res_df = pd.DataFrame({
        'CVRJ_Baseline': cvrj_forecast.values,
        'Culpeper_Add': culpeper_forecast.values
    }, index=years_future)
    res_df['Total_Projected'] = res_df['CVRJ_Baseline'] + res_df['Culpeper_Add']
    
    print("\nProjected Capacity Usage:")
    print(res_df)
    
    plt.figure(figsize=(10,6))
    plt.plot(res_df.index, res_df['Total_Projected'], label='Projected Total (CVRJ + Culpeper)', marker='o')
    plt.plot(res_df.index, res_df['CVRJ_Baseline'], label='CVRJ Baseline', linestyle='--')
    plt.axhline(y=660, color='r', linestyle='-', label='Max Capacity (660)')
    plt.title('CVRJ Capacity Forecast with Culpeper Integration')
    plt.legend()
    plt.grid(True)
    plt.show()
"""

# Create a new code cell
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {
        "collapsed": False
    },
    "outputs": [],
    "source": CODE_TO_APPEND.split('\n')
}
# Fix newlines in source - split removes them, ipynb expects newline chars at end of strings usually or lists.
# Actually standard is list of strings, usually ending with \n.
new_cell["source"] = [line + '\n' for line in CODE_TO_APPEND.split('\n')]

# Create markdown cell
md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## Forecast Model Implementation\n", "The following code processes the raw booking data to calculate ADP, loads census data, and performs SARIMA forecasting."]
}

# Read and Append
with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

nb['cells'].append(md_cell)
nb['cells'].append(new_cell)

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
