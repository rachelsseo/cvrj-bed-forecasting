
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import os

# Define paths
DATA_DIR = "cvrj-bed-forecasting/ForecastBedData"
CVRJ_FILE = os.path.join(DATA_DIR, "cvrj_dataset_v2.csv")

# --- 1. Load and Process CVRJ Data ---
print("Loading CVRJ Data...")
df_cvrj = pd.read_csv(CVRJ_FILE)
df_cvrj['Book Date'] = pd.to_datetime(df_cvrj['Book Date'], errors='coerce')
df_cvrj['Release Date'] = pd.to_datetime(df_cvrj['Release Date'], errors='coerce')

# Filter for relevant years (2012-2025)
df_cvrj = df_cvrj[df_cvrj['Book Date'].dt.year >= 2012]
print(f"Total CVRJ rows: {len(df_cvrj)}")

# EXCLUDE CULPEPER (Code 47) from Baseline
# "The image with less [Culpeper Jail] shows inmates just in Culpeper... 
# data set with 'any jail' includes cvrj inmates...
# So to forecast Total Load = Baseline (No Culpeper) + Total Culpeper (Any Jail)."
df_cvrj_baseline = df_cvrj[df_cvrj['County Code'] != 47]
print(f"CVRJ Baseline (No Culpeper) rows: {len(df_cvrj_baseline)}")

# Calculate Daily Census (ADP) for Baseline
min_date = df_cvrj_baseline['Book Date'].min()
max_date = df_cvrj_baseline['Release Date'].max()
if pd.isnull(max_date) or max_date.year > 2030:
    max_date = pd.Timestamp('2025-12-31')

date_range = pd.date_range(start=min_date.normalize(), end=max_date.normalize(), freq='D')

events = []
for _, row in df_cvrj_baseline.iterrows():
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

# Annual ADP Baseline
annual_cvrj_adp = daily_census.resample('Y').mean()
print("CVRJ Baseline Annual ADP (Head):")
print(annual_cvrj_adp.head())

# --- 2. Load Population Data (CVRJ Excl Culpeper) ---
print("\nLoading Population Data...")
counties = ['Fluvanna', 'Greene', 'Louisa', 'Madison', 'Orange']
pop_data = {}

for county in counties:
    fpath = os.path.join(DATA_DIR, f"{county}Population.csv")
    try:
        temp_df = pd.read_csv(fpath, header=4)
        target_row = temp_df.iloc[1]
        years = [str(y) for y in range(2012, 2025)]
        pop_series = target_row[years].str.replace(',', '').astype(float)
        pop_series.index = pd.to_datetime(pop_series.index, format='%Y')
        pop_data[county] = pop_series
    except Exception as e:
        print(f"Error loading {county}: {e}")

total_cvrj_pop = pd.DataFrame(pop_data).sum(axis=1)
total_cvrj_pop.index = total_cvrj_pop.index + pd.offsets.YearEnd()

# --- 3. Load Culpeper Data (Total 'Any Jail' Demand) ---
print("\nLoading Culpeper Data (Any Jail)...")
culpeper_data = {
    '2020': 148.93,
    '2021': 185.75,
    '2022': 210.82,
    '2023': 200.24,
    '2024': 312.33,
    '2025': 227.81
}
culpeper_adp = pd.Series(culpeper_data)
culpeper_adp.index = pd.to_datetime(culpeper_adp.index, format='%Y') + pd.offsets.YearEnd()

# Culpeper Census
culp_pop_file = os.path.join(DATA_DIR, "CulpeperPopulation.csv")
try:
    temp_culp = pd.read_csv(culp_pop_file, header=4)
    culp_pop_row = temp_culp.iloc[1]
    years = [str(y) for y in range(2012, 2025)]
    culp_pop = culp_pop_row[years].str.replace(',', '').astype(float)
    culp_pop.index = pd.to_datetime(culp_pop.index, format='%Y') + pd.offsets.YearEnd()
except Exception as e:
    print(f"Error loading Culpeper Census: {e}")

# --- 4. SARIMA Modeling ---
def fit_forecast(adp_series, exog_series, name, steps=10):
    print(f"\nModeling {name}...")
    combined = pd.concat([adp_series, exog_series], axis=1).dropna()
    combined.columns = ['ADP', 'Pop']
    
    if combined.empty:
        print(f"Error: Empty training data for {name}")
        return None, None
    
    try:
        model = SARIMAX(combined['ADP'], exog=combined['Pop'], order=(1,1,1), seasonal_order=(0,0,0,0))
        results = model.fit(disp=False)
        print(results.summary())
        
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

# Run CVRJ Model (Baseline, no 47)
if not total_cvrj_pop.empty:
    last_pop = total_cvrj_pop.iloc[-1]
    prev_pop = total_cvrj_pop.iloc[-2]
    pop_2025 = last_pop + (last_pop - prev_pop)
    total_cvrj_pop.loc[pd.Timestamp('2025-12-31')] = pop_2025
    
cvrj_forecast, cvrj_fut_pop = fit_forecast(annual_cvrj_adp, total_cvrj_pop, "CVRJ_Baseline")

# Run Culpeper Model (Total Demand)
if not culp_pop.empty:
    last_c_pop = culp_pop.iloc[-1]
    prev_c_pop = culp_pop.iloc[-2]
    c_pop_2025 = last_c_pop + (last_c_pop - prev_c_pop)
    culp_pop.loc[pd.Timestamp('2025-12-31')] = c_pop_2025

culpeper_clean = culpeper_adp[~culpeper_adp.index.year.isin([2020, 2021])]
culpeper_forecast, culp_fut_pop = fit_forecast(culpeper_clean, culp_pop, "Culpeper_Total")

# --- 5. Analysis ---
if cvrj_forecast is not None and culpeper_forecast is not None:
    years_future = pd.date_range(start='2026-12-31', periods=10, freq='Y')
    print("\nForecast Results (2026-2035):")
    res_df = pd.DataFrame({
        'CVRJ_Baseline_NoCulpeper': cvrj_forecast.values,
        'Culpeper_Total_Demand': culpeper_forecast.values
    }, index=years_future)
    
    # Combined Load = CVRJ (No 47) + Culpeper (Total)
    res_df['Combined_Load'] = res_df['CVRJ_Baseline_NoCulpeper'] + res_df['Culpeper_Total_Demand']
    res_df['Capacity'] = 660
    res_df['Over_Capacity'] = res_df['Combined_Load'] > 660
    
    print(res_df)
    
    plt.figure(figsize=(10,6))
    plt.plot(res_df.index, res_df['Combined_Load'], label='Projected Combined Load', marker='o', linewidth=2)
    plt.plot(res_df.index, res_df['CVRJ_Baseline_NoCulpeper'], label='CVRJ Baseline (Excl. Culpeper)', linestyle='--')
    plt.plot(res_df.index, res_df['Culpeper_Total_Demand'], label='Culpeper Total Demand', linestyle='-')
    plt.axhline(y=660, color='r', linestyle='-', label='Max Capacity (660)')
    plt.title('CVRJ Capacity Forecast (Recalculated)')
    plt.ylabel('Inmate Population')
    plt.legend()
    plt.grid(True)
    plt.savefig('forecast_plot_revised.png')
    print("\nPlot saved to forecast_plot_revised.png")
