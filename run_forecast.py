
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

# Calculate Daily Census (ADP) for Baseline (vectorized)
df_b = df_cvrj_baseline.dropna(subset=['Book Date']).copy()
max_date = df_b['Release Date'].max()
if pd.isnull(max_date) or max_date.year > 2030:
    max_date = pd.Timestamp('2025-12-31')
min_date = df_b['Book Date'].min()
df_b['Release Date'] = df_b['Release Date'].fillna(max_date)
df_b['start_d'] = df_b['Book Date'].dt.normalize()
df_b['end_d'] = df_b['Release Date'].dt.normalize() + pd.Timedelta(days=1)
df_b = df_b[df_b['end_d'] > df_b['start_d']]
enter = df_b[['start_d']].rename(columns={'start_d': 'Date'})
enter['Change'] = 1
leave = df_b[['end_d']].rename(columns={'end_d': 'Date'})
leave['Change'] = -1
evt_df = pd.concat([enter, leave], ignore_index=True)
date_range = pd.date_range(start=min_date.normalize(), end=max_date.normalize(), freq='D')
if not evt_df.empty:
    evt_df = evt_df.groupby('Date')['Change'].sum().sort_index()
    daily_census = evt_df.reindex(date_range, fill_value=0).cumsum()
else:
    daily_census = pd.Series(0, index=date_range)

# Annual ADP Baseline
annual_cvrj_adp = daily_census.resample('YE').mean()
print("CVRJ Baseline Annual ADP (Head):")
print(annual_cvrj_adp.head())

# --- 1b. Culpeper-in-CVRJ (County Code 47 only) from same CSV — realistic add-on to CVRJ
# "Any jail" Culpeper (148.93, 185.75, ...) is total across CVRJ + RSW + Culpeper Jail; we need only CVRJ share.
df_culp = df_cvrj[df_cvrj['County Code'] == 47].dropna(subset=['Book Date']).copy()
if not df_culp.empty:
    max_c = df_culp['Release Date'].max()
    if pd.isnull(max_c) or max_c.year > 2030:
        max_c = pd.Timestamp('2025-12-31')
    min_c = df_culp['Book Date'].min()
    df_culp['Release Date'] = df_culp['Release Date'].fillna(max_c)
    df_culp['start_d'] = df_culp['Book Date'].dt.normalize()
    df_culp['end_d'] = df_culp['Release Date'].dt.normalize() + pd.Timedelta(days=1)
    df_culp = df_culp[df_culp['end_d'] > df_culp['start_d']]
    enter_c = df_culp[['start_d']].rename(columns={'start_d': 'Date'})
    enter_c['Change'] = 1
    leave_c = df_culp[['end_d']].rename(columns={'end_d': 'Date'})
    leave_c['Change'] = -1
    evt_c = pd.concat([enter_c, leave_c], ignore_index=True)
    date_range_c = pd.date_range(start=min_c.normalize(), end=max_c.normalize(), freq='D')
    evt_c = evt_c.groupby('Date')['Change'].sum().sort_index()
    daily_culp = evt_c.reindex(date_range_c, fill_value=0).cumsum()
    annual_culpeper_in_cvrj = daily_culp.resample('YE').mean()
else:
    annual_culpeper_in_cvrj = pd.Series(dtype=float)
print("Culpeper-in-CVRJ (County 47) Annual ADP (Head):")
print(annual_culpeper_in_cvrj.head() if not annual_culpeper_in_cvrj.empty else "No data")

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

# --- 3. Culpeper Census (for forecasting Culpeper-in-CVRJ)
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

# Run Culpeper-in-CVRJ Model (County 47 from CSV — realistic CVRJ add-on)
if not culp_pop.empty:
    last_c_pop = culp_pop.iloc[-1]
    prev_c_pop = culp_pop.iloc[-2]
    c_pop_2025 = last_c_pop + (last_c_pop - prev_c_pop)
    culp_pop.loc[pd.Timestamp('2025-12-31')] = c_pop_2025

# Forecast Culpeper-in-CVRJ (County 47) using CSV-derived ADP; drop 2020-2021 if desired for stability
culpeper_in_cvrj_forecast = None
if not annual_culpeper_in_cvrj.empty and not culp_pop.empty:
    culp_clean = annual_culpeper_in_cvrj[~annual_culpeper_in_cvrj.index.year.isin([2020, 2021])]
    if len(culp_clean) >= 3:
        culpeper_in_cvrj_forecast, culp_fut_pop = fit_forecast(culp_clean, culp_pop, "Culpeper_In_CVRJ")

# Save historical ADP series for capacity visualization (even if forecast fails)
annual_cvrj_adp.to_csv('forecast_annual_cvrj_adp.csv', header=['ADP'])
if not annual_culpeper_in_cvrj.empty:
    annual_culpeper_in_cvrj.to_csv('forecast_annual_culpeper_in_cvrj.csv', header=['ADP'])

# --- 5. Analysis ---
if cvrj_forecast is not None and culpeper_in_cvrj_forecast is not None:
    years_future = pd.date_range(start='2026-12-31', periods=10, freq='YE')
    print("\nForecast Results (2026-2035) — Combined = CVRJ + Culpeper-in-CVRJ (from CSV):")
    res_df = pd.DataFrame({
        'CVRJ_Baseline_NoCulpeper': cvrj_forecast.values,
        'Culpeper_In_CVRJ': culpeper_in_cvrj_forecast.values
    }, index=years_future)
    
    # Combined Load = CVRJ (No 47) + Culpeper inmates actually in CVRJ (County 47 from CSV)
    res_df['Combined_Load'] = res_df['CVRJ_Baseline_NoCulpeper'] + res_df['Culpeper_In_CVRJ']
    res_df['Capacity'] = 660
    res_df['Over_Capacity'] = res_df['Combined_Load'] > 660
    
    print(res_df)
    
    res_df.to_csv('forecast_results.csv')
    
    plt.figure(figsize=(10,6))
    plt.plot(res_df.index, res_df['Combined_Load'], label='Projected Combined Load', marker='o', linewidth=2)
    plt.plot(res_df.index, res_df['CVRJ_Baseline_NoCulpeper'], label='CVRJ Baseline (Excl. Culpeper)', linestyle='--')
    plt.plot(res_df.index, res_df['Culpeper_In_CVRJ'], label='Culpeper in CVRJ (from CSV)', linestyle='-')
    plt.axhline(y=660, color='r', linestyle='-', label='Max Capacity (660)')
    plt.title('CVRJ Capacity Forecast (Recalculated)')
    plt.ylabel('Inmate Population')
    plt.legend()
    plt.grid(True)
    plt.savefig('forecast_plot_revised.png')
    print("\nPlot saved to forecast_plot_revised.png")
