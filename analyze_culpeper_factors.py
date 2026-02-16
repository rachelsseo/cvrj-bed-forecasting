
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_DIR = "cvrj-bed-forecasting/ForecastBedData"

# --- Load Culpeper ADP (Any Jail) ---
culpeper_data = {
    '2022': 210.82,
    '2023': 200.24,
    '2024': 312.33,
    '2025': 227.81
}
# Note: 2020/2021 excluded as per instructions (COVID)
adp_series = pd.Series(culpeper_data)
adp_series.index = pd.to_datetime(adp_series.index, format='%Y') + pd.offsets.YearEnd()

# --- Load Culpeper Census ---
culp_pop_file = os.path.join(DATA_DIR, "CulpeperPopulation.csv")
try:
    temp_culp = pd.read_csv(culp_pop_file, header=4)
    culp_pop_row = temp_culp.iloc[1]
    years = [str(y) for y in range(2022, 2025)] # Match ADP years (2025 might be missing in file, check)
    # File likely goes up to 2024.
    
    pop_series = culp_pop_row[years].str.replace(',', '').astype(float)
    pop_series.index = pd.to_datetime(pop_series.index, format='%Y') + pd.offsets.YearEnd()
    
    # Extrapolate 2025 Pop if needed
    if '2025' not in culp_pop_row:
        last_val = pop_series.iloc[-1]
        prev_val = pop_series.iloc[-2]
        pop_2025 = last_val + (last_val - prev_val)
        pop_series.loc[pd.Timestamp('2025-12-31')] = pop_2025
        
except Exception as e:
    print(f"Error loading Census: {e}")
    pop_series = pd.Series()

# --- Analysis ---
print("Culpeper Analysis:")
df = pd.concat([adp_series, pop_series], axis=1).dropna()
df.columns = ['ADP', 'Population']

print(df)
corr = df.corr().iloc[0,1]
print(f"\nCorrelation between Population and ADP: {corr:.4f}")

# Calculate Growth Rates
df['ADP_Growth'] = df['ADP'].pct_change()
df['Pop_Growth'] = df['Population'].pct_change()

print("\nYear-over-Year Growth:")
print(df[['ADP_Growth', 'Pop_Growth']])

# Plot
fig, ax1 = plt.subplots(figsize=(10,6))

color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('Jail ADP', color=color)
ax1.plot(df.index, df['ADP'], color=color, marker='o', label='Jail ADP')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Census Population', color=color)  
ax2.plot(df.index, df['Population'], color=color, linestyle='--', marker='s', label='Census Population')
ax2.tick_params(axis='y', labelcolor=color)

plt.title(f'Culpeper: Jail ADP vs Population (Corr: {corr:.2f})')
fig.tight_layout()  
plt.savefig('culpeper_factors.png')
print("Plot saved to culpeper_factors.png")
