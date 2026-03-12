import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os

# Define paths (rooted at project, not CWD)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_ROOT, "data", "raw")
CVRJ_FILE = os.path.join(_ROOT, "data", "processed", "cvrj_dataset_v2.csv")

def main():
    print("Loading CVRJ Data...")
    df_cvrj = pd.read_csv(CVRJ_FILE)
    df_cvrj['Book Date'] = pd.to_datetime(df_cvrj['Book Date'], errors='coerce')
    df_cvrj['Release Date'] = pd.to_datetime(df_cvrj['Release Date'], errors='coerce')

    # Filter for relevant years (2012-2025)
    df_cvrj = df_cvrj[df_cvrj['Book Date'].dt.year >= 2012]
    
    # EXCLUDE CULPEPER (Code 47) from Baseline to avoid double counting
    df_cvrj_baseline = df_cvrj[df_cvrj['County Code'] != 47].copy()

    # --- 1. Compute Historical Daily Census and Annual ADP for Baseline ---
    df_b = df_cvrj_baseline.dropna(subset=['Book Date']).copy()
    max_date = df_b['Release Date'].max()
    if pd.isnull(max_date) or max_date.year > 2030:
        max_date = pd.Timestamp('2025-12-31')
    min_date = df_b['Book Date'].min()
    
    df_b['Release Date'] = df_b['Release Date'].fillna(max_date)
    df_b['start_d'] = df_b['Book Date'].dt.normalize()
    df_b['end_d'] = df_b['Release Date'].dt.normalize() + pd.Timedelta(days=1)
    df_b = df_b[df_b['end_d'] > df_b['start_d']]
    
    # Extract LOS for future Monte Carlo simulation
    df_b['census_days'] = (df_b['end_d'] - df_b['start_d']).dt.days
    historical_los = df_b['census_days'].values
    mean_los = np.mean(historical_los)
    print(f"Mean Historical Length of Stay: {mean_los:.2f} days")

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
    annual_cvrj_adp = daily_census.resample('Y').mean()

    # --- 2. Load Population Data & Run SARIMA Forecast for BASELINE ---
    print("\nLoading Population Data & Running SARIMA (CVRJ Baseline)...")
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
    
    if not total_cvrj_pop.empty:
        last_pop = total_cvrj_pop.iloc[-1]
        prev_pop = total_cvrj_pop.iloc[-2]
        pop_2025 = last_pop + (last_pop - prev_pop)
        total_cvrj_pop.loc[pd.Timestamp('2025-12-31')] = pop_2025

    combined_base = pd.concat([annual_cvrj_adp, total_cvrj_pop], axis=1).dropna()
    combined_base.columns = ['ADP', 'Pop']
    
    np.random.seed(42)  # For reproducible noise
    cvrj_forecast_vals = None
    if not combined_base.empty:
        model = SARIMAX(combined_base['ADP'], exog=combined_base['Pop'], order=(1,1,1), seasonal_order=(0,0,0,0))
        results = model.fit(disp=False)
        pop_y = combined_base['Pop'].values
        pop_x = np.arange(len(pop_y))
        z = np.polyfit(pop_x, pop_y, 1)
        p = np.poly1d(z)
        future_x = np.arange(len(pop_y), len(pop_y) + 10)
        future_pop_base = p(future_x)
        forecast = results.get_forecast(steps=10, exog=future_pop_base.reshape(-1,1))
        cvrj_forecast_vals = forecast.predicted_mean
    
    if cvrj_forecast_vals is None:
        print("SARIMA forecast failed.")
        return

    # --- 3. Forecast Target ADP for TOTAL CULPEPER ---
    print("\nForecasting Target ADP for Total Culpeper...")
    # These are the "Held for Culpeper County" figures provided
    culpeper_data = {
        '2022': 210.82,
        '2023': 200.24,
        '2024': 312.33,
        '2025': 227.81
    }
    culp_adp_series = pd.Series(culpeper_data)
    culp_adp_series.index = pd.to_datetime(culp_adp_series.index, format='%Y') + pd.offsets.YearEnd()

    culp_pop_file = os.path.join(DATA_DIR, "CulpeperPopulation.csv")
    try:
        temp_culp = pd.read_csv(culp_pop_file, header=4)
        culp_pop_row = temp_culp.iloc[1]
        years_culp = [str(y) for y in range(2022, 2025)] 
        culp_pop_series = culp_pop_row[years_culp].str.replace(',', '').astype(float)
        culp_pop_series.index = pd.to_datetime(culp_pop_series.index, format='%Y') + pd.offsets.YearEnd()
        
        # Extrapolate 2025
        if '2025' not in culp_pop_row:
            last_val = culp_pop_series.iloc[-1]
            prev_val = culp_pop_series.iloc[-2]
            pop_2025 = last_val + (last_val - prev_val)
            culp_pop_series.loc[pd.Timestamp('2025-12-31')] = pop_2025
    except Exception as e:
        print(f"Error loading Culpeper Census: {e}")
        culp_pop_series = pd.Series()

    df_c = pd.concat([culp_adp_series, culp_pop_series], axis=1).dropna()
    df_c.columns = ['ADP', 'Pop']

    # Extrapolate population to 2035 
    pop_y = df_c['Pop'].values
    pop_x = np.arange(len(pop_y))
    z_pop = np.polyfit(pop_x, pop_y, 1)
    p_pop = np.poly1d(z_pop)
    future_x = np.arange(len(pop_y), len(pop_y) + 10)
    future_pop_culp = p_pop(future_x)
    
    # Simple ratio forecast: mean(ADP / Population) * Future_Population
    # Because 4 points with a massive spike in 2024 makes pure linear regression unstable.
    mean_ratio = (df_c['ADP'] / df_c['Pop']).mean()
    culp_target_future = pd.Series(future_pop_culp * mean_ratio, 
                                   index=pd.date_range(start='2026-12-31', periods=10, freq='Y'))
    print("Culpeper Future Target ADPs:")
    print(culp_target_future)

    # --- 4. Prepare Smooth Baselines & Synthesize Noise ---
    future_dates = pd.date_range(start='2026-01-01', end='2035-12-31', freq='D')
    year_centers = [pd.Timestamp(year=y, month=7, day=1) for y in range(2025, 2036)]
    
    def create_smooth_line(anchor_val, future_targets):
        b_vals = [anchor_val] + list(future_targets.values)
        b_ser = pd.Series(b_vals, index=year_centers)
        interp = b_ser.reindex(pd.date_range(start=year_centers[0], end='2035-12-31', freq='D'))
        interp = interp.interpolate(method='linear').reindex(future_dates)
        return interp.bfill().ffill()
    
    anchor_base_2025 = combined_base['ADP'].iloc[-1] if not combined_base.empty else annual_cvrj_adp.iloc[-1]
    smooth_base = create_smooth_line(anchor_base_2025, cvrj_forecast_vals)
    
    anchor_culp_2025 = df_c['ADP'].iloc[-1]
    smooth_culp = create_smooth_line(anchor_culp_2025, culp_target_future)

    def generate_noise(target_annual_series, name_prefix=""):
        print(f"Simulating daily arrivals to extract noise for {name_prefix}...")
        simulated_census = np.zeros(len(future_dates))
        for year in range(2026, 2036):
            year_idx = year - 2026
            target_adp = target_annual_series.iloc[year_idx]
            daily_arrival_rate = target_adp / mean_los
            year_mask = (future_dates.year == year)
            days_in_year = np.where(year_mask)[0]
            
            for i in days_in_year:
                arrivals = np.random.poisson(daily_arrival_rate)
                if arrivals > 0:
                    sampled_los = np.random.choice(historical_los, size=arrivals, replace=True)
                    for stay in sampled_los:
                        end_idx = min(i + stay, len(future_dates))
                        simulated_census[i:end_idx] += 1
                        
        sim_series = pd.Series(simulated_census, index=future_dates)
        sim_means = sim_series.resample('Y').mean()
        
        noise_series = pd.Series(index=future_dates, dtype=float)
        for year in range(2026, 2036):
            year_mask = (future_dates.year == year)
            year_sim = sim_series[year_mask]
            year_mean = sim_means.loc[pd.Timestamp(f"{year}-12-31")]
            noise_series[year_mask] = year_sim - year_mean
            
        # Copy 2027 noise over 2026 to fix the cold-start spin-up zeros in Jan 2026
        days_in_2026 = noise_series[noise_series.index.year == 2026].index
        safe_noise_2027 = noise_series[noise_series.index.year == 2027].values[:len(days_in_2026)]
        noise_series.loc[days_in_2026] = safe_noise_2027
        return noise_series

    # Generate distinct, randomized noise tracks for each
    noise_base = generate_noise(cvrj_forecast_vals, "CVRJ Baseline")
    noise_culp = generate_noise(culp_target_future, "Total Culpeper")

    # Combine
    projected_daily_base = smooth_base + noise_base
    projected_daily_culp = smooth_culp + noise_culp
    final_combined_daily = projected_daily_base + projected_daily_culp
    
    # Final smooth combined for reference
    smooth_combined = smooth_base + smooth_culp

    # --- 5. Export and Plot ---
    # Merge history with future for plotting
    history_base_plot = daily_census[daily_census.index.year >= 2022] 
    
    plt.figure(figsize=(15, 8))
    
    # History Baseline
    plt.plot(history_base_plot.index, history_base_plot.values, label='Historical CVRJ Baseline (Excl. 47)', color='gray', alpha=0.5, linewidth=1)
    
    # Culpeper History is just 4 annual bars/points, we'll plot them as horizontal levels across the year
    for year_str in culpeper_data.keys():
        year = int(year_str)
        val = culpeper_data[year_str]
        year_dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
        plt.plot(year_dates, [val]*len(year_dates), color='orange', alpha=0.4, linewidth=2)
    # Re-plot last year with label
    plt.plot(year_dates, [val]*len(year_dates), color='orange', alpha=0.4, linewidth=2, label='Culpeper Historical Target (Annual Avg)')

    # Add historical stacked approximation:
    approx_combined_hist = pd.Series(index=history_base_plot.index, dtype=float)
    for date in history_base_plot.index:
        y_str = str(date.year)
        c_val = culpeper_data.get(y_str, 0) # If no data (e.g. before 2022 drops to 0)
        approx_combined_hist[date] = history_base_plot[date] + c_val
    approx_combined_hist = approx_combined_hist[approx_combined_hist.index.year >= 2022]
    
    plt.plot(approx_combined_hist.index, approx_combined_hist.values, label='Approx Historical Combined load', color='darkgray', alpha=0.8, linewidth=1.5)

    # Future Projection (Combined Noise)
    plt.plot(final_combined_daily.index, final_combined_daily.values, label='Projected Combined Daily Load (Baseline + Culpeper)', color='midnightblue', linewidth=1, alpha=0.9)
    
    # Future Smooth Baselines
    plt.plot(smooth_base.index, smooth_base.values, label='SARIMA CVRJ Smooth Target', color='dodgerblue', linestyle='--', linewidth=2)
    plt.plot(smooth_culp.index, smooth_culp.values, label='Culpeper Linear Smooth Target', color='darkorange', linestyle='--', linewidth=2)
    plt.plot(smooth_combined.index, smooth_combined.values, label='Total Combined Smooth Target', color='red', linestyle='-', linewidth=2)

    # Add max capacity
    plt.axhline(660, color='darkred', linestyle=':', label='Max Capacity (Total: 660)', linewidth=3)
        
    plt.title('CVRJ Capacity Forecast: 10-Year Daily Projection of Combined Load (Real Noise Simulated)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Average Daily Population (ADP)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_img = 'visuals/combined_daily_projection.png'
    plt.savefig(out_img, dpi=300)
    print(f"\nPlot saved to {out_img}")
    
    # Save the daily projection data
    out_csv = '../data/outputs/projected_combined_adp.csv'
    res_df = pd.DataFrame({
        'CVRJ_Baseline_Noisy': projected_daily_base,
        'Total_Culpeper_Noisy': projected_daily_culp,
        'Combined_Total_Daily': final_combined_daily,
        'Combined_Smooth_Target': smooth_combined
    })
    res_df.to_csv(out_csv)
    print(f"Daily projection data saved to {out_csv}")
    
    print("\n--- Verification: Annual Averages of Combined Projection ---")
    final_combined_annual = final_combined_daily.resample('Y').mean()
    cvrj_base_annual = projected_daily_base.resample('Y').mean()
    culp_annual = projected_daily_culp.resample('Y').mean()
    
    years = range(2026, 2036)
    for i, year in enumerate(years):
        proj_mean = final_combined_annual.iloc[i]
        cvrj_part = cvrj_base_annual.iloc[i]
        culp_part = culp_annual.iloc[i]
        print(f"{year}: Total Combined = {proj_mean:.1f} (CVRJ: {cvrj_part:.1f} + Culpeper: {culp_part:.1f})")

if __name__ == "__main__":
    main()
