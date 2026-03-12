import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os
import random

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
    
    # EXCLUDE CULPEPER (Code 47) from Baseline
    df_cvrj_baseline = df_cvrj[df_cvrj['County Code'] != 47].copy()

    # --- 1. Compute Historical Daily Census and Annual ADP ---
    df_b = df_cvrj_baseline.dropna(subset=['Book Date']).copy()
    max_date = df_b['Release Date'].max()
    if pd.isnull(max_date) or max_date.year > 2030:
        max_date = pd.Timestamp('2025-12-31')
    min_date = df_b['Book Date'].min()
    
    df_b['Release Date'] = df_b['Release Date'].fillna(max_date)
    df_b['start_d'] = df_b['Book Date'].dt.normalize()
    df_b['end_d'] = df_b['Release Date'].dt.normalize() + pd.Timedelta(days=1)
    df_b = df_b[df_b['end_d'] > df_b['start_d']]
    
    # Calculate LOS array for future sampling (Length of stay in days, representing days in census)
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

    # --- 2. Load Population Data & Run SARIMA Forecast ---
    print("\nLoading Population Data & Running SARIMA...")
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

    combined = pd.concat([annual_cvrj_adp, total_cvrj_pop], axis=1).dropna()
    combined.columns = ['ADP', 'Pop']
    
    np.random.seed(42)  # For reproducible noise
    cvrj_forecast_vals = None
    if not combined.empty:
        model = SARIMAX(combined['ADP'], exog=combined['Pop'], order=(1,1,1), seasonal_order=(0,0,0,0))
        results = model.fit(disp=False)
        pop_y = combined['Pop'].values
        pop_x = np.arange(len(pop_y))
        z = np.polyfit(pop_x, pop_y, 1)
        p = np.poly1d(z)
        future_x = np.arange(len(pop_y), len(pop_y) + 10)
        future_pop = p(future_x)
        forecast = results.get_forecast(steps=10, exog=future_pop.reshape(-1,1))
        cvrj_forecast_vals = forecast.predicted_mean
    
    if cvrj_forecast_vals is None:
        print("SARIMA forecast failed.")
        return

    # Create daily dates for future (2026-2035)
    future_dates = pd.date_range(start='2026-01-01', end='2035-12-31', freq='D')
    
    # Create smooth daily baseline via interpolation of annual forecasts
    # Assign annual baseline correctly to year centers (July 1st) for smooth interpolation
    year_centers = [pd.Timestamp(year=y, month=7, day=1) for y in range(2025, 2036)]
    # Use 2025 true mean as anchor
    anchor_2025 = combined['ADP'].iloc[-1] if not combined.empty else annual_cvrj_adp.iloc[-1]
    baseline_vals = [anchor_2025] + list(cvrj_forecast_vals.values)
    baseline_series = pd.Series(baseline_vals, index=year_centers)
    
    # Reindex to all daily dates spanning from mid-2025 to end-2035, interpolate, then slice to target
    full_interpolated = baseline_series.reindex(pd.date_range(start=year_centers[0], end='2035-12-31', freq='D'))
    full_interpolated = full_interpolated.interpolate(method='linear')
    smooth_future_baseline = full_interpolated.reindex(future_dates)
    smooth_future_baseline = smooth_future_baseline.bfill().ffill() # Handle edges

    # --- 3. Simulate Future Bookings for Realistic Daily Noise ---
    print("\nSimulating future bookings utilizing historical LOS...")
    # Initialize array to track simulated census
    simulated_census = np.zeros(len(future_dates))
    
    # We simulate arrivals per day. Lambda = Target ADP / Mean LOS
    # Use the annual forecast to adjust the daily arrival rate
    for year in range(2026, 2036):
        year_idx = year - 2026
        target_adp = cvrj_forecast_vals.iloc[year_idx]
        daily_arrival_rate = target_adp / mean_los
        
        # Determine day indices for this year
        year_mask = (future_dates.year == year)
        days_in_year = np.where(year_mask)[0]
        
        for i in days_in_year:
            # How many arrivals today?
            arrivals = np.random.poisson(daily_arrival_rate)
            if arrivals > 0:
                # Sample 'arrivals' number of length_of_stay from history
                sampled_los = np.random.choice(historical_los, size=arrivals, replace=True)
                for stay in sampled_los:
                    # Inmate is in census from day i to i + stay - 1
                    end_idx = min(i + stay, len(future_dates))
                    simulated_census[i:end_idx] += 1
    
    # Note: A simulation starting cold on 2026-01-01 will "spin up" and initially be 0.
    # To fix this, we can either simulate a burn-in period, or just mean-center the simulation noise!
    # Because we ONLY want the noise profile!
    simulated_series = pd.Series(simulated_census, index=future_dates)
    
    # Group by year and calculate the difference between the daily simulation and its annual mean.
    # This extracts the pure "noise" waveform that contains the realistic LOS autocorrelation.
    simulated_annual_means = simulated_series.resample('Y').mean()
    
    noise_series = pd.Series(index=future_dates, dtype=float)
    for year in range(2026, 2036):
        year_mask = (future_dates.year == year)
        year_sim = simulated_series[year_mask]
        year_sim_mean = simulated_annual_means.loc[pd.Timestamp(f"{year}-12-31")]
        # The noise is how far the daily varies from the annual mean
        noise_series[year_mask] = year_sim - year_sim_mean

    # Some spin-up artifact might exist in early 2026 noise since census built from 0.
    # To avoid using the actual zero-spinup in Jan 2026, let's copy the noise from 2027 to 2026
    # (matching up day-of-year roughly)
    days_in_2026 = noise_series[noise_series.index.year == 2026].index
    safe_noise_2027 = noise_series[noise_series.index.year == 2027].values[:len(days_in_2026)]
    noise_series.loc[days_in_2026] = safe_noise_2027

    # --- 4. Final Projection: Baseline + Noise ---
    # According to prompt: "This should just be cvrj baseline + noise"
    final_daily_projection = smooth_future_baseline + noise_series

    # Calculate final annual ADP to verify it precisely matches the SARIMA baseline
    final_annual_adp = final_daily_projection.resample('Y').mean()

    # --- 5. Export and Plot ---
    # Merge history with future for plotting
    history_to_plot = daily_census[daily_census.index.year >= 2022] # Plot last few years of history
    
    plt.figure(figsize=(15, 8))
    
    # History
    plt.plot(history_to_plot.index, history_to_plot.values, label='Historical CVRJ Daily Census (Exclude Culpeper)', color='gray', alpha=0.7, linewidth=1)
    
    # Future Projection
    plt.plot(final_daily_projection.index, final_daily_projection.values, label='Projected CVRJ Daily Census (Baseline + Simulation Noise)', color='dodgerblue', linewidth=1, alpha=0.9)
    
    # Future Smooth Baseline
    plt.plot(smooth_future_baseline.index, smooth_future_baseline.values, label='SARIMA Smooth Baseline', color='red', linestyle='--', linewidth=2)
    
    # Add Annual Means as scatter points for baseline verification
    plt.scatter([pd.Timestamp(year=y, month=7, day=1) for y in range(2026, 2036)], cvrj_forecast_vals.values, color='darkred', s=50, zorder=5, label='SARIMA Annual Target')

    # Add Max Capacity line
    plt.axhline(660, color='orange', linestyle=':', label='Max Capacity (Total 660)', linewidth=2)
    
    plt.title('CVRJ Bed Forecasting: 10-Year Daily Projection using Historical LOS and Arrival Simulation', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Average Daily Population (ADP)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_img = 'visuals/baseline_projected_daily_adp.png'
    plt.savefig(out_img, dpi=300)
    print(f"\nPlot saved to {out_img}")
    
    # Save the daily projection data
    out_csv = '../data/outputs/baseline_projected_daily_adp.csv'
    final_daily_projection.to_frame(name='Projected_Daily_ADP').to_csv(out_csv)
    print(f"Daily projection data saved to {out_csv}")
    
    # Print the verification summary
    print("\n--- Verification: Annual Averages of Daily Projection vs SARIMA Baseline ---")
    years = range(2026, 2036)
    for i, year in enumerate(years):
        proj_mean = final_annual_adp.iloc[i]
        base_val = cvrj_forecast_vals.iloc[i]
        diff = proj_mean - base_val
        print(f"{year}: Projected Mean = {proj_mean:.1f} | SARIMA Target = {base_val:.1f} | Diff = {diff:.2f}")

if __name__ == "__main__":
    main()
