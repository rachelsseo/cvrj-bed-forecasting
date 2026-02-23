"""
CVRJ capacity forecast: projected bed need WITH vs WITHOUT Culpeper County,
with max capacity (660 beds) shown. Includes a clear explanation of how the
forecast models are built.

Uses forecast_annual_cvrj_adp.csv and forecast_results.csv if present (from
run_forecast.py); otherwise runs the same logic inline (slower).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MAX_CAPACITY = 660
FORECAST_ADP_CSV = "../data/outputs/forecast_annual_cvrj_adp.csv"
FORECAST_RESULTS_CSV = "../data/outputs/forecast_results.csv"
CULPEPER_IN_CVRJ_CSV = "../data/outputs/forecast_annual_culpeper_in_cvrj.csv"  # County 47 from cvrj_dataset_v2.csv
FORECAST_START_YEAR = 2026


def load_forecast_data():
    """
    Load historical CVRJ ADP, Culpeper-in-CVRJ (County 47 from CSV), and forecast results.
    Returns (annual_cvrj_adp, res_df, annual_culpeper_in_cvrj) or (None, None, None) if files missing.
    """
    if not os.path.exists(FORECAST_ADP_CSV) or not os.path.exists(FORECAST_RESULTS_CSV):
        return None, None, None
    adp = pd.read_csv(FORECAST_ADP_CSV, index_col=0, parse_dates=True).squeeze()
    adp = adp.loc[adp.index <= pd.Timestamp('2025-12-31')]
    res = pd.read_csv(FORECAST_RESULTS_CSV, index_col=0, parse_dates=True)
    culp = None
    if os.path.exists(CULPEPER_IN_CVRJ_CSV):
        culp = pd.read_csv(CULPEPER_IN_CVRJ_CSV, index_col=0, parse_dates=True).squeeze()
        culp = culp.loc[culp.index <= pd.Timestamp('2025-12-31')]
    return adp, res, culp


def build_combined_historical(annual_cvrj_adp, annual_culpeper_in_cvrj):
    """
    Historical combined = CVRJ baseline + Culpeper-in-CVRJ (County 47 from CSV).
    Realistic total: only inmates actually in CVRJ per the booking data, not 'any jail' totals.
    """
    combined = annual_cvrj_adp.copy()
    if annual_culpeper_in_cvrj is not None and not annual_culpeper_in_cvrj.empty:
        for t in combined.index:
            if t in annual_culpeper_in_cvrj.index:
                combined.loc[t] = annual_cvrj_adp.loc[t] + annual_culpeper_in_cvrj.loc[t]
    return combined


def plot_capacity(ax, annual_cvrj_adp, years_future, cvrj_forecast_vals, combined_forecast,
                  combined_historical, draw_forecast_marker=True):
    """Draw capacity plot onto ax: blue = CVRJ (hist + forecast), orange = combined (hist + forecast), red = 660."""
    hist_years = annual_cvrj_adp.index
    # Ensure chronological order: historical then forecast
    all_years = list(hist_years) + list(years_future)
    all_vals_cvrj = np.concatenate([annual_cvrj_adp.values, cvrj_forecast_vals])
    all_vals_combined = np.concatenate([combined_historical.values, combined_forecast])

    # Blue: one continuous line 2012 → end of forecast (CVRJ baseline excluding Culpeper)
    ax.plot(all_years, all_vals_cvrj, 'o-', color='steelblue', markersize=5, linewidth=1.5,
            label='CVRJ baseline (excl. Culpeper)')

    # Orange: one continuous line 2012 → end of forecast (combined; 2020+ adds Culpeper, then forecast)
    ax.plot(all_years, all_vals_combined, 's-', color='darkorange', markersize=5, linewidth=2,
            label='CVRJ + Culpeper (combined)')

    # Max capacity
    ax.axhline(y=MAX_CAPACITY, color='red', linestyle='-', linewidth=2,
               label=f'Max capacity ({MAX_CAPACITY} beds)')

    # Mark when forecast starts
    if draw_forecast_marker:
        ax.axvline(x=pd.Timestamp(f'{FORECAST_START_YEAR}-01-01'), color='gray', linestyle='--', linewidth=1.5,
                   label='Forecast begins', zorder=0)
        ymin, ymax = ax.get_ylim()
        ax.axvspan(pd.Timestamp(f'{FORECAST_START_YEAR}-01-01'), all_years[-1], alpha=0.08, color='gray', zorder=0)
        ax.set_ylim(ymin, ymax)
        ax.text(0.28, 0.97, 'Historical', transform=ax.transAxes, ha='center', va='top', fontsize=9, color='gray')
        ax.text(0.82, 0.97, 'Forecast', transform=ax.transAxes, ha='center', va='top', fontsize=9, color='gray')

    ax.set_xlabel('Year')
    ax.set_ylabel('Average daily population (beds)')
    ax.set_ylim(0, None)
    ax.set_xlim(pd.Timestamp('2012-01-01'), pd.Timestamp('2036-01-01'))
    ax.grid(True, alpha=0.5)
    ax.legend(loc='upper left', fontsize=9)


def main():
    annual_cvrj_adp, res_df, annual_culpeper_in_cvrj = load_forecast_data()
    if annual_cvrj_adp is None or res_df is None:
        print("Run run_forecast.py first to generate forecast_annual_cvrj_adp.csv and forecast_results.csv")
        print("Then run this script again to create the capacity visualization.")
        return
    if annual_culpeper_in_cvrj is None or annual_culpeper_in_cvrj.empty:
        print("Warning: forecast_annual_culpeper_in_cvrj.csv not found; combined line = CVRJ only (re-run run_forecast.py).")

    years_future = res_df.index
    cvrj_forecast_vals = res_df['CVRJ_Baseline_NoCulpeper'].values
    combined_forecast = res_df['Combined_Load'].values
    combined_historical = build_combined_historical(annual_cvrj_adp, annual_culpeper_in_cvrj)

    # --- Figure 1: Capacity with vs without Culpeper + 660 line ---
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_capacity(ax, annual_cvrj_adp, years_future, cvrj_forecast_vals, combined_forecast,
                  combined_historical, draw_forecast_marker=True)
    ax.set_title('CVRJ bed need: with vs without Culpeper County (max capacity = 660 beds)')
    plt.tight_layout()
    plt.savefig('visuals/capacity_forecast_with_and_without_culpeper.png', dpi=150, bbox_inches='tight')
    print("Saved: visuals/capacity_forecast_with_and_without_culpeper.png")
    plt.close()

    # --- Figure 2: Same plot + methodology text ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    plot_capacity(ax2, annual_cvrj_adp, years_future, cvrj_forecast_vals, combined_forecast,
                  combined_historical, draw_forecast_marker=True)
    ax2.set_title('CVRJ bed need: with vs without Culpeper County (max capacity = 660 beds)')

    methodology = (
        "How the forecasts are created:\n\n"
        "1. Historical ADP: From cvrj_dataset_v2.csv (Book/Release dates). Daily census =\n"
        "   cumsum of +1 on book, -1 day after release; annual ADP = mean per year.\n"
        "   CVRJ baseline excludes County 47. Culpeper-in-CVRJ = County 47 only (realistic\n"
        "   add-on; not 'any jail' total, which includes RSW and Culpeper Jail).\n\n"
        "2. Forecast: SARIMAX(ADP, exog=Population), order (1,1,1). CVRJ uses 5-county\n"
        "   population; Culpeper-in-CVRJ uses Culpeper county population.\n\n"
        "3. Combined load: CVRJ baseline + Culpeper-in-CVRJ (from CSV). If combined > 660,\n"
        "   CVRJ would be over capacity if Culpeper joins."
    )
    ax2.text(0.98, 0.98, methodology, transform=ax2.transAxes, fontsize=7,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
    plt.tight_layout()
    plt.savefig('visuals/capacity_forecast_with_methodology.png', dpi=150, bbox_inches='tight')
    print("Saved: visuals/capacity_forecast_with_methodology.png")
    plt.close()

    print("\nDone. Red line = 660-bed capacity. Orange = CVRJ + Culpeper-in-CVRJ (from CSV); blue = CVRJ only.")


if __name__ == "__main__":
    main()
