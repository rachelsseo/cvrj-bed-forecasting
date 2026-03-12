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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)

MAX_CAPACITY = 660
FORECAST_ADP_CSV = os.path.join(_ROOT, "data", "outputs", "forecast_annual_cvrj_adp.csv")
FORECAST_RESULTS_CSV = os.path.join(_ROOT, "data", "outputs", "forecast_results.csv")
CULPEPER_IN_CVRJ_CSV = os.path.join(_ROOT, "data", "outputs", "forecast_annual_culpeper_in_cvrj.csv")
FORECAST_START_YEAR = 2026
FONT_TITLE = 18
FONT_LABEL = 15
FONT_LEGEND = 12
FONT_ANNO = 14


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
                  annual_culpeper_in_cvrj, culpeper_forecast_vals, combined_se=None,
                  annual_cvrj_excl47_for_combined=None, draw_forecast_marker=True, show_legend=True):
    """Draw capacity plot: CVRJ (blue, incl. 47); Culpeper all jails (green); Combined = CVRJ excl. 47 + Culpeper (orange)."""
    # Trim historical portion to start at 2016
    hist_years_full = annual_cvrj_adp.index
    cutoff = pd.Timestamp('2015-12-31')
    hist_years = hist_years_full[hist_years_full >= cutoff]
    annual_cvrj_adp = annual_cvrj_adp.loc[hist_years]
    if annual_culpeper_in_cvrj is not None and not annual_culpeper_in_cvrj.empty:
        annual_culpeper_in_cvrj = annual_culpeper_in_cvrj.reindex(hist_years).fillna(0.0)

    all_years = list(hist_years) + list(years_future)
    x_forecast = pd.Timestamp(f'{FORECAST_START_YEAR}-01-01')

    # CVRJ baseline (incl. County 47): entire plot
    all_vals_cvrj = np.concatenate([annual_cvrj_adp.values, cvrj_forecast_vals])
    ax.plot(all_years, all_vals_cvrj, 'o-', color='steelblue', markersize=5, linewidth=1.5,
            label='CVRJ (incl. County 47)')

    # Culpeper only: historical total ADP (all jails) + optional forecast
    if annual_culpeper_in_cvrj is not None and not annual_culpeper_in_cvrj.empty:
        # Align historical Culpeper to hist_years (fill missing with NaN or 0)
        culp_hist_vals = np.array([
            annual_culpeper_in_cvrj.loc[t] if t in annual_culpeper_in_cvrj.index else np.nan
            for t in hist_years
        ])
        culp_hist_vals = np.nan_to_num(culp_hist_vals, nan=0.0)
        if culpeper_forecast_vals is not None:
            all_vals_culp = np.concatenate([culp_hist_vals, culpeper_forecast_vals])
            y_vals_culp = all_vals_culp
            x_vals_culp = all_years
        else:
            # Historical only
            y_vals_culp = culp_hist_vals
            x_vals_culp = hist_years
        ax.plot(x_vals_culp, y_vals_culp, '^-', color='green', markersize=4, linewidth=1.2,
                label='Culpeper (all jails, historical ADP)')

    # Ensure orange line starts exactly at forecast start year (branching from combined historical)
    start_date = pd.Timestamp(f'{FORECAST_START_YEAR}-01-01')

    last_hist_year = hist_years[-1]
    last_hist_cvrj = annual_cvrj_adp.loc[last_hist_year]
    if annual_culpeper_in_cvrj is not None and not annual_culpeper_in_cvrj.empty and last_hist_year in annual_culpeper_in_cvrj.index:
        last_hist_culp = annual_culpeper_in_cvrj.loc[last_hist_year]
    else:
        last_hist_culp = 0.0
    # Orange anchor = (CVRJ excl. 47 at 2025) + (Culpeper all jails at 2025) when available
    if annual_cvrj_excl47_for_combined is not None and not annual_cvrj_excl47_for_combined.empty and last_hist_year in annual_cvrj_excl47_for_combined.index:
        last_hist_combined = annual_cvrj_excl47_for_combined.loc[last_hist_year] + last_hist_culp
    else:
        last_hist_combined = last_hist_cvrj + last_hist_culp

    years_future_trim = years_future[years_future >= start_date]
    combined_trim = combined_forecast[years_future >= start_date]

    years_combo_plot = [last_hist_year] + list(years_future_trim)
    vals_combo_plot = [last_hist_combined] + list(combined_trim)

    ax.plot(years_combo_plot, vals_combo_plot,
            's-', color='darkorange',
            markersize=5, linewidth=2,
            label='CVRJ + Culpeper (combined)')

    # One-sigma band around the combined forecast (optional)
    if combined_se is not None:
        se_trim = combined_se[years_future >= start_date]
        se_combo_plot = [0.0] + list(se_trim)  # 0 variance at the anchor point
        upper_bound = np.array(vals_combo_plot) + np.array(se_combo_plot)
        lower_bound = np.array(vals_combo_plot) - np.array(se_combo_plot)
        ax.fill_between(years_combo_plot, lower_bound, upper_bound,
                        color='darkorange', alpha=0.15)

    # Maximum capacity line
    ax.axhline(y=MAX_CAPACITY, color='red', linestyle='-', linewidth=2,
            label='Maximum Capacity (660 beds)')

    # Label the red line directly
    ax.text(pd.Timestamp('2035-01-01'), MAX_CAPACITY + 5,
            'Maximum Capacity = 660',
            color='red',
            fontsize=FONT_ANNO,
            ha='right',
            va='bottom')
    # Forecast demarcation: vertical line + shaded region + arrows with labels
    if draw_forecast_marker:
        ax.axvline(x=x_forecast, color='gray', linestyle='--', linewidth=1.5, zorder=0)
        ymin, ymax = ax.get_ylim()
        ax.axvspan(x_forecast, all_years[-1], alpha=0.08, color='gray', zorder=0)
        ax.set_ylim(ymin, ymax)
        # Place the \"Historical\" / \"Forecast\" labels well away from the capacity line
        # so they don't overlap the red 660-bed line.
        y_arrow = MAX_CAPACITY + 50  # clearly below the capacity line

        offset_forecast = pd.DateOffset(years=4)
        offset_historical = pd.DateOffset(years=8)


        # Forecast arrow (pointing right from dashed line)
        ax.annotate('Forecast',
                   xy=(x_forecast, y_arrow),
                   xytext=(x_forecast + offset_forecast, y_arrow),
                   fontsize=20,
                   ha='left', va='center',
                   color='black')
                   #arrowprops=dict(arrowstyle='<-', color='gray', lw=1.5))

        # Historical arrow (pointing left toward dashed line)
        ax.annotate('Historical',
                    xy=(x_forecast, y_arrow),
                    xytext=(x_forecast - offset_historical, y_arrow),
                    fontsize=20,
                    ha='left', va='center',
                    color='black')
                    #arrowprops=dict(arrowstyle='<-', color='gray', lw=1.5))

    ax.set_xlabel('Year', fontsize=FONT_LABEL)
    ax.set_ylabel('Average Daily Population (Beds)', fontsize=FONT_LABEL)
    
    # Direct labels ON the lines at the end of the forecast area (2035)
    last_year_forecast = years_future[-1]
    
    # 1. Orange Combined Line (Above)
    last_val_combo = vals_combo_plot[-1]
    ax.text(last_year_forecast - pd.DateOffset(months=3), last_val_combo + 15, 
            'Combined\n(CVRJ + Culpeper)', 
            color='darkorange', fontsize=FONT_ANNO, fontweight='bold', 
            ha='right', va='bottom')
            
    # 2. Blue CVRJ Baseline (below the blue line, right/forecast side)
    last_val_cvrj = cvrj_forecast_vals[-1]
    ax.text(last_year_forecast - pd.DateOffset(months=3), last_val_cvrj - 55, 
            'CVRJ Baseline', 
            color='steelblue', fontsize=FONT_ANNO, fontweight='bold', 
            ha='right', va='top')
            
    # 3. Green Culpeper (all jails) — right side in forecast section
    if annual_culpeper_in_cvrj is not None and not annual_culpeper_in_cvrj.empty:
        y_culp = culpeper_forecast_vals[-1] if culpeper_forecast_vals is not None else annual_culpeper_in_cvrj.iloc[-1]
        ax.text(last_year_forecast - pd.DateOffset(months=3), y_culp - 55, 
                'Culpeper (all jails)', 
                color='green', fontsize=FONT_ANNO, fontweight='bold', 
                ha='right', va='top')
                
    ax.tick_params(axis='both', labelsize=FONT_LEGEND)
    ax.set_ylim(0, None)
    ax.set_xlim(pd.Timestamp('2016-01-01'), pd.Timestamp('2036-01-01'))
    ax.grid(True, alpha=0.5)
    # Optional legend (can be disabled for cleaner main figure)
    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.87),
                  fontsize=FONT_LEGEND, framealpha=0.95)


def main():
    annual_cvrj_adp, res_df, _ = load_forecast_data()
    if annual_cvrj_adp is None or res_df is None:
        print("Run run_forecast.py first to generate forecast_annual_cvrj_adp.csv and forecast_results.csv")
        print("Then run this script again to create the capacity visualization.")
        return

    # Load historical + forecast ADP for inmates held for Culpeper County (any jail), 2014–2035.
    culpeper_forecast_path = os.path.join(_ROOT, "data", "outputs", "culpeper_total_adp_forecast.csv")
    if os.path.exists(culpeper_forecast_path):
        annual_culpeper_total_adp = pd.read_csv(culpeper_forecast_path, index_col=0, parse_dates=True).squeeze()
    else:
        annual_culpeper_total_adp = pd.Series(dtype=float)
    # CVRJ excl. 47 (for orange-line anchor and combined = this + Culpeper all jails).
    cvrj_excl47_path = os.path.join(_ROOT, "data", "outputs", "forecast_annual_cvrj_excl47.csv")
    annual_cvrj_excl47 = None
    if os.path.exists(cvrj_excl47_path):
        annual_cvrj_excl47 = pd.read_csv(cvrj_excl47_path, index_col=0, parse_dates=True).squeeze()
        annual_cvrj_excl47 = annual_cvrj_excl47[annual_cvrj_excl47.index <= pd.Timestamp('2025-12-31')]

    years_future = res_df.index
    # Blue line: CVRJ incl. County 47 (from CSV)
    cvrj_forecast_vals = res_df['CVRJ_Baseline_NoCulpeper'].values
    # Combined (orange) = CVRJ excl. 47 + Culpeper all jails (2014-2025 + forecast from culpeper_total_adp_forecast.csv).
    culpeper_hist = annual_culpeper_total_adp[annual_culpeper_total_adp.index <= pd.Timestamp('2025-12-31')]
    culpeper_future = annual_culpeper_total_adp.reindex(years_future).ffill()
    annual_culpeper_total_adp = culpeper_hist
    culpeper_forecast_vals = culpeper_future.values
    if 'CVRJ_Excl47_Forecast' in res_df.columns:
        combined_forecast = res_df['CVRJ_Excl47_Forecast'].values + culpeper_forecast_vals
        combined_se = res_df['CVRJ_Excl47_SE'].values if 'CVRJ_Excl47_SE' in res_df.columns else None
    else:
        combined_forecast = cvrj_forecast_vals + culpeper_forecast_vals
        combined_se = res_df['CVRJ_SE'].values if 'CVRJ_SE' in res_df.columns else None

    # --- Figure 1: Capacity with vs without Culpeper + 660 line (main visual, no title/legend) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_capacity(ax, annual_cvrj_adp, years_future, cvrj_forecast_vals, combined_forecast,
                  annual_culpeper_total_adp, culpeper_forecast_vals,
                  combined_se=combined_se, annual_cvrj_excl47_for_combined=annual_cvrj_excl47,
                  draw_forecast_marker=True, show_legend=False)
    plt.tight_layout()
    out1 = os.path.join(_ROOT, 'visuals', 'capacity_forecast_with_and_without_culpeper.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"Saved: {out1}")
    plt.close()

    # --- Figure 2: Same plot + methodology text ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    plot_capacity(ax2, annual_cvrj_adp, years_future, cvrj_forecast_vals, combined_forecast,
                  annual_culpeper_total_adp, culpeper_forecast_vals, combined_se=combined_se,
                  annual_cvrj_excl47_for_combined=annual_cvrj_excl47, draw_forecast_marker=True)
    ax2.set_title('CVRJ Bed Need: With vs Without Culpeper County', fontsize=FONT_TITLE)

    methodology = (
        "How the forecasts are created:\n\n"
        "1. Historical ADP: From cvrj_dataset_v2.csv (Book/Release dates). Daily census =\n"
        "   cumsum of +1 on book, -1 day after release; annual ADP = mean per year.\n"
        "   CVRJ baseline excludes County 47. The green line shows total Culpeper County ADP\n"
        "   (all jails) from CORIS, 2014–2025.\n\n"
        "2. Forecast: SARIMAX(ADP, exog=Population), order (1,1,1). CVRJ uses 5-county\n"
        "   population; Culpeper forecasts (when used) are based on Culpeper county population.\n\n"
        "3. Combined load: CVRJ (excl. 47) + Culpeper all jails (2014-2025 + forecast). If combined > 660,\n"
        "   CVRJ would be over capacity if Culpeper joins."
    )
    ax2.text(0.98, 0.98, methodology, transform=ax2.transAxes, fontsize=8,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))
    plt.tight_layout()
    out2 = os.path.join(_ROOT, 'visuals', 'capacity_forecast_with_methodology.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")
    plt.close()

    print("\nDone. Red line = 660-bed capacity. Orange = CVRJ (excl. 47) + Culpeper all jails; blue = CVRJ (excl. 47).")


if __name__ == "__main__":
    main()
