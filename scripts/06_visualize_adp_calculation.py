"""
Visualize how ADP (Average Daily Population) is calculated from CVRJ booking data.

The model uses Book Date and Release Date from cvrj_dataset_v2.csv (length of stay
is implicit: Release Date - Book Date). It does NOT use the "Length of Stay" column
directly; the algorithm is event-based:

  1. Each booking = one interval [Book Date, Release Date].
  2. On Book Date: add +1 to census (inmate enters).
  3. On (Release Date + 1 day): add -1 (inmate leaves).
  4. Build a full date range, sum net changes per day, then CUMSUM = daily census.
  5. Annual ADP = mean of daily census for each year.

This script replicates that logic and plots daily census + annual ADP for CVRJ
(baseline, excluding Culpeper) and optionally Culpeper-from-CSV (County Code 47).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

DATA_DIR = "../data/raw"
CVRJ_FILE = "../data/processed/cvrj_dataset_v2.csv"


def compute_daily_census_and_adp(df, max_date_cap=None):
    """
    Compute daily census and annual ADP from a dataframe with 'Book Date' and 'Release Date'.

    Returns:
        daily_census: Series index by date, values = count in jail that day
        annual_adp: Series index by year-end, values = mean daily census for that year
    """
    min_date = df['Book Date'].min()
    max_date = df['Release Date'].max()
    if pd.isnull(max_date) or (max_date_cap is not None and max_date.year > max_date_cap):
        max_date = pd.Timestamp('2025-12-31')
    date_range = pd.date_range(start=min_date.normalize(), end=max_date.normalize(), freq='D')

    events = []
    for _, row in df.iterrows():
        start = row['Book Date']
        end = row['Release Date']
        if pd.isnull(start):
            continue
        if pd.isnull(end):
            end = max_date  # treat missing release as still in custody through max_date
        start_date = start.normalize()
        end_date = end.normalize()
        if end_date < start_date:
            continue
        events.append((start_date, 1))
        events.append((end_date + pd.Timedelta(days=1), -1))

    evt_df = pd.DataFrame(events, columns=['Date', 'Change'])
    if evt_df.empty:
        daily_census = pd.Series(0, index=date_range)
    else:
        evt_df = evt_df.groupby('Date')['Change'].sum().sort_index()
        daily_census = evt_df.reindex(date_range, fill_value=0).cumsum()

    annual_adp = daily_census.resample('YE').mean()
    return daily_census, annual_adp


def main():
    print("Loading CVRJ Data...")
    df = pd.read_csv(CVRJ_FILE)
    df['Book Date'] = pd.to_datetime(df['Book Date'], errors='coerce')
    df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
    df = df[df['Book Date'].dt.year >= 2012]

    # CVRJ baseline = exclude Culpeper (County Code 47)
    df_baseline = df[df['County Code'] != 47]
    df_culpeper = df[df['County Code'] == 47].copy()

    print("Computing daily census and annual ADP (CVRJ baseline, no Culpeper)...")
    daily_census, annual_adp = compute_daily_census_and_adp(df_baseline, 2030)
    print("Computing Culpeper-from-CSV (County Code 47) using same method...")
    daily_culp, annual_adp_culp = compute_daily_census_and_adp(df_culpeper, 2030)

    # Hardcoded Culpeper ADP used in run_forecast.py (for comparison)
    culpeper_hardcoded = {
        '2020': 148.93, '2021': 185.75, '2022': 210.82,
        '2023': 200.24, '2024': 312.33, '2025': 227.81
    }
    culp_series = pd.Series(culpeper_hardcoded)
    culp_series.index = pd.to_datetime(culp_series.index, format='%Y') + pd.offsets.YearEnd()

    # ---- Figure: 3 panels ----
    fig = plt.figure(figsize=(14, 10))

    # --- Panel A: Conceptual diagram (how one booking contributes to daily census) ---
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 4)
    ax1.axis('off')
    ax1.set_title('How ADP is derived from Book Date & Release Date (length of stay)', fontsize=12)

    # One booking: book on day 2, release on day 5 → contributes to census on days 2,3,4,5
    ax1.add_patch(Rectangle((1.5, 1.5), 4, 1, facecolor='steelblue', alpha=0.6, edgecolor='black'))
    ax1.annotate('Booking interval\n(Book Date → Release Date)', xy=(3.5, 2), fontsize=9, ha='center')
    ax1.plot([0.5, 9.5], [1, 1], 'k-', lw=0.8)
    for i in range(10):
        ax1.axvline(0.5 + i, color='gray', alpha=0.3)
    ax1.set_xticks(np.arange(1, 10))
    ax1.set_xticklabels(['Day 1', 'Day 2\n(book)', 'Day 3', 'Day 4', 'Day 5\n(release)', 'Day 6', 'Day 7', 'Day 8', 'Day 9'])
    ax1.set_xlabel('Time (days)')
    ax1.text(5, 0.4, 'Step 1: For each booking, +1 on Book Date and -1 on day after Release Date.')
    ax1.text(5, 0.0, 'Step 2: Sum changes per day → cumsum = daily census.  Step 3: Annual ADP = mean(daily census) for that year.')

    # --- Panel B: Daily census (last ~2 years so it's readable) ---
    ax2 = fig.add_subplot(3, 1, 2)
    tail = daily_census.loc[daily_census.index >= (daily_census.index.max() - pd.Timedelta(days=730))]
    ax2.fill_between(tail.index, tail.values, alpha=0.5, color='steelblue')
    ax2.plot(tail.index, tail.values, color='steelblue', linewidth=0.8, label='Daily census (CVRJ baseline)')
    ax2.set_ylabel('Number in jail')
    ax2.set_title('Daily census (recent 2 years) — average of this series per year = Annual ADP')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.5)

    # --- Panel C: Annual ADP comparison ---
    ax3 = fig.add_subplot(3, 1, 3)
    years = annual_adp.index.year
    ax3.bar(years - 0.2, annual_adp.values, width=0.35, label='CVRJ baseline (from booking data)', color='steelblue', alpha=0.8)
    ax3.bar(years + 0.2, annual_adp_culp.values, width=0.35, label='Culpeper from CSV (County 47, same method)', color='darkorange', alpha=0.8)
    # Overlay hardcoded Culpeper as line
    hc_years = [pd.Timestamp(y).year for y in culp_series.index]
    ax3.plot(hc_years, culp_series.values, 'o-', color='red', markersize=8, label='Culpeper (hardcoded in run_forecast.py)')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Annual ADP')
    ax3.set_title('Annual ADP: CVRJ (event-based from csv) vs Culpeper (from csv vs hardcoded)')
    # Explain why orange bars (Culpeper from CSV) are low then spike in 2024–2025
    ax3.annotate(
        'Orange bars spike in 2024–2025 because Culpeper (County 47) booking counts\nin the CSV jump in those years (data/recording change or more Culpeper inmates in CVRJ).',
        xy=(0.02, 0.98), xycoords='axes fraction', fontsize=8, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.5)

    plt.tight_layout()
    out_path = 'visuals/eda_adp_calculation.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

    # ---- Second figure: Full daily census + annual ADP on same plot (CVRJ only) ----
    fig2, ax_left = plt.subplots(figsize=(12, 5))
    ax_right = ax_left.twinx()

    full_daily = daily_census
    ax_left.fill_between(full_daily.index, full_daily.values, alpha=0.4, color='steelblue')
    ax_left.plot(full_daily.index, full_daily.values, color='steelblue', linewidth=0.6, label='Daily census')
    ax_left.set_ylabel('Daily census (count)', color='steelblue')
    ax_left.tick_params(axis='y', labelcolor='steelblue')
    ax_left.set_ylim(0, None)

    annual_adp_plot = annual_adp.copy()
    annual_adp_plot.index = annual_adp_plot.index - pd.offsets.MonthBegin(6)  # center bar in year
    ax_right.bar(annual_adp_plot.index, annual_adp_plot.values, width=200, color='darkgreen', alpha=0.7, label='Annual ADP (mean of daily)')
    ax_right.set_ylabel('Annual ADP', color='darkgreen')
    ax_right.tick_params(axis='y', labelcolor='darkgreen')

    ax_left.set_xlabel('Date')
    ax_left.set_title('CVRJ Baseline: Daily census (blue) and annual ADP (green bars) from Book/Release dates in cvrj_dataset_v2.csv')
    ax_left.legend(loc='upper left')
    ax_right.legend(loc='upper right')
    ax_left.grid(True, alpha=0.5)
    plt.tight_layout()
    out_path2 = 'visuals/adp_daily_and_annual.png'
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path2}")
    plt.close()

    print("\nDone. ADP is computed from Book Date and Release Date (length of stay is implicit);")
    print("Culpeper ADP in the forecast script is currently hardcoded; this script also shows")
    print("Culpeper ADP derived from the same CSV (County Code 47) using the same event-based method.")


if __name__ == "__main__":
    main()
