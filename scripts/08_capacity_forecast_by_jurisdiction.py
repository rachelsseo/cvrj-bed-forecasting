"""
Per-jurisdiction capacity forecasts for CVRJ.

For each existing CVRJ member county (Fluvanna, Greene, Louisa, Madison, Orange),
this script creates a plot with:
- Blue: CVRJ baseline (excl. Culpeper) – same for all plots
- Green: <County>-in-CVRJ (County Code from booking data)
- Orange: forecasted combined load = <County>-in-CVRJ + Culpeper-in-CVRJ
         with a one-sigma band around the orange forecast

All plots share the same structure as the main capacity figure:
- Historical vs Forecast split at 2026
- Maximum capacity line at 660 beds
- X-axis from 2016–2036
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)

DATA_DIR = os.path.join(_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
OUT_DIR = os.path.join(_ROOT, "visuals")
MAX_CAPACITY = 660
FORECAST_RESULTS_CSV = os.path.join(DATA_DIR, "outputs", "forecast_results.csv")
CVRJ_ADP_CSV = os.path.join(DATA_DIR, "outputs", "forecast_annual_cvrj_adp.csv")
CULP_IN_CVRJ_CSV = os.path.join(DATA_DIR, "outputs", "forecast_annual_culpeper_in_cvrj.csv")

FONT_LABEL = 13
FONT_ANNO = 12
Y_MAX = 725  # keep consistent with main capacity plot

JURISDICTIONS = [
    {"name": "Fluvanna", "code": 65},
    {"name": "Greene", "code": 79},
    {"name": "Louisa", "code": 109},
    {"name": "Madison", "code": 113},
    {"name": "Orange", "code": 137},
]


def compute_daily_census(df, max_year_cap=2030):
    """Event-based daily census from Book/Release dates."""
    df = df.dropna(subset=["Book Date"]).copy()
    max_date = df["Release Date"].max()
    if pd.isnull(max_date) or max_date.year > max_year_cap:
        max_date = pd.Timestamp("2025-12-31")
    min_date = df["Book Date"].min()
    df["Release Date"] = df["Release Date"].fillna(max_date)
    df["start_d"] = df["Book Date"].dt.normalize()
    df["end_d"] = df["Release Date"].dt.normalize() + pd.Timedelta(days=1)
    df = df[df["end_d"] > df["start_d"]]
    enter = df[["start_d"]].rename(columns={"start_d": "Date"})
    enter["Change"] = 1
    leave = df[["end_d"]].rename(columns={"end_d": "Date"})
    leave["Change"] = -1
    evt = pd.concat([enter, leave], ignore_index=True)
    date_range = pd.date_range(start=min_date.normalize(), end=max_date.normalize(), freq="D")
    if evt.empty:
        return pd.Series(0, index=date_range)
    evt = evt.groupby("Date")["Change"].sum().sort_index()
    return evt.reindex(date_range, fill_value=0).cumsum()


def load_population_series(county_name):
    """Load annual population series for a given county from its Population.csv."""
    path = os.path.join(RAW_DIR, f"{county_name}Population.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, header=4)
    row = df.iloc[1]
    years = [str(y) for y in range(2012, 2025)]
    pop_series = row[years].str.replace(",", "").astype(float)
    pop_series.index = pd.to_datetime(pop_series.index, format="%Y") + pd.offsets.YearEnd()
    return pop_series


def fit_sarimax_with_se(adp, pop, steps=10, label=""):
    """Fit SARIMAX(1,1,1) with exogenous population, return forecast mean and se."""
    combined = pd.concat([adp, pop], axis=1).dropna()
    combined.columns = ["ADP", "Pop"]
    if combined.empty or len(combined) < 3:
        print(f"Not enough data to model {label}")
        return None, None
    model = SARIMAX(
        combined["ADP"],
        exog=combined[["Pop"]],
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
    )
    results = model.fit(disp=False)
    pop_vals = combined["Pop"].values
    x = np.arange(len(pop_vals))
    z = np.polyfit(x, pop_vals, 1)
    future_x = np.arange(len(pop_vals), len(pop_vals) + steps)
    future_pop = np.poly1d(z)(future_x)
    fc = results.get_forecast(steps=steps, exog=future_pop.reshape(-1, 1))
    return fc.predicted_mean, fc.se_mean


def plot_jurisdiction(jur, cvrj_adp, cvrj_forecast, culp_hist_adp, culp_forecast,
                     culp_se_series, years_future, df_cvrj):
    name = jur["name"]
    code = jur["code"]

    # Historical J-in-CVRJ annual ADP
    df_j = df_cvrj[df_cvrj["County Code"] == code].copy()
    daily_j = compute_daily_census(df_j)
    ann_j = daily_j.resample("YE").mean()

    pop_j = load_population_series(name)
    j_forecast, j_se = fit_sarimax_with_se(ann_j, pop_j, steps=len(years_future), label=f"{name} in CVRJ")
    if j_forecast is None:
        return

    # Align historical to start at 2016
    cutoff = pd.Timestamp("2016-01-01")
    hist_years_full = cvrj_adp.index
    hist_years = hist_years_full[hist_years_full >= cutoff]

    ann_j = ann_j.reindex(hist_years).fillna(0.0)
    culp_hist_adp = culp_hist_adp.reindex(hist_years).fillna(0.0)

    all_years = list(hist_years) + list(years_future)

    fig, ax = plt.subplots(figsize=(12, 6))

    # CVRJ baseline full (blue)
    all_cvrj_vals = np.concatenate([cvrj_adp.loc[hist_years].values, cvrj_forecast.values])
    ax.plot(all_years, all_cvrj_vals, "o-", color="steelblue", linewidth=1.5, markersize=5)

    # J-in-CVRJ (green)
    all_j_vals = np.concatenate([ann_j.values, j_forecast.values])
    ax.plot(all_years, all_j_vals, "^-", color="green", linewidth=1.2, markersize=4)

    # Combined J + Culpeper on forecast side only
    x_forecast = pd.Timestamp("2026-01-01")
    last_hist_year = hist_years[-1]
    last_hist_combined = ann_j.loc[last_hist_year] + culp_hist_adp.loc[last_hist_year]

    years_future_trim = years_future[years_future >= x_forecast]
    j_trim = j_forecast[years_future >= x_forecast]
    culp_trim = culp_forecast[years_future >= x_forecast]
    combined_vals = j_trim.values + culp_trim.values

    years_combo = [last_hist_year] + list(years_future_trim)
    vals_combo = [last_hist_combined] + list(combined_vals)
    ax.plot(years_combo, vals_combo, "s-", color="darkorange", linewidth=2, markersize=5)

    # One-sigma band for combined line (sqrt of J and Culpeper variances)
    if j_se is not None and culp_se_series is not None:
        mask_fc = years_future >= x_forecast
        j_se_trim = j_se[mask_fc]
        culp_se_trim = culp_se_series.loc[years_future[mask_fc]].values
        combo_se = np.sqrt(j_se_trim**2 + culp_se_trim**2)
        se_combo = np.concatenate([[0.0], combo_se])  # 0 at anchor point
        upper = np.array(vals_combo) + se_combo
        lower = np.array(vals_combo) - se_combo
        ax.fill_between(years_combo, lower, upper, color="darkorange", alpha=0.15)

    # Capacity line
    ax.axhline(MAX_CAPACITY, color="red", linewidth=2)
    ax.text(pd.Timestamp("2035-01-01"), MAX_CAPACITY + 5, "Maximum Capacity = 660",
            color="red", fontsize=FONT_ANNO, ha="right", va="bottom")

    # Forecast demarcation
    ax.axvline(x_forecast, color="gray", linestyle="--", linewidth=1.5)
    ymin, ymax = ax.get_ylim()
    ax.axvspan(x_forecast, all_years[-1], alpha=0.08, color="gray")
    ax.set_ylim(ymin, ymax)

    ax.set_xlim(pd.Timestamp("2016-01-01"), pd.Timestamp("2036-01-01"))
    ax.set_ylim(0, Y_MAX)
    ax.set_xlabel("Year", fontsize=FONT_LABEL)
    ax.set_ylabel("Average Daily Population (Beds)", fontsize=FONT_LABEL)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.5)

    # Direct labels on lines near the end of the forecast
    last_year = years_future[-1]
    # CVRJ Baseline label (above blue line)
    ax.text(last_year - pd.DateOffset(months=3),
            all_cvrj_vals[-1] - 20,
            "CVRJ Baseline",
            color="steelblue",
            fontsize=FONT_ANNO,
            fontweight="bold",
            ha="right",
            va="top")
    # County-in-CVRJ label (near green line)
    ax.text(last_year - pd.DateOffset(months=3),
            all_j_vals[-1] + 30,
            f"{name}-in-CVRJ",
            color="green",
            fontsize=FONT_ANNO,
            fontweight="bold",
            ha="right",
            va="top")
    # Combined label (above orange line)
    ax.text(last_year - pd.DateOffset(months=3),
            vals_combo[-1] + 20,
            f"Combined ({name} + Culpeper)",
            color="darkorange",
            fontsize=FONT_ANNO,
            fontweight="bold",
            ha="right",
            va="bottom")

    # Historical / Forecast labels at top of plot
    y_text = MAX_CAPACITY + 35  # safely above red capacity line
    mid_hist = pd.Timestamp("2018-01-01")
    mid_fore = pd.Timestamp("2031-01-01")
    ax.text(mid_hist, y_text, "Historical", ha="center", va="bottom",
            fontsize=FONT_ANNO + 2, color="black")
    ax.text(mid_fore, y_text, "Forecast", ha="center", va="bottom",
            fontsize=FONT_ANNO + 2, color="black")

    ax.set_title(f"{name} Bed Need", fontsize=18)

    out_path = os.path.join(OUT_DIR, f"capacity_forecast_{name.lower()}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    # Load base series from existing outputs
    cvrj_adp = pd.read_csv(CVRJ_ADP_CSV, index_col=0, parse_dates=True).squeeze()
    cvrj_adp = cvrj_adp.loc[cvrj_adp.index <= pd.Timestamp("2025-12-31")]
    culp_hist = pd.read_csv(CULP_IN_CVRJ_CSV, index_col=0, parse_dates=True).squeeze()
    culp_hist = culp_hist.loc[culp_hist.index <= pd.Timestamp("2025-12-31")]
    res_df = pd.read_csv(FORECAST_RESULTS_CSV, index_col=0, parse_dates=True)

    years_future = res_df.index
    cvrj_forecast = res_df["CVRJ_Baseline_NoCulpeper"]
    culp_forecast = res_df["Culpeper_In_CVRJ"]
    culp_se_series = res_df["Culpeper_SE"] if "Culpeper_SE" in res_df.columns else None

    # Raw CVRJ booking data
    cvrj_file = os.path.join(PROC_DIR, "cvrj_dataset_v2.csv")
    df = pd.read_csv(cvrj_file)
    df["Book Date"] = pd.to_datetime(df["Book Date"], errors="coerce")
    df["Release Date"] = pd.to_datetime(df["Release Date"], errors="coerce")
    df = df[df["Book Date"].dt.year >= 2012]

    for jur in JURISDICTIONS:
        plot_jurisdiction(jur, cvrj_adp, cvrj_forecast, culp_hist,
                          culp_forecast, culp_se_series, years_future, df)


if __name__ == "__main__":
    main()

