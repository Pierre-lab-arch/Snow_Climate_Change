"""Snow depth trends in the French Alps (Meteo-France daily data, Matiu et al. 2021).

Outputs (in OUT_DIR):
  - temporal_trend.png : mean winter snow depth + anomaly with Theil-Sen trend
  - spatial_trends.png : per-station trends (map + trend vs elevation if METADATA exists,
                         otherwise ranked bar plot)
  - station_trends.csv : per-station trend table
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import theilslopes, kendalltau

# ---------------- Config ----------------
DATA = "data/data_daily_FR*.csv"
METADATA = "data/meta_all.csv"
OUT_DIR = Path("outputs")
VAR = "HS_after_gapfill"         # snow depth in cm
SEASON_MONTHS = [12, 1, 2, 3]    # DJFM
MIN_COVERAGE = 0.8               # min fraction of valid days in a season
MIN_SEASONS = 30                 # min valid seasons per station for trends
REF_PERIOD = (1981, 2010)        # climatology for anomalies
MIN_REF_SEASONS = 20
YEARS = (1961, 2020)

OUT_DIR.mkdir(exist_ok=True)

# ---------------- Load & seasonal means ----------------
files = sorted(Path(".").glob(DATA))
print("Files:", [f.name for f in files])
df = pd.concat(pd.read_csv(f, usecols=["Name", "Date", VAR], parse_dates=["Date"]) for f in files)
df = df[df["Date"].dt.month.isin(SEASON_MONTHS)].copy()
# winter year = year of the January (December belongs to the next winter)
df["winter"] = df["Date"].dt.year + (df["Date"].dt.month == 12)
n_days = len(pd.date_range("2001-12-01", "2002-03-31", freq="D"))  # DJFM (non-leap)

g = df.groupby(["Name", "winter"])[VAR]
seas = pd.DataFrame({"hs": g.mean(), "cov": g.count() / n_days}).reset_index()
seas = seas[(seas["cov"] >= MIN_COVERAGE) & seas["winter"].between(*YEARS)]
hs = seas.pivot(index="winter", columns="Name", values="hs")  # years x stations

# keep stations with enough seasons and a reference climatology
ref = hs.loc[REF_PERIOD[0]:REF_PERIOD[1]]
keep = (hs.count() >= MIN_SEASONS) & (ref.count() >= MIN_REF_SEASONS)
hs = hs.loc[:, keep]
anom = hs - hs.loc[REF_PERIOD[0]:REF_PERIOD[1]].mean()
print(f"{hs.shape[1]} stations kept, {hs.shape[0]} winters")


def trend(y):
    """Theil-Sen slope (cm/decade) and Mann-Kendall p-value."""
    y = y.dropna()
    if len(y) < 10:
        return np.nan, np.nan
    slope = theilslopes(y.values, y.index.values)[0]
    return slope * 10, kendalltau(y.index.values, y.values)[1]


# ---------------- Temporal trend ----------------
mean_hs, mean_anom = hs.mean(axis=1), anom.mean(axis=1)
slope, p = trend(mean_anom)
icpt = np.median(mean_anom.values - theilslopes(mean_anom.values, mean_anom.index.values)[0] * mean_anom.index.values)

fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax[0].plot(mean_hs.index, mean_hs, "o-", ms=3, color="tab:blue")
ax[0].set_ylabel("Mean DJFM snow depth (cm)")
ax[0].set_title("Mean snow depth across stations (station set varies with time)")

ax[1].bar(mean_anom.index, mean_anom, color=np.where(mean_anom >= 0, "tab:blue", "tab:red"))
ax[1].plot(mean_anom.index, mean_anom.rolling(10, center=True, min_periods=5).mean(),
           "k", lw=2, label="10-yr running mean")
ax[1].plot(mean_anom.index, icpt + slope / 10 * mean_anom.index, "k--",
           label=f"Theil-Sen: {slope:+.1f} cm/decade (p={p:.3f})")
ax[1].set_ylabel(f"Anomaly vs {REF_PERIOD[0]}-{REF_PERIOD[1]} (cm)")
ax[1].set_xlabel("Winter (year of January)")
ax[1].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "temporal_trend.png", dpi=200)

# ---------------- Spatial trends ----------------
res = pd.DataFrame({c: trend(anom[c]) for c in anom.columns}, index=["slope", "p"]).T
res["pct_slope"] = 100 * res["slope"] / hs.loc[REF_PERIOD[0]:REF_PERIOD[1]].mean()  # %/decade
res["n_seasons"] = hs.count()
res.index.name = "Name"

if Path(METADATA).exists():
    meta = pd.read_csv(METADATA).set_index("Name")[["Longitude", "Latitude", "Elevation"]]
    res = res.join(meta)

res.to_csv(OUT_DIR / "station_trends.csv")

if {"Longitude", "Latitude"} <= set(res.columns):
    r = res.dropna(subset=["Longitude", "Latitude"])
    fig = plt.figure(figsize=(14, 5.5))
    ax = [fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree()), fig.add_subplot(1, 2, 2)]
    vmax = np.nanpercentile(np.abs(r["slope"]), 95)
    ax[0].set_extent([r["Longitude"].min() - 0.5, r["Longitude"].max() + 0.5,
                      r["Latitude"].min() - 0.5, r["Latitude"].max() + 0.5])
    ax[0].add_feature(cfeature.NaturalEarthFeature("physical", "land", "10m"), facecolor="0.95")
    ax[0].add_feature(cfeature.LAKES.with_scale("10m"), facecolor="lightsteelblue", edgecolor="none")
    ax[0].add_feature(cfeature.RIVERS.with_scale("10m"), edgecolor="lightsteelblue", lw=0.5)
    ax[0].add_feature(cfeature.BORDERS.with_scale("10m"), lw=0.8, ls="--")
    ax[0].add_feature(cfeature.COASTLINE.with_scale("10m"), lw=0.6)
    ax[0].gridlines(draw_labels=True, lw=0.3, color="grey", alpha=0.6)
    sc = ax[0].scatter(r["Longitude"], r["Latitude"], c=r["slope"], cmap="RdBu", vmin=-vmax, vmax=vmax,
                       s=np.where(r["p"] < 0.05, 70, 25), edgecolor="k", lw=0.4,
                       transform=ccrs.PlateCarree(), zorder=3)
    ax[0].set_title("Snow depth trend (large markers: p < 0.05)")
    plt.colorbar(sc, ax=ax[0], label="cm/decade", shrink=0.8)
    ax[1].scatter(r["Elevation"], r["slope"], c=np.where(r["p"] < 0.05, "k", "grey"), s=20)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("Elevation (m)"); ax[1].set_ylabel("Trend (cm/decade)")
    ax[1].set_title("Trend vs elevation")
else:
    print(f"'{METADATA}' not found -> ranked bar plot instead of map")
    r = res.sort_values("slope")
    fig, ax = plt.subplots(figsize=(7, max(6, 0.12 * len(r))))
    ax.barh(r.index, r["slope"], color=np.where(r["p"] < 0.05, "tab:red", "lightgrey"))
    ax.set_xlabel("Trend (cm/decade), red: p < 0.05")
    ax.tick_params(axis="y", labelsize=4)

fig.tight_layout()
fig.savefig(OUT_DIR / "spatial_trends_maps.png", dpi=200)
print(res["slope"].describe())
print(f"Significant stations (p<0.05): {(res['p'] < 0.05).sum()} / {len(res)}")