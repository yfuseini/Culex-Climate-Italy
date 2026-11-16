"""
Culex pipiens (Mission Space): ONE feature pipeline for both the training table and the habitat-suitability map.

Why one script: the model must see identical feature definitions at training time and at map time.
Every feature below is computed by the same functions for (a) the Longo trap cells and (b) every land cell of
the 0.1 deg E-OBS grid.

Inputs
  * Longo et al. aggregated abundance table
  * E-OBS daily tas / tasmin / tasmax (0.1 deg)
  * static layers from culex_prepare_static_layers.py (CORINE fractions, DEM, JRC water)

Feature groups written to <table>_feature_groups.json (read by culex_ablation_robust.py):
  season             week_of_year
  coords             longitude, latitude                    (baseline only, never in the final model)
  landcover_clc      16 CORINE group fractions in the 0.1 deg cell
  landcover_clc_nbhd the same fractions averaged over the 3x3 cells around it (about 33 km window)
  topo               elev_mean, elev_std
  water              water_frac, water_frac_nb
  climate_temp       8 temperature features at the sampling date (lags, hot days, degree-days, winter tasmin)
Dropped because they did not improve held-out skill: climate_water (precipitation, humidity) and protocol.

Outputs
  OUT_CSV                      training table (cell-week)
  OUT_GROUPS                   feature groups json
  OUT_MAP_STATIC_CSV           static features for every map cell
  OUT_MAP_CLIM_NC              climate features for every (year, week, map cell)

Run:  python culex_prep_features_v2.py     (needs several GB RAM: use a compute node)
"""
import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from scipy.ndimage import uniform_filter

# ───────────────────────────── CONFIG ─────────────────────────────
BASE = "/work/ch0636/u301861/phd/EURO-CORDEX/EUR-CMIP6/Culex-pipens"
LONGO_CSV = f"{BASE}/Longo_files/longo_et_al_db_aggregated.csv"
OUT_CSV   = f"{BASE}/Culex-pipens_training_table2.csv"
OUT_GROUPS = OUT_CSV.replace(".csv", "_feature_groups.json")
OUT_MAP_STATIC_CSV = OUT_CSV.replace(".csv", "_map_static.csv")
OUT_MAP_CLIM_NC    = OUT_CSV.replace(".csv", "_map_climate.nc")

EOBS_FILES = {
    "tas":    "/work/ch0636/u301861/phd/deep-esd/EOBS/Obs_tas_EOBS_0.1deg_1950-2025.nc",
    "tasmin": "/work/ch0636/u301861/phd/deep-esd/EOBS/Obs_tasmin_EOBS_0.1deg_1950-2025.nc",
    "tasmax": "/work/ch0636/u301861/phd/deep-esd/EOBS/Obs_tasmax_EOBS_0.1deg_1950-2025.nc",
}
EOBS_VARNAME = {k: k for k in EOBS_FILES}

STATIC_DIR = f"{BASE}/static_layers"
CLC_FRACTIONS_FILE = f"{STATIC_DIR}/clc_fractions_0.1deg.nc"
DEM_FILE   = f"{STATIC_DIR}/dem_italy.tif"
WATER_FILE = f"{STATIC_DIR}/water_inland_italy.tif"
WATER_THRESHOLD = 25          # JRC occurrence (%) above which a pixel counts as water
NBHD_SIZE = 3                 # neighbourhood window in 0.1 deg cells

# Grid used by the static-layer script (must match it)
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 6.0, 19.5, 35.0, 48.0
CELL = 0.1
LAG_START = "2007-06-01"      # earliest date needed for lags and degree-days
LAG_END   = "2022-12-31"

TRAP_TYPES = ["CDC_CO2", "CDC_LIKE_CO2"]
SEX = "F"
GDD_BASE = 10.0
MAX_FILL_DEG = 0.3            # borrow E-OBS values for coastal trap cells with no data within this distance

# Map domain: E-OBS land cells inside this box (lon_min, lon_max, lat_min, lat_max).
# The box still contains parts of neighbouring countries; mask them afterwards if the challenge is Italy only.
MAP_BBOX = (6.5, 18.8, 36.0, 47.2)
MAP_YEARS = list(range(2008, 2023))          # weather years averaged in the map
MAP_WEEK_QUANTILES = (0.05, 0.95)            # map weeks = range of training weeks between these quantiles

CLIM_FEATURES = ["tas_0_2w", "tas_2_4w", "tas_4_8w", "tasmin_0_4w", "tasmax_0_4w",
                 "hot_days_0_4w", "gdd_ytd", "winter_tasmin"]
# ───────────────────────────────────────────────────────────────────
CENTER_WEEKS = (21, 41)      # weeks used to compute each region's mean abundance for the region-centred target

def filter_longo(path):
    df = pd.read_csv(path, na_values=["NA"])
    n0 = len(df)
    df = df[(df["sex"] == SEX) & (df["trap_type"].isin(TRAP_TYPES))]
    df = df[df["municipality_centroid"] == "no"]
    print(f"Longo rows: {n0} -> {len(df)} after sex/trap/centroid filters; {df['cell_id'].nunique()} cells")
    df = df.copy()
    df["ref_date"] = [pd.Timestamp.fromisocalendar(int(y), int(w), 7) for y, w in zip(df["year"], df["week"])]
    return df


# ───────────────────────── static layers ─────────────────────────
def _grid_shape():
    return int(round((LAT_MAX - LAT_MIN) / CELL)), int(round((LON_MAX - LON_MIN) / CELL))


def make_da(arr_topdown):
    """(ny, nx) array, row 0 = northernmost cell  ->  DataArray with ascending lat."""
    ny, nx = arr_topdown.shape
    lat = np.round(LAT_MAX - CELL / 2 - CELL * np.arange(ny), 4)
    lon = np.round(LON_MIN + CELL / 2 + CELL * np.arange(nx), 4)
    return xr.DataArray(arr_topdown, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}).sortby("lat")


def block_edges(tr, shape):
    ny, nx = _grid_shape()
    col = np.round((LON_MIN + CELL * np.arange(nx + 1) - tr.c) / tr.a).astype(int)
    row = np.round((tr.f - (LAT_MAX - CELL * np.arange(ny + 1))) / (-tr.e)).astype(int)
    col, row = np.clip(col, 0, shape[1]), np.clip(row, 0, shape[0])
    if (np.diff(col) <= 0).any() or (np.diff(row) <= 0).any():
        raise RuntimeError("Raster does not cover the 0.1 deg grid: check the static-layer bounds.")
    return row, col


def block_sum(a, row, col):
    a = a[row[0]:row[-1], col[0]:col[-1]]
    s = np.add.reduceat(a, row[:-1] - row[0], axis=0, dtype=np.float64)
    return np.add.reduceat(s, col[:-1] - col[0], axis=1, dtype=np.float64)


def dem_layers():
    with rasterio.open(DEM_FILE) as src:
        a = src.read(1).astype("float32")
        tr = src.transform
    valid = np.isfinite(a) & (a > -9000)
    row, col = block_edges(tr, a.shape)
    n = block_sum(valid.astype("float32"), row, col)
    e = np.where(valid, a, 0.0).astype("float64")
    s1 = block_sum(e, row, col)
    s2 = block_sum(e * e, row, col)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s1 / n, np.nan)
        std = np.sqrt(np.clip(np.where(n > 0, s2 / n - mean ** 2, np.nan), 0, None))
    return {"elev_mean": make_da(mean.astype("float32")), "elev_std": make_da(std.astype("float32"))}


def water_layers():
    with rasterio.open(WATER_FILE) as src:
        a = src.read(1)
        tr = src.transform
    row, col = block_edges(tr, a.shape)
    wet = block_sum((a > WATER_THRESHOLD).astype("float32"), row, col)
    counts = np.outer(np.diff(row), np.diff(col)).astype("float64")
    frac = (wet / counts).astype("float32")
    nb = uniform_filter(frac, size=NBHD_SIZE, mode="nearest").astype("float32")
    return {"water_frac": make_da(frac), "water_frac_nb": make_da(nb)}


def clc_layers():
    ds = xr.open_dataset(CLC_FRACTIONS_FILE).sortby("lat")
    names = [v for v in ds.data_vars if v.startswith("clc_") and v != "clc_valid_frac"]
    focal, nbhd = {}, {}
    for v in names:
        a = ds[v].values.astype("float64")
        focal[v] = ds[v].astype("float32")
        nb = uniform_filter(a, size=NBHD_SIZE, mode="nearest").astype("float32")
        nbhd["clc_nb_" + v[4:]] = xr.DataArray(nb, dims=ds[v].dims, coords=ds[v].coords)
    return focal, nbhd
##########################
COMPOSITES = {
    "forest_natural": ["forest", "natural_open", "bare_sparse"],
    "agri_open":      ["arable_rainfed", "permanent_crops", "pastures", "mixed_agriculture"],
    "agri_wet":       ["arable_irrigated", "rice"],
    "urban":          ["urban_fabric", "industrial_transport", "green_urban_leisure"],
    "wet_water":      ["wetland_inland", "wetland_coastal", "lagoon_estuary", "water_inland"],
}


def composite_layers(nbhd):
    out = {}
    for name, parts in COMPOSITES.items():
        arrs = [nbhd["clc_nb_" + p] for p in parts if "clc_nb_" + p in nbhd]
        tot = arrs[0].astype("float32")
        for a in arrs[1:]:
            tot = tot + a.astype("float32")
        out["lcc_" + name] = tot.astype("float32")
    return out


def build_static_layers():
    focal, nbhd = clc_layers()
    comp = composite_layers(nbhd)
    topo, water = dem_layers(), water_layers()
    layers = {**focal, **nbhd, **comp, **topo, **water}
    groups = {"landcover_clc": list(focal), "landcover_clc_nbhd": list(nbhd), "landcover_comp": list(comp),
              "topo": list(topo), "water": list(water)}
    print(f"Static layers: {len(layers)} ({', '.join(f'{g}={len(c)}' for g, c in groups.items())})")
    return layers, groups
##########################

def sample_static(points, layers):
    la = xr.DataArray(points["latitude"].values, dims="p")
    lo = xr.DataArray(points["longitude"].values, dims="p")
    out = {n: da.sel(lat=la, lon=lo, method="nearest", tolerance=CELL * 0.6).values.astype("float32")
           for n, da in layers.items()}
    return pd.DataFrame(out, index=points.index)


# ───────────────────────── climate ─────────────────────────
def _coord_names(ds):
    return ("longitude" if "longitude" in ds.coords else "lon",
            "latitude" if "latitude" in ds.coords else "lat")


def load_eobs_daily(points):
    """{var: DataFrame(index=date, columns=cell_id)} at the nearest E-OBS cell."""
    lons = xr.DataArray(points["longitude"].values, dims="cell")
    lats = xr.DataArray(points["latitude"].values, dims="cell")
    out = {}
    for var, pattern in EOBS_FILES.items():
        ds = xr.open_mfdataset(pattern, combine="by_coords")
        lon_name, lat_name = _coord_names(ds)
        da = ds[EOBS_VARNAME[var]].sel(time=slice(LAG_START, LAG_END))
        da = da.sel({lon_name: slice(LON_MIN, LON_MAX), lat_name: slice(LAT_MIN, LAT_MAX)}).load()
        sel = da.sel({lon_name: lons, lat_name: lats}, method="nearest", tolerance=0.06)
        arr = sel.values.astype("float32").copy()
        bad = np.where(np.isnan(arr).all(axis=0))[0]
        if len(bad):
            valid = da.notnull().any("time").values
            vlon, vlat = np.meshgrid(da[lon_name].values, da[lat_name].values)
            n_filled = 0
            for j in bad:
                d2 = (vlon - lons.values[j]) ** 2 + (vlat - lats.values[j]) ** 2
                d2 = np.where(valid, d2, np.inf)
                iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
                if d2[iy, ix] <= MAX_FILL_DEG ** 2:
                    arr[:, j] = da.isel({lat_name: iy, lon_name: ix}).values
                    n_filled += 1
            print(f"  {var}: {len(bad)} cells had no data, {n_filled} filled from nearest valid cell")
        out[var] = pd.DataFrame(arr, index=pd.DatetimeIndex(sel["time"].values), columns=points["cell_id"].values)
        print(f"  {var}: {arr.shape}, missing fraction {np.isnan(arr).mean():.3f}")
    return out


def eobs_land_cells():
    """Every E-OBS land cell inside MAP_BBOX."""
    ds = xr.open_mfdataset(EOBS_FILES["tas"], combine="by_coords")
    lon_name, lat_name = _coord_names(ds)
    da = ds[EOBS_VARNAME["tas"]].sel(time=slice(LAG_START, LAG_END))
    da = da.sel({lon_name: slice(MAP_BBOX[0], MAP_BBOX[1]), lat_name: slice(MAP_BBOX[2], MAP_BBOX[3])})
    valid = da.notnull().any("time").load()
    lon2, lat2 = np.meshgrid(valid[lon_name].values, valid[lat_name].values)
    m = valid.values
    pts = pd.DataFrame({"cell_id": np.arange(int(m.sum())),
                        "longitude": np.round(lon2[m], 3), "latitude": np.round(lat2[m], 3)})
    print(f"Map domain: {len(pts)} land cells")
    return pts


def window_mean(df, n_days, lag_days=0):
    return df.rolling(n_days, min_periods=max(1, int(n_days * 0.8))).mean().shift(lag_days)


def window_sum(df, n_days, lag_days=0):
    return df.rolling(n_days, min_periods=max(1, int(n_days * 0.8))).sum().shift(lag_days)


def winter_tasmin(tmin):
    """Mean tasmin over Dec(y-1)-Feb(y) per year and cell."""
    shifted = tmin.copy()
    shifted.index = shifted.index + pd.DateOffset(months=1)
    djf = shifted[shifted.index.month.isin([1, 2, 3])]
    return djf.groupby(djf.index.year).mean()


def climate_at_dates(d, dates):
    """{feature: float32 array (n_dates, n_cells)}; the single definition used for training AND map."""
    dates = pd.DatetimeIndex(dates)
    tas, tmin, tmax = d["tas"], d["tasmin"], d["tasmax"]
    di = tas.index.get_indexer(dates)
    ok = di >= 0

    def pick(fr):
        out = np.full((len(di), fr.shape[1]), np.nan, dtype="float32")
        out[ok] = fr.values[di[ok]]
        return out

    f = {}
    f["tas_0_2w"] = pick(window_mean(tas, 14))
    f["tas_2_4w"] = pick(window_mean(tas, 14, 14))
    f["tas_4_8w"] = pick(window_mean(tas, 28, 28))
    f["tasmin_0_4w"] = pick(window_mean(tmin, 28))
    f["tasmax_0_4w"] = pick(window_mean(tmax, 28))
    f["hot_days_0_4w"] = pick(window_sum((tmax > 30).astype(float).where(tmax.notna()), 28))
    gdd = (tas - GDD_BASE).clip(lower=0)
    f["gdd_ytd"] = pick(gdd.groupby(gdd.index.year).cumsum())
    wt = winter_tasmin(tmin)
    f["winter_tasmin"] = wt.reindex(dates.year).values.astype("float32")
    return f


# ───────────────────────── training table ─────────────────────────
def build_training_table(layers, static_groups):
    table = filter_longo(LONGO_CSV)
    points = table[["cell_id", "longitude", "latitude"]].drop_duplicates("cell_id").reset_index(drop=True)
    print("Reading E-OBS for trap cells ...")
    d = load_eobs_daily(points)
    dates = pd.DatetimeIndex(np.sort(table["ref_date"].unique()))
    arrs = climate_at_dates(d, dates)
    di = dates.get_indexer(pd.DatetimeIndex(table["ref_date"]))
    ci = table["cell_id"].map({c: i for i, c in enumerate(points["cell_id"])}).values
    for k, a in arrs.items():
        table[k] = a[di, ci]
    stat = sample_static(points, layers)
    stat.insert(0, "cell_id", points["cell_id"].values)
    table = table.merge(stat, on="cell_id", how="left")
    table["week_of_year"] = table["week"]
    table["log1p_value"] = np.log1p(table["value"])
############################
    win = table[table["week"].between(*CENTER_WEEKS)]
    region_mean = win.groupby(["Region", "cell_id"])["log1p_value"].mean().groupby("Region").mean()
    table["log1p_value_rc"] = table["log1p_value"] - table["Region"].map(region_mean)
    print("Region mean log1p abundance (weeks %d-%d):\n%s" % (*CENTER_WEEKS, region_mean.round(2).to_string()))
############################

    groups = {"season": ["week_of_year"], "coords": ["longitude", "latitude"], **static_groups,
              "climate_core": ["gdd_ytd", "tas_0_2w", "tas_4_8w"],
              "climate_extra": ["tas_2_4w", "tasmin_0_4w", "tasmax_0_4w", "hot_days_0_4w", "winter_tasmin"],
              "climate_temp": CLIM_FEATURES}
    feat_cols = [c for g in groups.values() for c in g]
####################
    keep = ["cell_id", "year", "week", "ref_date", "value", "log1p_value", "log1p_value_rc", "Region", "trap_type",
            "n_records_aggregated"] + [c for c in dict.fromkeys(feat_cols) if c not in ("longitude", "latitude")]
    keep = keep[:10] + ["longitude", "latitude"] + keep[10:]
    table = table[keep]
####################
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    with open(OUT_GROUPS, "w") as f:
        json.dump(groups, f, indent=2)
    print(f"Saved {len(table)} rows -> {OUT_CSV}\nFeature groups -> {OUT_GROUPS}")
    miss = table[list(dict.fromkeys(feat_cols))].isna().mean()
    print("Features with missing values:\n", miss[miss > 0].round(3).to_string() if (miss > 0).any() else "none")
    return table


# ───────────────────────── map features ─────────────────────────
def build_map_features(table, layers):
    land = eobs_land_cells()
    stat = sample_static(land, layers)
    mp = pd.concat([land, stat], axis=1)
    n_nan = int(mp[list(layers)].isna().any(axis=1).sum())
    print(f"Map cells with any missing static feature: {n_nan}")
    mp.to_csv(OUT_MAP_STATIC_CSV, index=False)

    q0, q1 = table["week"].quantile(MAP_WEEK_QUANTILES[0]), table["week"].quantile(MAP_WEEK_QUANTILES[1])
    weeks = np.arange(int(np.floor(q0)), int(np.ceil(q1)) + 1)
    years = np.array(MAP_YEARS)
    print(f"Map weeks {weeks.min()}-{weeks.max()}, years {years.min()}-{years.max()}")
    dates = [pd.Timestamp.fromisocalendar(int(y), int(w), 7) for y in years for w in weeks]

    print("Reading E-OBS for the map domain ...")
    d = load_eobs_daily(land)
    arrs = climate_at_dates(d, dates)
    ds = xr.Dataset({k: (("year", "week", "cell"), a.reshape(len(years), len(weeks), len(land)))
                     for k, a in arrs.items()},
                    coords={"year": years, "week": weeks, "cell": land["cell_id"].values})
    ds.to_netcdf(OUT_MAP_CLIM_NC)
    print(f"Map climate features -> {OUT_MAP_CLIM_NC}")


def main():
    layers, static_groups = build_static_layers()
    table = build_training_table(layers, static_groups)
    build_map_features(table, layers)


if __name__ == "__main__":
    main()
