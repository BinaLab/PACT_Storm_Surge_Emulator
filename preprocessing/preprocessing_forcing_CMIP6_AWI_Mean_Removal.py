# %%
import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# ------------------------
# Paths
# ------------------------
base_dir = "./CMIP6_AWI"
forcing_dir = base_dir
out_dir = "./Processed_Forcing_CMIP6_AWI"
os.makedirs(out_dir, exist_ok=True)

cdf_file = "./CMIP6_AWI/uas_3hr_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_197001010300-197101010000_lat0.0-55.0_lon-100.0--40.0_years1970-1971.nc"

# Years you want to process (adjust if needed)
years = list(range(1979, 2100))

# ------------------------
# Define local region
# ------------------------
LAT_MIN, LAT_MAX = 0.0, 55.0
LON_MIN, LON_MAX = -100.0, -40.0

print("Will process years:", years[0], "to", years[-1])
print("Local region lat:", (LAT_MIN, LAT_MAX), "lon:", (LON_MIN, LON_MAX))


def infer_coord_name(ds, candidates):
    for c in candidates:
        if c in ds.coords:
            return c
    for c in candidates:
        if c in ds.dims:
            return c
    return None


# ---- 1) read lat/lon from NetCDF to infer global forcing grid ----
ds = xr.open_dataset(cdf_file, decode_times=False)

lat_name = infer_coord_name(ds, ["lat", "latitude", "nav_lat", "y"])
lon_name = infer_coord_name(ds, ["lon", "longitude", "nav_lon", "x"])

if lat_name is None or lon_name is None:
    raise RuntimeError(f"Cannot find lat/lon. coords={list(ds.coords)} dims={list(ds.dims)}")

lat_nc = ds[lat_name].values
lon_nc = ds[lon_name].values

print("lat_nc shape:", lat_nc.shape, "ndim:", lat_nc.ndim)
print("lon_nc shape:", lon_nc.shape, "ndim:", lon_nc.ndim)

if lat_nc.ndim != 1 or lon_nc.ndim != 1:
    raise RuntimeError(
        "This pipeline assumes 1D lat and 1D lon (regular lat-lon grid). "
        f"Got lat ndim={lat_nc.ndim}, lon ndim={lon_nc.ndim}."
    )

# CMIP6 NetCDF often has lat increasing (S->N), but fort.22 ordering is N->S
lat_inc = bool(np.all(np.diff(lat_nc) > 0))
lat_forcing = lat_nc[::-1] if lat_inc else lat_nc
lon_forcing = lon_nc

Nlat = lat_forcing.size
Nlon = lon_forcing.size
cells_per_timestep = Nlat * Nlon

print("\nGrid size from NetCDF:")
print("  Nlat =", Nlat, "lat range:", (float(lat_forcing.min()), float(lat_forcing.max())),
      "first/last:", float(lat_forcing[0]), float(lat_forcing[-1]))
print("  Nlon =", Nlon, "lon range:", (float(lon_forcing.min()), float(lon_forcing.max())),
      "first/last:", float(lon_forcing[0]), float(lon_forcing[-1]))

# Build local masks in the SAME ordering as the forcing tensor (fort.22)
lat_mask = (lat_forcing >= LAT_MIN) & (lat_forcing <= LAT_MAX)
lon_mask = (lon_forcing >= LON_MIN) & (lon_forcing <= LON_MAX)

lat_local = lat_forcing[lat_mask]
lon_local = lon_forcing[lon_mask]

print("\nLocal lat count:", lat_local.size, "indices head:", np.where(lat_mask)[0][:10])
print("Local lon count:", lon_local.size, "indices head:", np.where(lon_mask)[0][:10])

if lat_local.size == 0 or lon_local.size == 0:
    print("\n⚠️ WARNING: Local region mask produced empty lat or lon.")
    print("    Check LAT/LON bounds or file coverage.")

# 2D lon/lat grid for the local region
Lon_grid_local, Lat_grid_local = np.meshgrid(lon_local, lat_local)  # (nlat, nlon)

# Optional save of grid metadata
np.savez(
    os.path.join(out_dir, "grid_local_cmip6_awi.npz"),
    lat_local=lat_local,
    lon_local=lon_local,
    lat_forcing=lat_forcing,
    lon_forcing=lon_forcing,
    LAT_MIN=LAT_MIN, LAT_MAX=LAT_MAX, LON_MIN=LON_MIN, LON_MAX=LON_MAX,
)
print("\nSaved local grid metadata to:", os.path.join(out_dir, "grid_local_cmip6_awi.npz"))

# Feature index of pressure in the 3 columns coming from fort.22
P_IDX = 2

for year in years:
    fort_path = os.path.join(forcing_dir, f"fort_{year}.22")
    if not os.path.exists(fort_path):
        print(f"WARNING: {fort_path} not found, skipping.")
        continue

    forcing = np.loadtxt(fort_path)  # (Nrows, 3)
    if forcing.ndim != 2 or forcing.shape[1] != 3:
        print(f"WARNING: unexpected shape {forcing.shape} in {fort_path}, skipping.")
        continue

    Nrows = forcing.shape[0]
    TIME_3h = Nrows // cells_per_timestep

    if Nrows != TIME_3h * cells_per_timestep:
        print(f"WARNING: {year}: Nrows={Nrows} not divisible by Nlat*Nlon={cells_per_timestep}. Skipping.")
        continue

    forcing_4d = forcing.reshape(TIME_3h, Nlat, Nlon, 3)  # (TIME_3h, Nlat, Nlon, 3)

    # Subset local region (in fort.22 ordering)
    forcing_4d_local = forcing_4d[:, lat_mask, :][:, :, lon_mask, :]  # (TIME_3h, nlat, nlon, 3)

    nlat = lat_local.size
    nlon = lon_local.size

    # Expand lon/lat over TIME
    Lon_3d_local = np.broadcast_to(Lon_grid_local, (TIME_3h, nlat, nlon))
    Lat_3d_local = np.broadcast_to(Lat_grid_local, (TIME_3h, nlat, nlon))

    # Stack: (TIME_3h, nlat, nlon, 5) = [u, v, p, lon, lat]
    forcing_5d_local_3h = np.concatenate(
        [
            forcing_4d_local,
            Lon_3d_local[..., None],
            Lat_3d_local[..., None],
        ],
        axis=-1,
    )

    # ----------------------------------------------------------
    # NEW: per-time-step spatial mean removal for pressure (local box)
    # p_anom(t,x,y) = p(t,x,y) - mean_{x,y}(p(t,x,y))
    # Save p_mean_t as (TIME_3h,)
    # ----------------------------------------------------------
    p3 = forcing_5d_local_3h[..., P_IDX]                     # (TIME_3h, nlat, nlon)
    p_mean_t_3h = p3.mean(axis=(1, 2))                       # (TIME_3h,)
    forcing_5d_local_3h[..., P_IDX] = p3 - p_mean_t_3h[:, None, None]

    # -------------------------
    # Downsample 3h -> 6h
    # -------------------------
    forcing_5d_local_6h = forcing_5d_local_3h[::2]           # every other step
    p_mean_t_6h = p_mean_t_3h[::2]
    TIME_6h = forcing_5d_local_6h.shape[0]

    # Timestamp printing (your assumed convention)
    start_dt = datetime(year, 10, 25, 0, 0, 0)
    end_dt_3h = start_dt + timedelta(hours=3 * (TIME_3h - 1)) if TIME_3h > 0 else start_dt
    end_dt_6h = start_dt + timedelta(hours=6 * (TIME_6h - 1)) if TIME_6h > 0 else start_dt

    # -------------------------
    # Save outputs
    # -------------------------
    out_path_3h_npy = os.path.join(out_dir, f"forcing_local_3h_{year}.npy")
    out_path_6h_npy = os.path.join(out_dir, f"forcing_local_{year}.npy")

    out_path_3h_npz = os.path.join(out_dir, f"forcing_local_3h_{year}.npz")
    out_path_6h_npz = os.path.join(out_dir, f"forcing_local_{year}.npz")

    np.save(out_path_3h_npy, forcing_5d_local_3h)
    np.save(out_path_6h_npy, forcing_5d_local_6h)

    np.savez(
        out_path_3h_npz,
        forcing=forcing_5d_local_3h,   # pressure is anomaly
        p_mean_t=p_mean_t_3h,          # spatial mean of ABS pressure each timestep
        year=year,
        dt_hours=3,
        LAT_MIN=LAT_MIN, LAT_MAX=LAT_MAX, LON_MIN=LON_MIN, LON_MAX=LON_MAX,
        lat_local=lat_local,
        lon_local=lon_local,
        lat_forcing=lat_forcing,
        lon_forcing=lon_forcing,
    )
    np.savez(
        out_path_6h_npz,
        forcing=forcing_5d_local_6h,   # pressure is anomaly
        p_mean_t=p_mean_t_6h,
        year=year,
        dt_hours=6,
        LAT_MIN=LAT_MIN, LAT_MAX=LAT_MAX, LON_MIN=LON_MIN, LON_MAX=LON_MAX,
        lat_local=lat_local,
        lon_local=lon_local,
        lat_forcing=lat_forcing,
        lon_forcing=lon_forcing,
    )

    print(
        f"{year}: "
        f"3h TIME={TIME_3h}, shape={forcing_5d_local_3h.shape}, "
        f"t=[{start_dt} -> {end_dt_3h}] | "
        f"6h TIME={TIME_6h}, shape={forcing_5d_local_6h.shape}, "
        f"t=[{start_dt} -> {end_dt_6h}]"
    )

print("\nDone. Saved processed CMIP6_AWI local forcing under:", out_dir)
print("Each year now has both .npy and .npz (npz includes p_mean_t).")
