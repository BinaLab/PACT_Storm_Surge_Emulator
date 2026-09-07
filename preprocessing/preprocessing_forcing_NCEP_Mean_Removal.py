# %%
import os
import numpy as np
from datetime import datetime, timedelta

years = np.arange(1979, 2014 + 1)  # 1979-2014

# Global grid size
NWLAT = 94
NWLON = 192

# 1D lat / lon arrays for the global grid
lat = np.array([
    88.542, 86.6531, 84.7532, 82.8508, 80.9473, 79.0435, 77.1394, 75.2351,
    73.3307, 71.4262, 69.5217, 67.6171, 65.7125, 63.8079, 61.9033, 59.9986,
    58.0939, 56.1893, 54.2846, 52.3799, 50.4752, 48.5705, 46.6658, 44.7611,
    42.8564, 40.9517, 39.047, 37.1422, 35.2375, 33.3328, 31.4281, 29.5234,
    27.6186, 25.7139, 23.8092, 21.9044, 19.9997, 18.095, 16.1902, 14.2855,
    12.3808, 10.47604, 8.57131, 6.66657, 4.76184, 2.8571, 0.952368,
    -0.952368, -2.8571, -4.76184, -6.66657, -8.57131, -10.47604, -12.3808,
    -14.2855, -16.1902, -18.095, -19.9997, -21.9044, -23.8092, -25.7139,
    -27.6186, -29.5234, -31.4281, -33.3328, -35.2375, -37.1422, -39.047,
    -40.9517, -42.8564, -44.7611, -46.6658, -48.5705, -50.4752, -52.3799,
    -54.2846, -56.1893, -58.0939, -59.9986, -61.9033, -63.8079, -65.7125,
    -67.6171, -69.5217, -71.4262, -73.3307, -75.2351, -77.1394, -79.0435,
    -80.9473, -82.8508, -84.7532, -86.6531, -88.542
])

lon = np.array([
    -178.125, -176.25, -174.375, -172.5, -170.625, -168.75, -166.875, -165.,
    -163.125, -161.25, -159.375, -157.5, -155.625, -153.75, -151.875, -150.,
    -148.125, -146.25, -144.375, -142.5, -140.625, -138.75, -136.875, -135.,
    -133.125, -131.25, -129.375, -127.5, -125.625, -123.75, -121.875, -120.,
    -118.125, -116.25, -114.375, -112.5, -110.625, -108.75, -106.875, -105.,
    -103.125, -101.25, -99.375, -97.5, -95.625, -93.75, -91.875, -90.,
    -88.125, -86.25, -84.375, -82.5, -80.625, -78.75, -76.875, -75.,
    -73.125, -71.25, -69.375, -67.5, -65.625, -63.75, -61.875, -60.,
    -58.125, -56.25, -54.375, -52.5, -50.625, -48.75, -46.875, -45.,
    -43.125, -41.25, -39.375, -37.5, -35.625, -33.75, -31.875, -30.,
    -28.125, -26.25, -24.375, -22.5, -20.625, -18.75, -16.875, -15.,
    -13.125, -11.25, -9.375, -7.5, -5.625, -3.75, -1.875, 0.,
    1.875, 3.75, 5.625, 7.5, 9.375, 11.25, 13.125, 15., 16.875, 18.75,
    20.625, 22.5, 24.375, 26.25, 28.125, 30., 31.875, 33.75, 35.625, 37.5,
    39.375, 41.25, 43.125, 45., 46.875, 48.75, 50.625, 52.5, 54.375, 56.25,
    58.125, 60., 61.875, 63.75, 65.625, 67.5, 69.375, 71.25, 73.125, 75.,
    76.875, 78.75, 80.625, 82.5, 84.375, 86.25, 88.125, 90., 91.875, 93.75,
    95.625, 97.5, 99.375, 101.25, 103.125, 105., 106.875, 108.75, 110.625,
    112.5, 114.375, 116.25, 118.125, 120., 121.875, 123.75, 125.625, 127.5,
    129.375, 131.25, 133.125, 135., 136.875, 138.75, 140.625, 142.5, 144.375,
    146.25, 148.125, 150., 151.875, 153.75, 155.625, 157.5, 159.375, 161.25,
    163.125, 165., 166.875, 168.75, 170.625, 172.5, 174.375, 176.25, 178.125,
    180.
])

assert lat.shape[0] == NWLAT
assert lon.shape[0] == NWLON

# ------------------------
# Define local region
# ------------------------
LAT_MIN, LAT_MAX = 0.0, 50.0
LON_MIN, LON_MAX = -105.0, -55.0

lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)

lat_local = lat[lat_mask]
lon_local = lon[lon_mask]

print("Local lat indices:", np.where(lat_mask)[0])
print("Local lon indices:", np.where(lon_mask)[0])
print("Local grid size:  nlat =", lat_local.size, ", nlon =", lon_local.size)

# 2D lon/lat grid for the local region
Lon_grid_local, Lat_grid_local = np.meshgrid(lon_local, lat_local)  # (nlat, nlon)

# Where to save processed local forcing
out_dir = "./Processed_Forcing_NCEP"
os.makedirs(out_dir, exist_ok=True)

# Feature indices in the final 5D array
# (TIME, nlat, nlon, 5) = [WVX, WVY, PR, lon, lat]
P_IDX = 2

# ------------------------
# Main loop over years
# ------------------------
for year in years:
    year = int(year)
    fort_path = f"./NCEP_Reanalysis/fort_{year}.22"
    if not os.path.exists(fort_path):
        print(f"WARNING: {fort_path} not found, skipping.")
        continue

    forcing = np.loadtxt(fort_path)   # (N, 3) global
    if forcing.ndim != 2 or forcing.shape[1] != 3:
        print(f"WARNING: unexpected shape {forcing.shape} in {fort_path}, skipping.")
        continue

    # Infer TIME from #rows
    N = forcing.shape[0]
    cells_per_timestep = NWLAT * NWLON
    TIME = N // cells_per_timestep
    if N != TIME * cells_per_timestep:
        raise ValueError(
            f"N={N} not divisible by NWLAT*NWLON={cells_per_timestep} for year={year} "
            f"(remainder={N - TIME * cells_per_timestep})"
        )

    # Reshape to (TIME, NWLAT, NWLON, 3)
    forcing_4d = forcing.reshape(TIME, NWLAT, NWLON, 3)

    # Subset to local region: (TIME, nlat, nlon, 3)
    forcing_4d_local = forcing_4d[:, lat_mask, :][:, :, lon_mask, :]

    nlat = lat_local.size
    nlon = lon_local.size

    # Expand 2D lon/lat to 3D over TIME
    Lon_3d_local = np.broadcast_to(Lon_grid_local, (TIME, nlat, nlon))
    Lat_3d_local = np.broadcast_to(Lat_grid_local, (TIME, nlat, nlon))

    # Stack: (TIME, nlat, nlon, 5) = [WVX, WVY, PR, lon, lat]
    forcing_5d_local = np.concatenate(
        [
            forcing_4d_local,
            Lon_3d_local[..., None],
            Lat_3d_local[..., None],
        ],
        axis=-1,
    )

    # ----------------------------------------------------------
    # NEW: per-time-step spatial-mean removal for pressure
    #   p_anom(t,x,y) = p(t,x,y) - mean_{x,y}(p(t,x,y))
    # Also SAVE the mean pressure per timestep as an extra array:
    #   p_mean_t: (TIME,)
    # ----------------------------------------------------------
    p = forcing_5d_local[..., P_IDX]                        # (TIME, nlat, nlon)
    p_mean_t = p.mean(axis=(1, 2))                          # (TIME,)
    forcing_5d_local[..., P_IDX] = p - p_mean_t[:, None, None]

    # Timestamp printout: NCEP assumed already 6-hour step
    start_dt = datetime(year, 10, 25, 0, 0, 0)
    end_dt = start_dt + timedelta(hours=6 * (TIME - 1)) if TIME > 0 else start_dt
    print(f"{year}: forcing_5d_local shape={forcing_5d_local.shape} | "
          f"p_mean_t shape={p_mean_t.shape} | "
          f"t=[{start_dt} -> {end_dt}] (dt=6h, steps={TIME})")

    # Save to disk:
    # - keep the original .npy for backward compatibility
    # - also save a .npz that includes the pressure mean time series
    out_path_npy = os.path.join(out_dir, f"forcing_local_{year}.npy")
    out_path_npz = os.path.join(out_dir, f"forcing_local_{year}.npz")

    np.save(out_path_npy, forcing_5d_local)
    np.savez(
        out_path_npz,
        forcing=forcing_5d_local,     # (TIME,nlat,nlon,5) with pressure anomaly
        p_mean_t=p_mean_t,            # (TIME,) spatial mean of absolute pressure per timestep
        # optional metadata (handy for debugging)
        year=year,
        dt_hours=6,
        LAT_MIN=LAT_MIN, LAT_MAX=LAT_MAX, LON_MIN=LON_MIN, LON_MAX=LON_MAX,
        lat_local=lat_local,
        lon_local=lon_local,
    )

print("\nDone. Saved processed NCEP local forcing under:", out_dir)
print("Note: each year now has both:")
print("  - forcing_local_YYYY.npy  (pressure already anomaly)")
print("  - forcing_local_YYYY.npz  (contains 'forcing' + 'p_mean_t' + metadata)")
