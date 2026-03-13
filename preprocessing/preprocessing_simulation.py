import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from tqdm import tqdm


# =========================
# Mesh reading and utilities
# =========================

@dataclass
class ADCIRCMesh:
    title: str
    ne: int           # number of elements
    np: int           # number of nodes
    node_ids: np.ndarray   # (NP,)
    x: np.ndarray          # (NP,)
    y: np.ndarray          # (NP,)
    depth: np.ndarray      # (NP,)
    elements: List[Tuple[int, int, List[int]]]  # (elem_id, nvert, [node ids])


def read_fort14(path: str) -> ADCIRCMesh:
    """
    Minimal parser for ADCIRC fort.14-style mesh (e.g., 'fort_mesh.14').

    Reads:
      - title line
      - NE, NP
      - NP node lines: node_id, x, y, depth
      - NE element lines: elem_id, nvert, node_ids...

    Boundary info after elements is ignored.
    """
    with open(path, "r") as f:
        title = f.readline().strip()

        # NE = number of elements, NP = number of nodes
        ne_np_line = f.readline().split()
        if len(ne_np_line) < 2:
            raise ValueError("Second line should contain NE and NP.")
        ne, np_ = map(int, ne_np_line[:2])

        # --- Read nodes ---
        node_ids = np.empty(np_, dtype=int)
        x = np.empty(np_, dtype=float)
        y = np.empty(np_, dtype=float)
        depth = np.empty(np_, dtype=float)

        for i in range(np_):
            parts = f.readline().split()
            if len(parts) < 4:
                raise ValueError(f"Node line {i+1} is malformed: {parts}")
            node_ids[i] = int(parts[0])
            x[i] = float(parts[1])
            y[i] = float(parts[2])
            depth[i] = float(parts[3])

        # --- Read elements ---
        elements: List[Tuple[int, int, List[int]]] = []
        for i in range(ne):
            line = f.readline()
            if not line:
                raise ValueError(f"File ended early while reading element {i+1}/{ne}")
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Element line {i+1} is malformed: {parts}")

            elem_id = int(parts[0])
            nvert = int(parts[1])
            conn = [int(v) for v in parts[2:2 + nvert]]

            if len(conn) != nvert:
                raise ValueError(
                    f"Element {elem_id} expected {nvert} vertices, got {len(conn)}."
                )

            elements.append((elem_id, nvert, conn))

    return ADCIRCMesh(
        title=title,
        ne=ne,
        np=np_,
        node_ids=node_ids,
        x=x,
        y=y,
        depth=depth,
        elements=elements,
    )


def find_nearest_node(mesh: ADCIRCMesh, lon: float, lat: float) -> Tuple[int, float]:
    """
    Find nearest mesh node (in degrees) to a given (lon, lat) using a crude
    latitude-corrected Euclidean distance on the sphere.
    """
    lat0_rad = np.deg2rad(lat)
    cos_lat0 = np.cos(lat0_rad)

    dlat = mesh.y - lat
    dlon = (mesh.x - lon) * cos_lat0
    dist2 = dlat**2 + dlon**2

    idx = int(np.argmin(dist2))
    return idx, float(np.sqrt(dist2[idx]))


# =========================
# Station definitions
# =========================

stations: Dict[str, Dict[str, float]] = {
    "Lewes_8557380": {
        "name": "Lewes, DE",
        "lat": 38.78278,
        "lon": -75.11916,
    },
    "Boston_8443970": {
        "name": "Boston, MA",
        "lat": 42.35392,
        "lon": -71.05033,
    },
    "CBBT_8638901": {
        "name": "CBBT, Chesapeake Channel, VA",
        "lat": 37.03290,
        "lon": -76.08330,
    },
    "Battery_8518750": {
        "name": "The Battery, NY",
        "lat": 40.70100,
        "lon": -74.01400,
    },
}

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess ADCIRC fort.63.nc simulation and tidal data."
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="./scripts/fort_mesh.14",
        help="Path to the ADCIRC fort.14 mesh file."
    )
    parser.add_argument(
        "--input-sim",
        type=str,
        default="./CMIP6_MPI",
        help="Input folder containing raw simulation fort.63.nc files."
    )
    parser.add_argument(
        "--input-tide",
        type=str,
        default="./Tidal",
        help="Input folder containing raw tidal fort.63.nc files."
    )
    parser.add_argument(
        "--output-full",
        type=str,
        default="./CMIP6_MPI_fort63_full_npz",
        help="Output folder for full time × node npz files."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./CMIP6_MPI_fort63_station_CSVs",
        help="Output folder for per-station CSV files."
    )
    return parser.parse_args()
    


# =========================
# Main processing pipeline
# =========================

def main():

    args = parse_args()
    # --- Paths and config ---
    mesh_path = args.mesh
    input_folder_sim = args.input_sim
    input_folder_tide = args.input_tide
    full_output_folder = args.output_full      # for full time × node arrays
    station_csv_folder = args.output_csv  # for station CSVs

    # Years to process (each fort_{year}.63.nc is assumed to cover year–year+1)
    year_list = [2005] #list(range(1979, 2101)) # 1979-2100

    os.makedirs(full_output_folder, exist_ok=True)
    os.makedirs(station_csv_folder, exist_ok=True)

    # --- Load mesh once ---
    print(f"Reading mesh from {mesh_path} ...")
    mesh = read_fort14(mesh_path)
    print(f"Mesh loaded: {mesh.np} nodes, {mesh.ne} elements")

    # --- Find nearest node for each station ---
    station_nodes: Dict[str, int] = {}
    print("\nNearest mesh node for each station:")
    for key, info in stations.items():
        node_idx, dist_deg = find_nearest_node(mesh, info["lon"], info["lat"])
        station_nodes[key] = node_idx
        print(
            f"  {info['name']} → node {node_idx} "
            f"(lon={mesh.x[node_idx]:.4f}, lat={mesh.y[node_idx]:.4f}), "
            f"approx distance ~ {dist_deg:.4f} deg"
        )

    # --- Loop over years ---
    for year in year_list:
        print(f"\n===== Processing simulation starting {year} (approx {year}_{year+1}) =====")

        nc_path = os.path.join(input_folder_sim, f"fort_{year}.63.nc")
        nc_tide_path = os.path.join(input_folder_tide, f"fort_{year}.63.nc")

        if not os.path.exists(nc_path):
            print(f"  WARNING: {nc_path} not found, skipping this year.")
            continue
        if not os.path.exists(nc_tide_path):
            print(f"  WARNING: {nc_tide_path} not found, skipping this year.")
            continue

        # Open both files
        with Dataset(nc_path) as nc, Dataset(nc_tide_path) as nc_tide:
            # Optional sanity check: node coordinates in NC vs fort.14
            if "x" in nc.variables and "y" in nc.variables:
                x_nc = nc.variables["x"][:]
                y_nc = nc.variables["y"][:]
                if x_nc.shape[0] != mesh.np or y_nc.shape[0] != mesh.np:
                    print("  WARNING: Node count mismatch between fort.14 and fort.63!")
                else:
                    if not (np.allclose(x_nc, mesh.x) and np.allclose(y_nc, mesh.y)):
                        print("  WARNING: Node order/coords differ between fort.14 and fort.63.")
                    else:
                        print("  Node coordinates match between fort.14 and fort.63.")
            else:
                print("  No x/y variables in netCDF; cannot verify node ordering here.")

            # Extract zeta arrays
            zeta = nc.variables["zeta"][:]          # (Nt_main, Nnode)
            tide_zeta = nc_tide.variables["zeta"][:]  # (Nt_tide, Nnode)

            if zeta.shape[1] != tide_zeta.shape[1]:
                raise ValueError(
                    f"Node dimension mismatch between zeta and tide_zeta for year {year}: "
                    f"{zeta.shape} vs {tide_zeta.shape}"
                )

            # --- Time conversion using both files (NEW: we will align them) ---
            time_var = nc.variables["time"]
            units = time_var.units
            calendar = getattr(time_var, "calendar", "standard")

            time_cf = num2date(time_var[:], units=units, calendar=calendar)
            time_dt = pd.to_datetime([t.isoformat() for t in time_cf])
            print(f"  nc time:   {time_dt[0]} → {time_dt[-1]}  (nt={len(time_dt)})")

            if "time" not in nc_tide.variables:
                raise ValueError(f"  nc_tide for year {year} has no 'time' variable.")

            time_tide_var = nc_tide.variables["time"]
            units_tide = time_tide_var.units
            calendar_tide = getattr(time_tide_var, "calendar", "standard")

            time_tide_cf = num2date(time_tide_var[:], units=units_tide, calendar=calendar_tide)
            time_tide_dt = pd.to_datetime([t.isoformat() for t in time_tide_cf])
            print(f"  tide time: {time_tide_dt[0]} → {time_tide_dt[-1]}  (nt={len(time_tide_dt)})")

            # ================================
            # NEW: Align and crop to common times
            # ================================
            idx_main = pd.Index(time_dt)
            idx_tide = pd.Index(time_tide_dt)

            common_times = idx_main.intersection(idx_tide)

            if common_times.empty:
                print("  ERROR: No overlapping times between nc and nc_tide, skipping this year.")
                continue

            # Get integer indices for common times in each array
            main_indices = idx_main.get_indexer(common_times)
            tide_indices = idx_tide.get_indexer(common_times)

            # Sanity check
            if np.any(main_indices < 0) or np.any(tide_indices < 0):
                print("  ERROR: Failed to align time indices, skipping this year.")
                continue

            # Apply cropping/alignment
            zeta_aligned = zeta[main_indices, :]
            tide_zeta_aligned = tide_zeta[tide_indices, :]
            time_aligned = pd.to_datetime(common_times)

            print(
                f"  Aligned time window: {time_aligned[0]} → {time_aligned[-1]} "
                f"(nt={len(time_aligned)})"
            )
            # ================================
            # End of NEW alignment block
            # ================================

            # ------------------------
            # (2) Save full data as npz
            # ------------------------
            full_npz_path = os.path.join(
                full_output_folder,
                f"fort_{year}_zeta_tide_full.npz"
            )
            data_dict = {
                "time": time_aligned.to_numpy(),  # datetime64[ns] array
                "nc": zeta_aligned,               # full ADCIRC run (aligned)
                "nc_tide": tide_zeta_aligned,     # tide-only run (aligned)
            }
            np.savez(full_npz_path, **data_dict)
            print(f"  Saved full aligned data to {full_npz_path}")

            # -----------------------------
            # (3–4) Per-station CSV exports
            # -----------------------------
            for key, info in stations.items():
                node_idx = station_nodes[key]

                eta = zeta_aligned[:, node_idx]          # (Nt_aligned,)
                tide_eta = tide_zeta_aligned[:, node_idx]

                # Build DataFrame with three columns: time, nc, nc_tide
                df_station = pd.DataFrame({
                    "time": time_aligned,
                    "nc": eta,
                    "nc_tide": tide_eta,
                })

                # Use short station name from key, e.g. "Battery_8518750" -> "Battery"
                short_name = key.split("_")[0]
                csv_name = f"{year}_{year+1}_{short_name}.csv"
                csv_path = os.path.join(station_csv_folder, csv_name)
                df_station.to_csv(csv_path, index=False)

                print(
                    f"  {info['name']} – node {node_idx}: "
                    f"min nc={eta.min():.3f}, max nc={eta.max():.3f}, "
                    f"min nc_tide={tide_eta.min():.3f}, max nc_tide={tide_eta.max():.3f}"
                )
                print(f"    Saved station CSV: {csv_path}")


if __name__ == "__main__":
    main()
