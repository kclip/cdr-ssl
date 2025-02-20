import os
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

from settings import PROJECT_FOLDER


METADATA = {  # Ray-tracing meta-data
    "bs_location": [173.0, 112.0, 73.3355],
    "rt_sim_gain_dbm": 30.0,  # Gain of RT simulation, in [dBm]
}


RX_SETS = ['r051','r052']
PATHS_FILEPATHS = [
    os.path.join(PROJECT_FOLDER, "datasets", "beamforming", "raw", f"GridRxs.paths.t001_46.{rx_set}.p2m")
    for rx_set in RX_SETS
]
RX_LOCATIONS_FILEPATHS = [
    os.path.join(PROJECT_FOLDER, "datasets", "beamforming", "raw", f"GridRxs.pg.t001_46.{rx_set}.p2m")
    for rx_set in RX_SETS
]

# Dataset
N_SAMPLES_TEST = 5975
TRAIN_DATASET_FILEPATH = os.path.join(PROJECT_FOLDER, "datasets", "beamforming", "train_dataset.npz")
TEST_DATASET_FILEPATH = os.path.join(PROJECT_FOLDER, "datasets", "beamforming", "test_dataset.npz")


# Load data
# ---------

def load_rx_locations(rx_locations_filepath: str, rx_idx_offset: int = 0) -> pd.DataFrame:
    rx_info = np.loadtxt(rx_locations_filepath)
    rx_df = pd.DataFrame(
        rx_info[:,:4],
        columns=["rx_idx", "rx_x", "rx_y", "rx_z"]
    )
    rx_df["rx_idx"] = rx_df["rx_idx"].astype(int) + rx_idx_offset
    return rx_df


def load_paths(paths_filepath: str, rx_idx_offset: int = 0, rt_sim_gain_dbm: float = 0) -> pd.DataFrame:
    with open(paths_filepath,'r') as f:
        all_lines=f.readlines()

    n_rx_with_paths = int(all_lines[21].strip('\n'))

    line_cursor = 22 # Start from the first Rx
    all_paths_info = []
    for _ in range(n_rx_with_paths):
        rx_data = all_lines[line_cursor].split()
        rx_idx = int(rx_data[0]) + rx_idx_offset  # Rx index
        n_paths = int(rx_data[1])  # Number of paths at current Rx

        if n_paths == 0:
            # No path data -> go to next Rx
            line_cursor += 1
        else:
            # Go to first Rx path
            line_cursor += 2
            for path_idx in range(n_paths):
                # Collect path info
                path_data = all_lines[line_cursor].split()
                n_interactions = int(path_data[1])
                path_info = [
                    rx_idx,
                    path_idx,
                    float(path_data[2]),              # Path power (dBm)
                    np.deg2rad(float(path_data[3])),  # Path phase (rad)
                    np.deg2rad(float(path_data[7])),  # Departure elevation (rad)
                    np.deg2rad(float(path_data[8])),  # Departure azimuth (rad)
                    n_interactions                    # Number of path-object interactions (excluding Tx and Rx)
                ]
                # Store info
                all_paths_info.append(path_info)
                # Skip path interactions info
                line_cursor += 4 + n_interactions

    # Store path data into dataframe
    paths_df = pd.DataFrame(
        all_paths_info,
        columns=["rx_idx", "path_idx", "power_dbm", "phase", "aod_elevation", "aod_azimuth", "n_interactions"]
    )
    int_cols = ["rx_idx", "path_idx", "n_interactions"]
    paths_df[int_cols] = paths_df[int_cols].astype(int)
    
    # Remove ray-tracing simulation gain
    paths_df["power_dbm"] = paths_df["power_dbm"] - rt_sim_gain_dbm

    return paths_df


# Process paths
# -------------

def get_n_strongest_paths_per_rx(paths_df: pd.DataFrame, n_strongest_paths: int) -> pd.DataFrame:
    return paths_df.sort_values(
        ["rx_idx", "power_dbm"], ascending=False
    ).groupby(
        "rx_idx", as_index=False
    ).head(
        n_strongest_paths
    ).sort_values(
        "rx_idx"
    )


# Run script
# ==========

def generate_beamforming_dataset():
    rt_sim_gain_dbm = METADATA["rt_sim_gain_dbm"]
    
    # Load data
    # ---------
    print("Loading data...")
    rx_idx_offset = 0
    rx_df_list = []
    paths_df_list = []
    for rx_locations_filepath, paths_filepath in zip(RX_LOCATIONS_FILEPATHS, PATHS_FILEPATHS):
        # Load data
        rx_df_iter = load_rx_locations(rx_locations_filepath, rx_idx_offset=rx_idx_offset)
        paths_df_iter = load_paths(paths_filepath, rx_idx_offset=rx_idx_offset, rt_sim_gain_dbm=rt_sim_gain_dbm)
        # Store and update rx index offset
        rx_idx_offset += len(rx_df_iter)
        rx_df_list.append(rx_df_iter)
        paths_df_list.append(paths_df_iter)
    # Concatenate data
    rx_df = pd.concat(rx_df_list, axis=0, ignore_index=True)
    paths_df = pd.concat(paths_df_list, axis=0, ignore_index=True)
    
    # Build LoS indicator feature
    # ---------------------------
    print("Building LoS indicator feature...")
    los_df = paths_df.groupby(
        "rx_idx", as_index=False
    )["n_interactions"].min()
    los_df["is_los"] = los_df["n_interactions"] == 0
    los_df = los_df.drop(columns=["n_interactions"])

    # Get AoD of strongest path
    # -------------------------
    print("Getting AoD of strongest path per Rx...")
    strongest_path_df = get_n_strongest_paths_per_rx(paths_df, n_strongest_paths=1)
    strongest_path_df = strongest_path_df[["rx_idx", "aod_elevation", "aod_azimuth"]].rename(
        columns={
            "aod_elevation": "aod_elevation_strongest_path",
            "aod_azimuth": "aod_azimuth_strongest_path"
        }
    )

    # Concatenate data and store
    # --------------------------
    print("Concatenating and storing dataset...")
    dataset_df = strongest_path_df.merge(
        los_df, on="rx_idx", how="left"
    ).merge(
        rx_df, on="rx_idx", how="left"
    )
    
    # Positions without path info
    no_path_locations_df = rx_df[~rx_df["rx_idx"].isin(dataset_df["rx_idx"])]
    
    # Split dataset and store
    dataset_np = dict(
        rx_positions=dataset_df[["rx_x", "rx_y", "rx_z"]].values,
        rx_los=dataset_df[["is_los"]].values,
        aod_strongest_path=dataset_df[["aod_elevation_strongest_path", "aod_azimuth_strongest_path"]].values,
        no_measurement_positions=no_path_locations_df[["rx_x", "rx_y", "rx_z"]].values
    )
    n_samples = len(dataset_np["rx_positions"])
    indices_test = np.random.choice(n_samples, size=N_SAMPLES_TEST, replace=False)
    mask_test = np.zeros(n_samples, dtype=np.bool)
    mask_test[indices_test] = True
    np.savez_compressed(
        TEST_DATASET_FILEPATH,
        rx_positions=dataset_np["rx_positions"][mask_test],
        rx_los=dataset_np["rx_los"][mask_test],
        aod_strongest_path=dataset_np["aod_strongest_path"][mask_test],
        no_measurement_positions=dataset_np["no_measurement_positions"]
    )
    np.savez_compressed(
        TRAIN_DATASET_FILEPATH,
        rx_positions=dataset_np["rx_positions"][~mask_test],
        rx_los=dataset_np["rx_los"][~mask_test],
        aod_strongest_path=dataset_np["aod_strongest_path"][~mask_test],
        no_measurement_positions=dataset_np["no_measurement_positions"]
    )
    

if __name__ == "__main__":
    print("Generating beamforming dataset from ray-traced paths...")
    generate_beamforming_dataset()
