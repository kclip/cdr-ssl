import os
from typing import List
import hydra
import pandas as pd
from omegaconf import DictConfig
import numpy as np
from scipy.interpolate import griddata
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from settings import DEVICE
from src.data.beamforming import BeamformingDataset
from src.train.criterion import AngleCosineLoss
from study.utils.loaders import load_dataset, load_model
from study.utils.plot import PLOT_KWARGS, set_rc_params, load_protocol_logs, savefig, plot_fill_between



# Protocol params
# ---------------
MAP_PROTOCOLS = {
    "beamforming_supervised_erm": PLOT_KWARGS["erm"],
    "beamforming_pseudo_erm": PLOT_KWARGS["pseudo_erm"],
    "beamforming_dr": PLOT_KWARGS["dr"],
    "beamforming_tdr": PLOT_KWARGS["tdr"],
    "beamforming_cdr": PLOT_KWARGS["cdr"],
}
BS_LOCATION = [173.0, 112.0, 73.3355]


# Line plot
# ---------

def test_loss_vs_labeled_ratio_plot(
    df: pd.DataFrame,
    labeled_ratio_col: str = "config.model_training.labeled_ratio",
    metric_col: str = "test.metrics.AngleCosineLoss"
):
    columns_setting = ["protocol_name", labeled_ratio_col]
    columns = columns_setting + [metric_col]

    # Sanity check (only one run per config)
    cols_w_run = columns + ["config.idx_run"]
    df_test = df[cols_w_run].groupby(cols_w_run, as_index=False).agg(count=("config.idx_run", "count"))
    mask_test = df_test["count"] > 1
    if (mask_test).any():
        raise ValueError(f"Identical run indices for equal settings detected:\n{df_test[mask_test]}")

    # Filter data
    df_p = df[columns].copy()
    protocol_names = list(MAP_PROTOCOLS.keys())
    df_p = df_p[df_p["protocol_name"].isin(protocol_names)]

    # Median and quantile metrics
    df_p = df_p.groupby(
        columns_setting, as_index=False
    )[metric_col].agg(
        median="median",
        q_low=lambda x: x.quantile(0.25),
        q_high=lambda x: x.quantile(0.75),
    ).reset_index()

    # Rearrange data
    df_p = df_p.sort_values(columns_setting)
    labeled_ratio_params = df_p[labeled_ratio_col].unique()

    fig, ax = plt.subplots()
    fig.set_size_inches(9, 5.5)

    # Get metric values
    for protocol_name, protocol_plot_kwargs in MAP_PROTOCOLS.items():
        # Get data
        df_protocol = df_p[df_p["protocol_name"] == protocol_name]

        # Sanity check
        if len(df_protocol) > len(labeled_ratio_params):
            print(f"WARNING: got more datapoints than available labeled_ratio params for protocol {protocol_name}")

        # Plot
        plot_fill_between(
            ax=ax,
            x=df_protocol[labeled_ratio_col],
            y=df_protocol["median"],
            y_err_low=df_protocol["q_low"],
            y_err_high=df_protocol["q_high"],
            plot_kwargs=protocol_plot_kwargs
        )

    # Log x-axis
    ax.set_xscale("log")

    # Axes
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.91),
        ncol=5,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0
    )
    ax.set_xlabel("Labeled Data Ratio")
    ax.set_ylabel("Loss")

    fig.tight_layout()

    return fig, ax


# Test loss map plots
# -------------------

def compute_maps_data(
    cfg,
    df: pd.DataFrame,
    source_cols: List[str],  # Columns in `df` to keep
    source_cols_labels: List[str]  # New name for source columns
):
    # Load raw dataset
    test_dataset_filepath = os.path.join(cfg.dataset.folder, "test_dataset.npz")
    dataset_np = np.load(test_dataset_filepath)
    raw_inputs = dataset_np["rx_positions"].astype(np.float32)
    outputs = torch.tensor(dataset_np["aod_strongest_path"], dtype=torch.float32).to(DEVICE)

    # Trained model uses normalized inputs
    inputs_transform = BeamformingDataset.get_inputs_normalization_transform()
    inputs = inputs_transform(torch.from_numpy(raw_inputs)).to(DEVICE)

    # Loss criterion without reduction
    criterion = AngleCosineLoss(reduction="none")

    # Compute and store loss values and capacity at each test point
    df_maps_rows = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Load model
        model = load_model(cfg)
        model_state_dict = torch.load(
            os.path.join(row["folder"], "trained_model.pth"),
            weights_only=True,
            map_location=DEVICE
        )
        model.load_state_dict(model_state_dict)
        model = model.to(DEVICE)

        # Evaluate model on dataset
        model.eval()
        with torch.no_grad():
            preds = model(inputs)
        losses = criterion(preds, outputs)

        # Store data
        df_maps_rows.append({
            **{col_label: row[col] for col, col_label in zip(source_cols, source_cols_labels)},
            "pred_elevations": preds[:, 0].cpu().numpy().tolist(),
            "pred_azimuths": preds[:, 1].cpu().numpy().tolist(),
            "losses": losses.cpu().numpy().tolist(),
            "mean_loss": losses.cpu().numpy().mean(),
        })

    # Concatenate into dataset
    df_maps = pd.DataFrame.from_records(df_maps_rows)
    df_maps = df_maps.sort_values(source_cols_labels, ascending=False)

    return dataset_np, df_maps


def get_median_maps(df_map_in):
    setting_columns = [
        "protocol_name",
        "labeled_ratio"
    ]
    metrics_columns = [
        "losses",
        "mean_loss"
    ]

    median_loss_maps = []
    for (protocol_name, labeled_ratio), df_g in df_map_in.groupby(setting_columns):
        median_metrics = dict()
        for metric_col in metrics_columns:
            maps = np.stack([
                np.asarray(m, dtype=np.float32)
                for m in df_g[metric_col]
            ], axis=0)
            median_metrics[metric_col] = np.median(maps, axis=0)
        median_loss_maps.append({
            "protocol_name": protocol_name,
            "labeled_ratio": labeled_ratio,
            **median_metrics
        })
    df_median_maps = pd.DataFrame.from_records(median_loss_maps)
    return df_median_maps


def loss_map_plot(
    locations: np.ndarray,
    no_path_locations: np.ndarray,
    values: np.ndarray,
    n_points_axes: int = 200,
    n_levels_z: int = 21,
    min_z: float = 0.0,
    max_z: float = 1.0,
    bs_location: list = None,
    locations_to_display: np.ndarray = None,
    locations_markersize: float = 0.2,
    na_value: float = -1.0,
    color_map_name: str = "turbo",
    c_ticks_every: int = 1,
    c_label: str = "Loss",
    ax=None
):
    locations_plot = np.concatenate(
        [locations[:, :2], no_path_locations[:, :2]],
        axis=0
    )
    values_plot = np.concatenate([
        values,
        na_value * np.ones(len(no_path_locations)),
    ])

    x_axis = np.linspace(locations_plot[:, 0].min(), locations_plot[:, 0].max(), n_points_axes)
    y_axis = np.linspace(locations_plot[:, 1].min(), locations_plot[:, 1].max(), n_points_axes)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    z_grid = griddata(
        points=locations_plot,
        values=values_plot,
        xi=(x_grid, y_grid),
        method="linear"
    )

    z_ticks = np.linspace(min_z, max_z, n_levels_z, endpoint=True)
    if np.isfinite(na_value):
        z_ticks = np.append(z_ticks, na_value)
        z_ticks.sort()
    z_ticks = z_ticks.flatten()

    # Instanciate plot
    if ax is None:
        fig, ax_plot = plt.subplots()
        fig.set_size_inches((7, 6))
    else:
        ax_plot = ax

    # Plot angle maps
    ax_plot.set_xlabel('$x_1$ [m]')
    ax_plot.set_ylabel('$x_2$ [m]')
    cmap = mpl.colormaps[color_map_name]
    cf = ax_plot.contourf(
        x_grid, y_grid, z_grid,
        cmap=cmap, levels=z_ticks,
        vmin=np.nanmin([min_z, na_value]),
        vmax=np.nanmax([max_z, na_value])
    )

    # Plot BS
    if bs_location is not None:
        ax_plot.plot(
            bs_location[0], bs_location[1],
            "x", color="black", markersize=12
        )

    # Plot locations
    if locations_to_display is not None:
        ax_plot.plot(
            locations_to_display[:, 0], locations_to_display[:, 1],
            "x", color="tab:pink", markersize=locations_markersize
        )

    # Return
    if ax is None:
        cbar = fig.colorbar(cf, ticks=z_ticks[::c_ticks_every])
        cbar.set_label(c_label, labelpad=20, rotation=270)
        return fig, ax
    else:
        return cf


# Per-context test loss barplots
# ------------------------------

def compute_loss_per_context(df_map_in: pd.DataFrame, data_info_in) -> pd.DataFrame:
    def mean_masked_array(mask):
        def fn(arr):
            return np.asarray(arr, dtype=np.float32)[mask].mean()
        return fn

    def q_low(x):
        return np.quantile(x, 0.25)

    def q_high(x):
        return np.quantile(x, 0.75)

    df_loss = df_map_in.copy()
    df_loss["mean_loss_los"] = df_loss["losses"].apply(mean_masked_array(data_info_in["rx_los"]))
    df_loss["mean_loss_nlos"] = df_loss["losses"].apply(mean_masked_array(~data_info_in["rx_los"]))
    source_cols = ["protocol_name", "labeled_ratio"]
    loss_cols = ["mean_loss_nlos", "mean_loss_los"]
    df_loss = df_loss[source_cols + loss_cols].groupby(
        source_cols, as_index=False
    ).agg({
        col: ["median", q_low, q_high]
        for col in loss_cols
    })
    df_loss.columns = ['_'.join([c for c in col if c != ""]) for col in df_loss.columns.values]
    
    return df_loss


def barplot_loss_per_context(entries, los_mask, log: bool = False):
    # Init plot
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 5)
    ax.set_ylabel("Loss")
    if log:
        ax.set_yscale('log')
        
    # Plot bars
    n_entries = len(entries)
    bar_labels = []
    for idx, row in enumerate(entries):
        protocol_name = row["protocol_name"]
        protocol_plot_kwargs = MAP_PROTOCOLS[protocol_name]
        bar_labels.append(protocol_plot_kwargs["label"])
        color = protocol_plot_kwargs["color"]
        for col_prefix, x_offset in [("mean_loss_nlos", 0), ("mean_loss_los", n_entries)]:
            median_val = row[f"{col_prefix}_median"]
            q_low_val = row[f"{col_prefix}_q_low"]
            q_high_val = row[f"{col_prefix}_q_high"]
            ax.bar(
                x=idx+x_offset, height=median_val,
                yerr=[
                    [median_val - q_low_val],
                    [q_high_val - median_val]
                ],
                color=color
            )
    
    # Plot labels
    ax.set_xticks(range(0, 2*n_entries), labels=(bar_labels + bar_labels))
    
    # Context delimitation
    ylim = list(ax.get_ylim())
    ylim[1] += 0.15 * (ylim[1] - ylim[0])
    ax.vlines(
        x=n_entries-0.5, ymin=ylim[0], ymax=ylim[1],
        linestyle="--",
        color="black",
        linewidth=1.5,
        zorder=0
    )
    y_text = ylim[0] + 0.95 * (ylim[1] - ylim[0])
    x_center = (n_entries - 1) / 2
    ax.text(x=x_center, y=y_text, s="NLoS", va="top", ha="center")
    ax.text(x=(n_entries + x_center), y=y_text, s="LoS", va="top", ha="center")
    ax.set_ylim(ylim)
    
    return fig, ax


# Dataset map plots
# -----------------

def compute_los_angles(
    bs_location: np.ndarray, # Shape [3]
    rx_locations: np.ndarray # Shape [N, 3]
):
    los_directions = rx_locations - bs_location[np.newaxis, :]
    los_directions /= np.linalg.norm(los_directions, ord=2, axis=1).reshape(-1, 1)
    los_elevations = np.arccos(los_directions[:, 2])
    los_azimuths = np.atan2(los_directions[:, 1], los_directions[:, 0])
    return np.stack([los_elevations, los_azimuths], axis=-1)


def load_angle_data(cfg):
    # Load entire dataset
    train_dataset_filepath = os.path.join(cfg.dataset.folder, "train_dataset.npz")
    test_dataset_filepath = os.path.join(cfg.dataset.folder, "test_dataset.npz")
    dataset_tr = np.load(train_dataset_filepath)
    dataset_te = np.load(test_dataset_filepath)
    gt_angles = np.concatenate(
        [dataset_tr["aod_strongest_path"], dataset_te["aod_strongest_path"]],
        axis=0
    )
    locations = np.concatenate(
        [dataset_tr["rx_positions"], dataset_te["rx_positions"]],
        axis=0
    )
    no_path_locations = dataset_tr["no_measurement_positions"]

    # Get teacher model predictions
    pseudo_angles = compute_los_angles(
        bs_location=np.asarray(BS_LOCATION),
        rx_locations=locations
    )

    return dict(
        locations=locations,
        no_path_locations=no_path_locations,
        gt_angles=gt_angles,
        pseudo_angles=pseudo_angles
    )


def plot_map_angles(
        locations: np.ndarray,
        no_path_locations: np.ndarray,
        values: np.ndarray,
        n_points_axes: int = 200,
        min_z: float = None,
        max_z: float = None,
        bs_location: list = None,
        locations_to_display: np.ndarray = None,
        locations_markersize: float = 0.2,
        na_value: float = -1.0,
        cmap: mpl.colors.Colormap = None,
        c_ticks: list = None,
        c_ticklabels: list = None,
        c_label: str = "Angle"
):
    # Init params
    vmin = np.nanmin([values.min(), na_value]) if min_z is None else np.nanmin([min_z, na_value])
    vmax = np.nanmax([values.max(), na_value]) if max_z is None else np.nanmax([max_z, na_value])

    # Concat values and NA zones
    locations_plot = np.concatenate(
        [locations[:, :2], no_path_locations[:, :2]],
        axis=0
    )
    values_plot = np.concatenate([
        values,
        na_value * np.ones(len(no_path_locations)),
    ])

    # Map values to colors
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.colormaps["jet"] if cmap is None else cmap

    color_values_plot = cmap(norm(values_plot))

    # Interpolate colors onto grid
    x_axis = np.linspace(locations_plot[:, 0].min(), locations_plot[:, 0].max(), n_points_axes)
    y_axis = np.linspace(locations_plot[:, 1].min(), locations_plot[:, 1].max(), n_points_axes)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    z_grid = griddata(
        points=locations_plot,
        values=color_values_plot,
        xi=(x_grid, y_grid),
        method="linear"
    )
    z_grid = np.flipud(z_grid)

    # Instanciate plot
    fig, ax = plt.subplots()
    fig.set_size_inches((7, 6))
    ax.set_xlabel('$x_1$ [m]')
    ax.set_ylabel('$x_2$ [m]')

    # Get map extent
    dx = (x_axis[1] - x_axis[0]) / 2.
    dy = (y_axis[1] - y_axis[0]) / 2.
    extent = [x_axis[0] - dx, x_axis[-1] + dx, y_axis[0] - dy, y_axis[-1] + dy]

    # Plot map
    im = ax.imshow(
        z_grid,
        extent=extent,
        cmap=cmap,
        norm=norm
    )

    # Plot BS
    if bs_location is not None:
        ax.plot(
            bs_location[0], bs_location[1],
            "x", color="black", markersize=12
        )

    # Plot locations
    if locations_to_display is not None:
        ax.plot(
            locations_to_display[:, 0], locations_to_display[:, 1],
            "x", color="tab:pink", markersize=locations_markersize
        )

    # Colorbar
    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi] if c_ticks is None else c_ticks
    ticklabels = [-180, -90, 0, 90, 180] if c_ticklabels is None else c_ticklabels
    cbar = fig.colorbar(mappable=im, ticks=ticks)
    cbar.set_label(c_label, labelpad=20, rotation=270)
    cbar.ax.set_yticklabels(ticklabels)

    return fig, ax


def get_sub_colormap(cmap_name, start, stop, n_points: int = 1000):
    original_cmap = mpl.colormaps[cmap_name]
    colors = original_cmap(np.linspace(start, stop, n_points))
    return mpl.colors.LinearSegmentedColormap.from_list("subset_cmap", colors)


# Tuning parameters plots
# -----------------------

def load_tuning_parameters(
    df_in,
    protocol_names: list,
    groupby_columns: list,
    cols_prefix: str = "train"
):
    def _remove_last_batch(row: dict) -> dict:
        batch_col = f"{cols_prefix}.batch"
        epoch_col = f"{cols_prefix}.epoch"
        tuning_param_col = f"{cols_prefix}.tuning_param"
        batches = np.asarray(row[batch_col])
        max_batch = batches.max()
        mask = batches != max_batch
        row[batch_col] = np.asarray(row[batch_col])[mask].tolist()
        row[epoch_col] = np.asarray(row[epoch_col])[mask].tolist()
        row[tuning_param_col] = np.asarray(row[tuning_param_col])[mask].tolist()
        return row
    
    def _quantile_tuning_param_seq(list_tuning_param_seq: list, quantile: float) -> list:
        return np.quantile(
            np.asarray([e for e in list_tuning_param_seq], dtype=np.float32),
            q=quantile,
            axis=0
        )
    
    df = df_in.copy()
    
    # Filter protocols
    df = df[df["protocol_name"].isin(protocol_names)]
    
    # Remove last tuning parameter of each epoch
    # Note: this is because the last batch of epoch contains a different
    # ratio of labeled/unlabeled samples which can drastically affect the value 
    # of the tuning parameter, making the plots hard to read
    df = df.apply(_remove_last_batch, axis=1)
    
    # Get time axis of each PPI ratio value
    if cols_prefix == "train":
        epoch_col = f"{cols_prefix}.epoch"
        batch_col = f"{cols_prefix}.batch"
        df["time"] = df[[epoch_col, batch_col]].apply(
            lambda row: (np.asarray(row[epoch_col]) * max(row[batch_col])) + np.asarray(row[batch_col]),
            axis=1
        )
    else:
        df["time"] = df[f"{cols_prefix}.epoch"]
    df = df.rename(columns={f"{cols_prefix}.tuning_param": "tuning_param"})
    
    # Aggregate runs
    _grpby_cols = ["protocol_name", *groupby_columns]
    df = df[[*_grpby_cols, "time", "tuning_param"]]
    df = df.groupby(
        _grpby_cols, as_index=False
    ).agg({
        "tuning_param": [
            ("median", lambda x: _quantile_tuning_param_seq(x, 0.50)),
            ("q_low", lambda x: _quantile_tuning_param_seq(x, 0.25)),
            ("q_high", lambda x: _quantile_tuning_param_seq(x, 0.75)),
        ],
        "time": ["first"]
    })
    df.columns = ['_'.join([c for c in col if c not in ("", "first")]) for col in df.columns.values]
    
    return df


def plot_tuning_parameters(
    df,
    filter_values: dict,
    max_batch: int = None,
    xlabel: str = "Batch",
    tdr_protocol_name: str = "beamforming_tdr",
    cdr_protocol_name: str = "beamforming_cdr",
    labeled_ratio: float = None,
    plot_kwargs: dict = None
):
    plot_kwargs = dict() if plot_kwargs is None else plot_kwargs
    
    # Filter dataframe
    df = df.copy()
    for col, val in filter_values.items():
        df = df[df[col] == val]
    
    # Init plot
    fig, ax = plt.subplots()
    fig.set_size_inches(7, 6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tuning $\lambda$")
    
    # Plot TDR
    df_tdr = df[df["protocol_name"] == tdr_protocol_name]
    tdr_plot_kwargs = MAP_PROTOCOLS[tdr_protocol_name]
    if len(df_tdr) > 1:
        raise ValueError("Got more than 1 row for TDR...")
    row_tdr = df_tdr.iloc[0]
    _max_batch = len(row_tdr["time"]) if max_batch is None else max_batch
    plot_fill_between(
        ax=ax,
        x=row_tdr["time"][:_max_batch],
        y=row_tdr["tuning_param_median"][:_max_batch],
        y_err_low=row_tdr["tuning_param_q_low"][:_max_batch],
        y_err_high=row_tdr["tuning_param_q_high"][:_max_batch],
        plot_kwargs={
            **tdr_plot_kwargs,
            **plot_kwargs,
        }
    )
    
    # Plot CDR
    df_cdr = df[df["protocol_name"] == cdr_protocol_name]
    cdr_plot_kwargs = MAP_PROTOCOLS[cdr_protocol_name]
    if len(df_cdr) > 1:
        raise ValueError("Got more than 1 row for TDR...")
    row_cdr = df_cdr.iloc[0]
    _max_batch = len(row_cdr["time"]) if max_batch is None else max_batch
    cdr_data = dict()
    for col in ["tuning_param_median", "tuning_param_q_low", "tuning_param_q_high"]:
        cdr_data[col] = np.asarray(row_cdr[col], dtype=np.float32)
    # Context 0 (NLoS)
    plot_fill_between(
        ax=ax,
        x=row_cdr["time"][:_max_batch],
        y=cdr_data["tuning_param_median"][:_max_batch, 0],
        y_err_low=cdr_data["tuning_param_q_low"][:_max_batch, 0],
        y_err_high=cdr_data["tuning_param_q_high"][:_max_batch, 0],
        plot_kwargs={
            **cdr_plot_kwargs,
            **plot_kwargs,
            "linestyle": "--",
            "label": None,
        }
    )
    # Context 1 (LoS)
    plot_fill_between(
        ax=ax,
        x=row_cdr["time"][:_max_batch],
        y=cdr_data["tuning_param_median"][:_max_batch, 1],
        y_err_low=cdr_data["tuning_param_q_low"][:_max_batch, 1],
        y_err_high=cdr_data["tuning_param_q_high"][:_max_batch, 1],
        plot_kwargs={
            **cdr_plot_kwargs,
            **plot_kwargs
        }
    )
    
    # Plot ratio 1 / (1 + r) for DR
    # Note: 1 / (1 + r) = 1 - \rho where \rho is the labeled data ratio
    if labeled_ratio is not None:
        ratio = 1 - labeled_ratio
        ax.plot(
            [min(row_cdr["time"]), max(row_cdr["time"][:_max_batch])], [ratio, ratio],
            color="black", linestyle="--",
            label="DR"
        )

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.9),
        ncol=3,
        frameon=False
    )
    
    fig.tight_layout()
    
    return fig, ax


# Run all plots
# -------------

@hydra.main(version_base=None, config_path="pkg://config", config_name="config")
def plots_beamforming(cfg: DictConfig) -> None:
    # Setup
    logs_dir = os.path.join(cfg.logs_dir, cfg.logs_subdir)
    save_dir = os.path.join(cfg.logs_dir, "figures")
    set_rc_params()

    # Load logs
    df = load_protocol_logs(logs_dir)

    # Test loss vs labeled ratio plot
    fig, ax = test_loss_vs_labeled_ratio_plot(df=df)
    savefig(
        fig=fig,
        dir=save_dir,
        prefix="beamforming_test_loss_vs_labeled_ratio",
        ext="svg"
    )

    # Compute predictions at test locations
    print("Computing angle predictions at test locations for each trained model...")
    source_cols = ["protocol_name", "config.model_training.labeled_ratio", "config.idx_run"]
    source_cols_labels = ["protocol_name", "labeled_ratio", "idx_run"]
    dataset, df_maps = compute_maps_data(cfg, df=df, source_cols=source_cols, source_cols_labels=source_cols_labels)
    data_info = dict(
        rx_positions=dataset["rx_positions"].astype(np.float32),
        target_elevations=dataset["aod_strongest_path"][:, 0].astype(np.float32),
        target_azimuths=dataset["aod_strongest_path"][:, 1].astype(np.float32),
        rx_los=dataset["rx_los"].reshape(-1),
        no_measurement_positions=dataset["no_measurement_positions"].astype(np.float32)
    )
    # Store maps data
    df_maps.to_pickle(os.path.join(save_dir, "beamforming_test_maps.pkl"))
    np.savez_compressed(os.path.join(save_dir, "beamforming_test_data.npz"), **data_info)

    # Get median values across runs
    df_median_maps = get_median_maps(df_maps)

    # Plot plot all median test loss maps
    print("Plotting all median test loss maps...")
    for _, row in tqdm(df_median_maps.iterrows(), total=len(df_median_maps)):
        fig, ax = loss_map_plot(
            locations=data_info["rx_positions"],
            no_path_locations=data_info["no_measurement_positions"],
            values=row["losses"],
            min_z=0.0,
            max_z=1.0,
            color_map_name="turbo",
            na_value=np.nan,
            n_levels_z=26,
            c_ticks_every=5
        )
        savefig(
            fig=fig,
            dir=os.path.join(save_dir, "beamforming_test_loss_maps"),
            prefix=row["protocol_name"],
            ext="png",
            # Setting params
            labeled_ratio=row["labeled_ratio"]
        )
    
    # Test loss per context barplots
    df_loss = compute_loss_per_context(df_map_in=df_maps, data_info_in=data_info)
    setting_cols = ["labeled_ratio"]
    setting_values = df_loss[setting_cols].drop_duplicates()
    for _, setting_row in setting_values.iterrows():
        # Select entries
        entries = []
        for protocol_name in MAP_PROTOCOLS.keys():
            # Filter protocol entries to setting
            filter_cols = setting_cols
            filter_values = [setting_row[col] for col in filter_cols]
            df_protocol_setting = df_loss[df_loss["protocol_name"] == protocol_name]
            for c, v in zip(filter_cols, filter_values):
                df_protocol_setting = df_protocol_setting[df_protocol_setting[c] == v]
            # Sanity check
            if len(df_protocol_setting) > 1:
                raise ValueError("Detected multiple entries for the same setting")
            # Store entry
            entries.append(df_protocol_setting.iloc[0])
        # Plot
        fig, ax = barplot_loss_per_context(entries=entries, los_mask=data_info["rx_los"], log=False)
        savefig(
            fig=fig,
            dir=os.path.join(save_dir, "loss_per_context"),
            prefix="beamforming_loss_per_context",
            ext="svg",
            labeled_ratio=setting_row['labeled_ratio']
        )

    # Plot ground-truth data and teacher model prediction angle maps
    angle_maps_data = load_angle_data(cfg)
    az_ticklabels = [-180, -120, -60, 0, 60, 120, 180]
    el_ticklabels = [90, 105, 120, 135, 150, 165, 180]
    # GT azimuth
    fig, ax = plot_map_angles(
        locations=angle_maps_data["locations"],
        no_path_locations=angle_maps_data["no_path_locations"],
        values=angle_maps_data["gt_angles"][:, 1],
        n_points_axes=1000,
        min_z=-np.pi,
        max_z=np.pi,
        na_value=np.nan,
        bs_location=BS_LOCATION,
        cmap=mpl.colormaps["hsv"],
        c_ticks=[(np.pi * deg / 180) for deg in az_ticklabels],
        c_ticklabels=az_ticklabels,
        c_label="Azimuth"
    )
    savefig(fig=fig, dir=save_dir, prefix="beamforming_gt_azimuth", ext="png")
    # GT elevation
    fig, ax = plot_map_angles(
        locations=angle_maps_data["locations"],
        no_path_locations=angle_maps_data["no_path_locations"],
        values=angle_maps_data["gt_angles"][:, 0],
        n_points_axes=1000,
        min_z=(np.pi / 2),
        max_z=np.pi,
        na_value=np.nan,
        bs_location=BS_LOCATION,
        cmap=get_sub_colormap("hsv", start=0.5, stop=1.0),
        c_ticks=[(np.pi * deg / 180) for deg in el_ticklabels],
        c_ticklabels=el_ticklabels,
        c_label="Elevation"
    )
    savefig(fig=fig, dir=save_dir, prefix="beamforming_gt_elevation", ext="png")
    # Teacher azimuth
    fig, ax = plot_map_angles(
        locations=angle_maps_data["locations"],
        no_path_locations=angle_maps_data["no_path_locations"],
        values=angle_maps_data["pseudo_angles"][:, 1],
        n_points_axes=1000,
        min_z=-np.pi,
        max_z=np.pi,
        na_value=np.nan,
        bs_location=BS_LOCATION,
        cmap=mpl.colormaps["hsv"],
        c_ticks=[(np.pi * deg / 180) for deg in az_ticklabels],
        c_ticklabels=az_ticklabels,
        c_label="Azimuth"
    )
    savefig(fig=fig, dir=save_dir, prefix="beamforming_pseudo_azimuth", ext="png")
    # Teacher elevation
    fig, ax = plot_map_angles(
        locations=angle_maps_data["locations"],
        no_path_locations=angle_maps_data["no_path_locations"],
        values=angle_maps_data["pseudo_angles"][:, 0],
        n_points_axes=1000,
        min_z=(np.pi / 2),
        max_z=np.pi,
        na_value=np.nan,
        bs_location=BS_LOCATION,
        cmap=get_sub_colormap("hsv", start=0.5, stop=1.0),
        c_ticks=[(np.pi * deg / 180) for deg in el_ticklabels],
        c_ticklabels=el_ticklabels,
        c_label="Elevation"
    )
    savefig(fig=fig, dir=save_dir, prefix="beamforming_pseudo_elevation", ext="png")


    # Plot tuning parameters
    df_tuning = load_tuning_parameters(
        df_in=df,
        protocol_names=[
            "beamforming_tdr",
            "beamforming_cdr"
        ],
        groupby_columns=[
            "config.model_training.labeled_ratio"
        ]
    ).rename(columns={
        "config.model_training.labeled_ratio": "labeled_ratio"
    })
    
    # Plot all tuning parameters training sequences
    MAX_BATCH = 150  # Max batch displayed in plots
    TRAINING_DATASET_SIZE = 30000  # Number of available training points before splitting between labeled/unlabeled data
    setting_cols = ["labeled_ratio"]
    setting_values = df_tuning[setting_cols].drop_duplicates()[setting_cols]
    for _, setting_row in setting_values.iterrows():
        filter_values = {
            col: setting_row[col] for col in setting_cols
        }
        
        # Get labeled ratio per batch
        batch_size_labeled = round(setting_row["labeled_ratio"] * TRAINING_DATASET_SIZE)
        if cfg.model_training.labeled_batch_size != -1:  # If not all labeled data is loaded at each batch
            batch_size_labeled = min(batch_size_labeled, cfg.model_training.labeled_batch_size)
        batch_size_unlabeled = (
            TRAINING_DATASET_SIZE - batch_size_labeled  # If all unlabeled data is loaded at each batch
            if (cfg.model_training.unlabeled_batch_size == -1) else
            cfg.model_training.unlabeled_batch_size
        )
        labeled_ratio_batch = batch_size_labeled / (batch_size_labeled + batch_size_unlabeled)
        
        # Plots
        fig, ax = plot_tuning_parameters(
            df_tuning,
            filter_values=filter_values,
            labeled_ratio=labeled_ratio_batch,
            max_batch=MAX_BATCH,
            plot_kwargs={
                "markevery": 30
            }
        )
        ax.set_ylim([-0.05, 1.05])
        savefig(
            fig=fig,
            dir=os.path.join(save_dir, "tuning"),
            prefix="beamforming_tuning_parameter",
            ext="svg",
            **filter_values
        )
    

if __name__ == "__main__":
    plots_beamforming()
