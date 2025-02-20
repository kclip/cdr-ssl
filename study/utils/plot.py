import os
import json
from multiprocessing import Pool
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


# Plot parameters
# ---------------

# Plot kwargs per learning scheme
PLOT_KWARGS = {
    "erm": {
        "label": "ERM",
        "marker": "|",
        "color": "tab:blue",
    },
    "pseudo_erm": {
        "label": "P-ERM",
        "marker": "^",
        "color": "tab:red",
    },
    "dr": {
        "label": "DR",
        "marker": "D",
        "color": "tab:orange",
    },
    "tdr": {
        "label": "TDR",
        "marker": "x",
        "color": "tab:purple",
    },
    "cdr": {
        "label": "CDR",
        "marker": "o",
        "color": "tab:green",
    }
}

# Matplotlib params
def set_rc_params(
    scale: float = 1,
    fontsize_medium: int = 22,
    fontsize_large: int = 26,
):
    plt.rc('text', usetex=True)
    plt.rc('font', size=scale*fontsize_large, family="Times New Roman")
    plt.rc('axes', titlesize=scale*fontsize_large)
    plt.rc('axes', labelsize=scale*fontsize_large)
    plt.rc('xtick', labelsize=scale*fontsize_medium)
    plt.rc('ytick', labelsize=scale*fontsize_medium)
    plt.rc('legend', fontsize=scale*fontsize_medium)
    plt.rc('errorbar', capsize=scale*8)
    plt.rc('lines', linewidth=scale*3, markeredgewidth=scale*3, markersize=scale*8)


# Save/Load
# ---------

def _load_run_info(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return filepath, data

def load_protocol_logs(protocol_dir) -> pd.DataFrame:
    # Get filepaths
    run_info_files = []
    for (dirpath, dirnames, filenames) in os.walk(protocol_dir):
        for filename in filenames:
            if filename == "run_info.json":
                run_info_files.append(os.path.join(dirpath, filename))

    # Load JSONs and fuse into a single dataframe
    print(f"Detected {len(run_info_files)} run-info files. Loading files into a dataframe...")
    pool = Pool()
    df_entries = []
    for filepath, data in tqdm(pool.imap(_load_run_info, run_info_files), total=len(run_info_files)):
        data["folder"] = str(os.path.dirname(filepath))
        df_entries.append(pd.json_normalize(data))
    pool.close()
    pool.join()

    return pd.concat(df_entries, ignore_index=True)


def savefig(fig, dir, prefix=None, ext=".svg", close=True, **kwargs):
    def _sanitize_value(value):
        return str(value).replace(" ", "").replace(".", "u")

    # Create dir if it does not exist
    if not os.path.isdir(dir):
        os.makedirs(dir)

    fig.tight_layout()
    # Get filepath
    filename = ""
    for k, v in [(k, v) for k, v in kwargs.items()]:
        filename += f"_{k}_{_sanitize_value(v)}"
    if prefix is None:
        filename = filename[1:]
    else:
        filename = prefix + filename
    filename = f"{filename}.{ext}"
    filepath = os.path.join(dir, filename)
    # Save and close
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    if close:
        plt.close(fig)


# Plot utils
# ----------

def plot_fill_between(
    ax: plt.Axes,
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    y_err_low: Union[list, np.ndarray],
    y_err_high: Union[list, np.ndarray],
    plot_kwargs: dict = None
):
    if plot_kwargs is None:
        plot_kwargs = dict()

    # Plot
    ax.plot(x, y, **plot_kwargs)

    # Remove unecessary plot kwargs for shaded area
    kwargs_to_remove = ["label", "marker", "markevery"]
    cleaned_plot_kwargs = {
        k: v
        for k, v in plot_kwargs.items()
        if k not in kwargs_to_remove
    }

    # Shaded area
    fill_plot_kwargs = {
        **cleaned_plot_kwargs,
        "alpha": 0.3
    }
    ax.fill_between(x, y_err_low, y_err_high, **fill_plot_kwargs)
    err_limits_kwargs = {
        **cleaned_plot_kwargs,
        "alpha": 0.5,
        "linewidth": 0.7
    }
    for y_err in [y_err_low, y_err_high]:
        ax.plot(x, y_err, **err_limits_kwargs)