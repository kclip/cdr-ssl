import os
import json
import hydra
import pandas as pd
from omegaconf import DictConfig
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import DEVICE
from study.utils.loaders import load_dataset, load_model
from study.utils.plot import PLOT_KWARGS, set_rc_params, load_protocol_logs, savefig


# Protocol params
# ---------------
MAP_PROTOCOLS = {
    "toy_example_supervised_erm": PLOT_KWARGS["erm"],
    "toy_example_pseudo_erm": PLOT_KWARGS["pseudo_erm"],
    "toy_example_dr": PLOT_KWARGS["dr"],
    "toy_example_cdr": PLOT_KWARGS["cdr"]
}


# Ground-truth data and teacher model
# -----------------------------------
def ground_truth_func(x):
    w_neg = 2 * np.pi
    v_neg = 4 * np.pi
    w_pos = 4 * np.pi
    v_pos = 2 * np.pi
    return 0.5 * x * (
        (x < 0).astype(np.float32) * (
            np.cos(w_neg * x) - np.sin(v_neg * x)
        ) +
        (x >= 0).astype(np.float32) * (
            np.cos(w_pos * x) - np.sin(v_pos * x)
        )
    ) + 0.05

def pretrained_model(x):
    w_neg = 2 * np.pi
    v_neg = 4 * np.pi
    w_pos = 3 * np.pi
    v_pos = 5 * np.pi
    return 0.5 * x * (
        (x < 0).astype(np.float32) * (
            np.cos(w_neg * x) - np.sin(v_neg * x)
        ) +
        (x >= 0).astype(np.float32) * (
            np.cos(w_pos * x) - np.sin(v_pos * x)
        )
    )


# Plot functions
# --------------

def data_plot(cfg, n_points_plot: int = 10000):
    # Get ground-truth and teacher model preds
    x_axis = np.linspace(-1, 1, n_points_plot)
    y_gt = ground_truth_func(x_axis)
    y_pseudo = pretrained_model(x_axis)

    # Get labeled datapoints
    dataset_tr, _, _ = load_dataset(cfg)
    n_labeled = round(cfg.model_training.labeled_ratio * len(dataset_tr))
    x_labeled, y_labeled = [], []
    for i in range(n_labeled):
        x, y, _ = dataset_tr[i]
        x_labeled.append(x.item())
        y_labeled.append(y.item())

    # Plot
    fig, ax = plt.subplots()
    fig.set_size_inches((9.5, 4))
    ax.set_xlabel("Covariate $X$")
    ax.set_ylabel("Target $Y$")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.plot(
        x_axis, y_gt,
        linestyle="--",
        color="tab:blue",
        label="Ground-Truth"
    )
    ax.plot(
        x_axis, y_pseudo,
        color="tab:green",
        label="Pseudo-Labels"
    )
    ax.scatter(
        x_labeled, y_labeled,
        s=120,
        marker="x",
        color="tab:red",
        label="Labeled Data",
        zorder=10
    )

    # Context
    ylim = ax.get_ylim()
    ax.vlines(
        x=0, ymin=ylim[0], ymax=ylim[1],
        linestyle="--",
        color="black",
        linewidth=2,
        zorder=0
    )
    y_text = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax.text(x=-0.5, y=y_text, s="Context 0", va="bottom", ha="center")
    ax.text(x=0.5, y=y_text, s="Context 1", va="bottom", ha="center")
    ax.set_ylim(ylim)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.50, 0.85),
        ncol=3,
        handletextpad=0.5,
        columnspacing=1.0,
        frameon=False
    )
    fig.tight_layout()

    return fig, ax


def error_per_context_barplot(
    cfg,
    df: pd.DataFrame,
    n_points: int = 10000,
):
    # Get inputs and ground-truth targets
    x_axis = np.linspace(-1, 1, n_points).astype(np.float32)
    y_gt = ground_truth_func(x_axis)
    inputs = torch.from_numpy(x_axis.reshape(-1, 1)).to(DEVICE)

    # Get predictions per protocol
    all_preds = dict()
    for _, row in df.iterrows():
        # Load last logged model
        model = load_model(cfg).to(DEVICE)
        checkpoint_subpath = f"checkpoints/checkpoint_{cfg.model_training.n_epochs - 1:03}.pth"
        checkpoint_filepath = os.path.join(row["folder"], checkpoint_subpath)
        checkpoint = torch.load(checkpoint_filepath, weights_only=True, map_location=DEVICE)
        model_state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(model_state_dict)
        model.eval()
        # Get preds
        preds = model(inputs).detach().cpu().numpy().reshape(-1)
        all_preds[row["protocol_name"]] = preds

    # Average predictions error per context
    all_errors = dict()
    mask_context_0 = (x_axis < 0)
    mask_context_1 = ~mask_context_0
    for protocol_name, preds in all_preds.items():
        y_pred = all_preds[protocol_name]
        y_err = np.abs(y_pred - y_gt)
        all_errors[protocol_name] = {
            "context_0": y_err[mask_context_0].mean(),
            "context_1": y_err[mask_context_1].mean()
        }

    # Plot
    fig, ax = plt.subplots()
    fig.set_size_inches(9.5, 4)
    ax.set_ylabel("Mean Error $|Y - \hat{Y}|$")

    # Plot bars
    n_entries = len(all_errors)
    bar_labels = []
    for idx, (protocol_name, protocol_plot_kwargs) in enumerate(MAP_PROTOCOLS.items()):
        errors_per_context = all_errors[protocol_name]
        bar_labels.append(protocol_plot_kwargs["label"])
        color = protocol_plot_kwargs["color"]
        for context, x_offset in [("context_0", 0), ("context_1", n_entries)]:
            ax.bar(
                x=idx + x_offset, height=errors_per_context[context],
                color=color
            )

    # Plot labels
    ax.set_xticks(range(0, 2 * n_entries), labels=(bar_labels + bar_labels))

    # Context delimitation
    ylim = list(ax.get_ylim())
    ylim[1] += 0.15 * (ylim[1] - ylim[0])
    ax.vlines(
        x=n_entries - 0.5, ymin=ylim[0], ymax=ylim[1],
        linestyle="--",
        color="black",
        linewidth=2,
        zorder=0
    )
    y_text = ylim[0] + 0.95 * (ylim[1] - ylim[0])
    x_center = (n_entries - 1) / 2
    ax.text(x=x_center, y=y_text, s="Context 0", va="top", ha="center")
    ax.text(x=(n_entries + x_center), y=y_text, s="Context 1", va="top", ha="center")
    ax.set_ylim(ylim)

    return fig, ax


@hydra.main(version_base=None, config_path="pkg://config", config_name="config")
def plots_toy_example(cfg: DictConfig) -> None:
    # Setup
    logs_dir = os.path.join(cfg.logs_dir, cfg.logs_subdir)
    save_dir = os.path.join(cfg.logs_dir, "figures")
    set_rc_params()

    # Dataset and teacher model plot
    fig, ax = data_plot(cfg)
    savefig(
        fig=fig,
        dir=save_dir,
        prefix="toy_example_data",
        ext="svg"
    )

    # Per-context error barplot
    df = load_protocol_logs(logs_dir)
    fig, ax = error_per_context_barplot(cfg, df=df)
    savefig(
        fig=fig,
        dir=save_dir,
        prefix="toy_example_error_barplot",
        ext="svg"
    )


if __name__ == "__main__":
    plots_toy_example()
