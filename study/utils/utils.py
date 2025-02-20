from typing import Tuple, Callable
import numpy as np
import torch
import torch.nn as nn

from settings import DEVICE
from src.train.tuning import compute_optimal_tuning_parameter_per_context, compute_optimal_tuning_parameter


# Parameters
# ----------

def get_batch_size(dataset, batch_size):
    return len(dataset) if (batch_size == -1) else batch_size


# Tuning
# ------

def compute_optimal_tuning(
    cfg,
    current_tuning_param: torch.Tensor,
    epoch: int,
    model: nn.Module,
    criterion: nn.Module,
    n_contexts: int,
    l_inputs: torch.Tensor,
    l_labels: torch.Tensor,
    l_pseudolabels: torch.Tensor,
    l_contexts: torch.Tensor,
    pl_inputs: torch.Tensor,
    pl_pseudolabels: torch.Tensor,
    pl_contexts: torch.Tensor,
    chunk_size: int = None,
    momentum: float = 0.0
) -> Tuple[Callable, torch.Tensor]:
    if cfg.tuning.type == "TabularTuning":
        new_tuning_param = compute_optimal_tuning_parameter(
            model=model,
            criterion=criterion,
            l_inputs=l_inputs,
            l_labels=l_labels,
            l_pseudolabels=l_pseudolabels,
            pl_inputs=pl_inputs,
            pl_pseudolabels=pl_pseudolabels,
            use_unlabeled_data=cfg.tuning.use_unlabeled_data,
            chunk_size=chunk_size
        )
    elif cfg.tuning.type == "ContextualTabularTuning":
        new_tuning_param = compute_optimal_tuning_parameter_per_context(
            model=model,
            criterion=criterion,
            n_contexts=n_contexts,
            l_inputs=l_inputs,
            l_labels=l_labels,
            l_pseudolabels=l_pseudolabels,
            l_contexts=l_contexts,
            pl_inputs=pl_inputs,
            pl_pseudolabels=pl_pseudolabels,
            pl_contexts=pl_contexts,
            use_unlabeled_data=cfg.tuning.use_unlabeled_data,
            chunk_size=chunk_size
        )
    elif cfg.tuning.type == "ManualContextualTabularTuning":
        schedule_value_idx = np.digitize(epoch, cfg.tuning.kwargs.schedule_epochs).item()
        new_tuning_param = torch.tensor(
            cfg.tuning.kwargs.schedule_values[schedule_value_idx],
            dtype=torch.float32,
            device=DEVICE
        )
    else:
        raise ValueError(f"PPI ratio type '{cfg.tuning.type}' not supported for variance minimization...")
    
    # Momentum
    if current_tuning_param is None:
        tuning_param = new_tuning_param
    else:
        tuning_param = momentum * current_tuning_param + (1 - momentum) * new_tuning_param
    
    # Wrap into context-callable
    if len(tuning_param.shape) == 0:
        tuning_param_model = lambda c: tuning_param
    else:
        tuning_param_model = lambda c: tuning_param[c]
    
    return tuning_param_model, tuning_param