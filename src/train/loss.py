from collections import OrderedDict
import torch
import torch.nn as nn


def compute_context_aware_dr_loss(
    model: nn.Module,
    tuning_param_model: nn.Module,
    criterion: nn.Module, 
    l_inputs: torch.Tensor,
    l_labels: torch.Tensor,
    l_contexts: torch.Tensor,
    l_pseudolabels: torch.Tensor,
    pl_inputs: torch.Tensor,
    pl_pseudolabels: torch.Tensor,
    pl_contexts: torch.Tensor,
    bias_estimate_schedule_value: float,
    outer_loop: bool = False,
    n_labeled: int = None,
    n_pseudolabeled: int = None
):
    """
    Compute the Context-aware Doubly-Robust (CDR) loss for a given tuning parameter and labeled and pseudo-labeled data.

    Parameters:
    -----------
    model : nn.Module
        The model used for predictions.
    
    tuning_param_model : nn.Module
        Model that computes the CDR-loss tuning parameter based on contextual data.
    
    criterion : nn.Module
        Loss function to calculate both the labeled and pseudo-losses.
    
    l_inputs : torch.Tensor ; shape [N x ...]
        Input tensor for labeled data.
    
    l_labels : torch.Tensor ; shape [N x ...]
        Ground truth labels for the labeled data.
    
    l_contexts : torch.Tensor ; shape [N x ...]
        Contextual data for the labeled inputs, used by the tuning parameter model.
    
    l_pseudolabels : torch.Tensor ; shape [N x ...]
        Pseudo-labels for the labeled data.
    
    pl_inputs : torch.Tensor ; shape [M x ...]
        Input tensor for unlabeled (pseudo-labeled) data.
    
    pl_pseudolabels : torch.Tensor ; shape [M x ...]
        Pseudo-labels for the unlabeled data.
    
    pl_contexts : torch.Tensor ; shape [M x ...]
        Contextual data for the unlabeled inputs, used by the tuning parameter model.
    
    bias_estimate_schedule_value : float
        Weight the bias estimate (`l_loss - l_pseudoloss`) by `bias_estimate_schedule_value`.
        Scheduling `bias_estimate_schedule_value` from `0` to `1` during training helps improve optimization stability.
    
    outer_loop : bool, optional, default=False
        If True, allows gradients to propagate through the tuning model parameters; if False, detaches the tuning parameters.
    
    n_labeled : int
        Number of labeled data points. Useful if the loss is computed over multiple chunks of batch data.
        
    n_pseudolabeled : int
        Number of pseudo-labeled data points. Useful if the loss is computed over multiple chunks of batch data.

    Returns:
    --------
    torch.Tensor
        The computed CDR loss value as a single scalar tensor.
    """
    n_labeled_ = len(l_inputs) if n_labeled is None else n_labeled
    n_pseudolabeled_ = len(pl_inputs) if n_pseudolabeled is None else n_pseudolabeled
    
    # Labeled data loss and pseudoloss
    l_preds = model(l_inputs)
    l_loss = torch.nan_to_num(criterion(l_preds, l_labels), nan=0.0)
    l_pseudoloss = torch.nan_to_num(criterion(l_preds, l_pseudolabels), nan=0.0)
    # Unlabeled data pseudoloss
    pl_preds = model(pl_inputs)
    pl_pseudoloss = torch.nan_to_num(criterion(pl_preds, pl_pseudolabels), nan=0.0)
    # Tuning parameters
    l_tuning_params = tuning_param_model(l_contexts)
    pl_tuning_params = tuning_param_model(pl_contexts)
    if not outer_loop:
        l_tuning_params, pl_tuning_params = l_tuning_params.detach(), pl_tuning_params.detach()
    # Final loss
    loss = torch.tensor(0.0, device=l_inputs.device)
    if len(l_inputs) > 0:
        loss += torch.sum(
            bias_estimate_schedule_value * (
                l_loss - (l_tuning_params * l_pseudoloss)
            )
        ) / n_labeled_
    if len(pl_inputs) > 0:
        loss += torch.sum(pl_tuning_params * pl_pseudoloss) / n_pseudolabeled_
    return loss


def compute_dr_loss(
    model: nn.Module,
    criterion: nn.Module,
    l_inputs: torch.Tensor,
    l_labels: torch.Tensor,
    l_pseudolabels: torch.Tensor,
    pl_inputs: torch.Tensor,
    pl_pseudolabels: torch.Tensor,
    bias_estimate_schedule_value: float,
    n_labeled: int = None,
    n_pseudolabeled: int = None
):
    """
    Compute the Doubly-Robust (DR) loss for a model based on labeled and pseudo-labeled data.

    Parameters:
    -----------
    model : nn.Module
        The model used for predictions.
    
    criterion : nn.Module
        Loss function to calculate both the labeled and pseudo-losses.
    
    l_inputs : torch.Tensor ; shape [N x ...]
        Input tensor for labeled data.
    
    l_labels : torch.Tensor ; shape [N x ...]
        Ground truth labels for the labeled data.
    
    l_pseudolabels : torch.Tensor ; shape [N x ...]
        Pseudo-labels for the labeled data.
    
    pl_inputs : torch.Tensor ; shape [M x ...]
        Input tensor for unlabeled (pseudo-labeled) data.
    
    pl_pseudolabels : torch.Tensor ; shape [M x ...]
        Pseudo-labels for the unlabeled data.
    
    bias_estimate_schedule_value : float
        Weight the bias estimate (`l_loss - l_pseudoloss`) by `bias_estimate_schedule_value`.
        Scheduling `bias_estimate_schedule_value` from `0` to `1` during training helps improve optimization stability.
    
    n_labeled : int
        Number of labeled data points. Useful if the loss is computed over multiple chunks of batch data.
        
    n_pseudolabeled : int
        Number of pseudo-labeled data points. Useful if the loss is computed over multiple chunks of batch data.

    Returns:
    --------
    torch.Tensor
        The computed DR loss value as a single scalar tensor.
    """
    n_labeled_ = len(l_inputs) if n_labeled is None else n_labeled
    n_pseudolabeled_ = len(pl_inputs) if n_pseudolabeled is None else n_pseudolabeled
    
    # Labeled data loss and pseudoloss
    l_preds = model(l_inputs)
    l_loss = torch.nan_to_num(criterion(l_preds, l_labels), nan=0.0)
    l_pseudoloss = torch.nan_to_num(criterion(l_preds, l_pseudolabels), nan=0.0)
    # Unlabeled data pseudoloss
    pl_preds = model(pl_inputs)
    pl_pseudoloss = torch.nan_to_num(criterion(pl_preds, pl_pseudolabels), nan=0.0)
    # Sum losses
    sum_l_loss = l_loss.sum()
    sum_l_pseudoloss = l_pseudoloss.sum()
    sum_pl_pseudoloss = pl_pseudoloss.sum()
    # Final loss
    loss = (1 / (n_labeled_ + n_pseudolabeled_)) * (sum_l_pseudoloss + sum_pl_pseudoloss)
    if len(l_inputs) > 0:
        loss += (bias_estimate_schedule_value / n_labeled_) * (sum_l_loss - sum_l_pseudoloss)
    
    return loss
