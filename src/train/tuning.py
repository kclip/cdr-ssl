from typing import Dict
import numpy as np
import torch
import torch.nn as nn


def batch_gradients(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    criterion: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    :param model: nn.Module
        Prediction model
    :param params:  Dict[str, torch.Tensor] ; tensors shape [P_1, ..., P_M]
        Dictionary of model parameters
    :param inputs:  torch.Tensor ; shape [N, ...]
        Batch of inputs
    :param labels:  torch.Tensor ; shape [N]
        Batch of associated ground-truth labels
    :return: Dict[str, torch.Tensor] ; tensors shape [N, P_1, ..., P_M]
        Gradients w.r.t. model parameters for each example in the (`inputs`, `labels`) batch
    """
    def compute_loss(params_, inputs, labels):
        preds = torch.func.functional_call(model, params_, (inputs.unsqueeze(0),))
        return criterion(preds, labels.unsqueeze(0))[0]
    
    return torch.func.vmap(
        torch.func.grad(compute_loss),
        in_dims=(None, 0, 0)
    )(params, inputs, labels)


def init_gradient(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(params)
        for name, params in params.items()
    }


def compute_optimal_tuning_parameter(
    model: nn.Module,
    criterion: nn.Module,
    l_inputs: torch.Tensor,
    l_labels: torch.Tensor,
    l_pseudolabels: torch.Tensor,
    pl_inputs: torch.Tensor,
    pl_pseudolabels: torch.Tensor,
    use_unlabeled_data: bool = False,
    chunk_size=None
) -> torch.Tensor:
    """
    :param model: nn.Module
        Prediction model
    :param criterion: nn.Module
        Loss criterion
    :param l_inputs:  torch.Tensor ; shape [N_l, ...]
        Batch of inputs with associated ground-truth labels
    :param l_labels:  torch.Tensor ; shape [N_l]
        Batch of ground-truth labels associated to `l_inputs`
    :param l_pseudolabels:  torch.Tensor ; shape [N_l]
        Batch of pseudo-labels associated to `l_inputs`
    :param pl_inputs:  torch.Tensor ; shape [N_pl, ...]
        Batch of inputs without associated ground-truth labels 
    :param pl_pseudolabels:  torch.Tensor ; shape [N_pl]
        Batch of pseudo-labels associated to `pl_inputs`
    :param use_unlabeled_data: bool
        Whether to use the unlabled data to compute the tuning parameter denominator
    :return: torch.Tensor ; shape []
        Scalar tuning parameter minimizing next parameter iterate variance for the given sets of labeled and
        unlabeled data
    """
    params = dict(model.named_parameters())
    n_labeled = l_inputs.shape[0]
    n_pseudolabeled = pl_inputs.shape[0]
    chunk_size = max(l_inputs.shape[0], pl_inputs.shape[0]) if chunk_size is None else chunk_size
    
    # Edge cases
    # ----------
    if n_labeled == 0:
        print("WARNING: no labeled data found for computing the tuning parameter ! Defaulting tuning parameter to one...")
        return torch.tensor(1.0, device=l_inputs.device)
    if n_pseudolabeled == 0:
        print("WARNING: no unlabeled data found when computing the tuning parameter ! Defaulting tuning parameter to zero...")
        return torch.tensor(0.0, device=l_inputs.device)
    
    
    # Labeled data
    # ------------
    # Compute mean centered-pseudo-gradient norm as E[|PG|^2] - |E[PG]|^2 and
    # mean centered-gradient centered-pseudo-gradient inner-product as  E[G^T PG] - E[G^T] E[PG]
    # Init mean gradients
    l_mean_grad = init_gradient(params)
    l_mean_pseudo_grad = init_gradient(params)
    # Init gradient pseudo-gradient inner-product and pseudo-gradient norm
    l_g_pg_inner_prod = torch.tensor(0.0, device=l_inputs.device)
    l_pg_norm = torch.tensor(0.0, device=l_inputs.device)

    # Lower memory requirements by splitting the computation into sequential chunks 
    n_l_chunks = int(np.ceil(n_labeled / chunk_size))
    for l_inputs_chunk, l_labels_chunk, l_pseudolabels_chunk in zip(
        l_inputs.chunk(n_l_chunks),
        l_labels.chunk(n_l_chunks),
        l_pseudolabels.chunk(n_l_chunks)
    ):
        jacobians = batch_gradients(
            model=model,
            params=params,
            criterion=criterion,
            inputs=l_inputs_chunk,
            labels=l_labels_chunk
        )
        pseudo_jacobians = batch_gradients(
            model=model,
            params=params,
            criterion=criterion,
            inputs=l_inputs_chunk,
            labels=l_pseudolabels_chunk
        )
        for param_name in params.keys():
            jac, pseudo_jac = jacobians[param_name], pseudo_jacobians[param_name]
            # Update mean gradients
            l_mean_grad[param_name] += (1 / n_labeled) * torch.sum(jac, dim=0)
            l_mean_pseudo_grad[param_name] += (1 / n_labeled) * torch.sum(pseudo_jac, dim=0)
            # Update inner-product and norm
            l_g_pg_inner_prod += torch.sum(
                (1 / n_labeled) * jac * pseudo_jac
            )
            l_pg_norm += torch.sum(
                (1 / n_labeled) * torch.pow(pseudo_jac, 2)
            )
    # Free memory
    del jac, pseudo_jac, jacobians, pseudo_jacobians

    # Mean-gradient mean-pseudo-gradient inner product and mean-pseudo-gradient norm
    l_mg_mpg_inner_prod = torch.tensor(0.0, device=l_inputs.device)
    l_mpg_norm = torch.tensor(0.0, device=l_inputs.device)
    for param_name in params.keys():
        l_mg, l_mpg = l_mean_grad[param_name], l_mean_pseudo_grad[param_name]
        l_mg_mpg_inner_prod += torch.sum(l_mg * l_mpg)
        l_mpg_norm += torch.sum(torch.pow(l_mpg, 2))

    # Get (biased) mean centered inner-product and norm
    l_g_pg_centered_inner_prod = l_g_pg_inner_prod - l_mg_mpg_inner_prod
    l_pg_centered_norm = l_pg_norm - l_mpg_norm
    
    
    # Pseudo-labeled data
    # -------------------
    # Compute mean centered-pseudo-gradient norm as E[|PG|^2] - |E[PG]|^2
    pl_pg_centered_norm = None
    if use_unlabeled_data and (n_pseudolabeled > 0):
        # Init mean pseudo-gradient and pseudo-gradient norm
        pl_mean_pseudo_grad = init_gradient(params)
        pl_pg_norm = torch.tensor(0.0, device=pl_inputs.device)

        # Lower memory requirements by splitting the computation into sequential chunks
        n_pl_chunks = int(np.ceil(n_pseudolabeled / chunk_size))
        for pl_inputs_chunk, pl_pseudolabels_chunk in zip(
            pl_inputs.chunk(n_pl_chunks),
            pl_pseudolabels.chunk(n_pl_chunks)
        ):
            pseudo_jacobians = batch_gradients(
                model=model,
                params=params,
                criterion=criterion,
                inputs=pl_inputs_chunk,
                labels=pl_pseudolabels_chunk
            )
            for param_name in params.keys():
                # Update mean pseudo-gradient and pseudo-gradient norm
                pseudo_jac = pseudo_jacobians[param_name]
                pl_mean_pseudo_grad[param_name] += (1 / n_pseudolabeled) * torch.sum(pseudo_jac, dim=0)
                pl_pg_norm += torch.sum(
                    (1 / n_pseudolabeled) * torch.pow(pseudo_jac, 2)
                )
        # Free memory
        del pseudo_jac, pseudo_jacobians

        # Mean-pseudo-gradient norm
        pl_mpg_norm = torch.tensor(0.0, device=pl_inputs.device)
        for param_name in params.keys():
            pl_mpg_norm += torch.sum(torch.pow(pl_mean_pseudo_grad[param_name], 2))

        # Get (biased) pseudo-gradient mean centered-norm
        pl_pg_centered_norm = pl_pg_norm - pl_mpg_norm
    
    
    # Tuning parameter
    # ----------------
    # Centered inner-prod (numerator)
    l_denom = (n_labeled - 1) if n_labeled > 1 else n_labeled
    g_pg_centered_inner_prod = (n_labeled / l_denom) * l_g_pg_centered_inner_prod
    
    # Centered norm (denominator)
    if pl_pg_centered_norm is None:  # Use only labeled data to get an estimate of the mean pseudo-gradient norm 
        pg_centered_norm =  (n_labeled / l_denom) * l_pg_centered_norm
    else:  # Use labeled and unlabeled data to get an estimate of the mean pseudo-gradient norm
        l_pl_denom = n_labeled + n_pseudolabeled - 1
        pg_centered_norm = (
            ((n_labeled / l_pl_denom) * l_pg_centered_norm) + 
            ((n_pseudolabeled / l_pl_denom) * pl_pg_centered_norm)
        )
    
    # Compute tuning parameter
    r = n_labeled / n_pseudolabeled
    tuning_param = g_pg_centered_inner_prod / ((1 + r) * pg_centered_norm)
    
    # Clip value of tuning parameter to [0, 1]
    tuning_param = torch.clip(tuning_param, min=0.0, max=1.0)
    
    return tuning_param


def compute_optimal_tuning_parameter_per_context(
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
    use_unlabeled_data: bool = False,
    chunk_size: int = None
) -> torch.Tensor:
    """
    :param model: nn.Module
        Prediction model
    :param criterion: nn.Module
        Loss criterion
    :param n_contexts: int
        Number of contexts (contexts must be discrete and go from `0` to `n_contexts - 1`)
    :param l_inputs: torch.Tensor ; shape [N_l, ...]
        Batch of inputs with associated ground-truth labels
    :param l_labels: torch.Tensor ; shape [N_l]
        Batch of ground-truth labels associated to `l_inputs`
    :param l_pseudolabels: torch.Tensor ; shape [N_l]
        Batch of pseudo-labels associated to `l_inputs`
    :param l_contexts: torch.Tensor ; shape [N_l]
        Batch of contexts associated to `l_inputs`
    :param pl_inputs: torch.Tensor ; shape [N_pl, ...]
        Batch of inputs without associated ground-truth labels 
    :param pl_pseudolabels: torch.Tensor ; shape [N_pl]
        Batch of pseudo-labels associated to `pl_inputs`
    :param pl_contexts: torch.Tensor ; shape [N_pl]
        Batch of contexts associated to `pl_inputs`
    :param use_unlabeled_data: bool
        Whether to use the unlabled data to compute the tuning parameter denominator
    :return: torch.Tensor ; shape []
        Tuning parameter vector minimizing next parameter iterate variance for the given sets of labeled and
        unlabeled data
    """
    tuning_param_per_context = []
    for c in range(n_contexts):
        l_mask = l_contexts == c
        pl_mask = pl_contexts == c
        tuning_param = compute_optimal_tuning_parameter(
            model=model,
            criterion=criterion,
            l_inputs=l_inputs[l_mask],
            l_labels=l_labels[l_mask],
            l_pseudolabels=l_pseudolabels[l_mask],
            pl_inputs=pl_inputs[pl_mask],
            pl_pseudolabels=pl_pseudolabels[pl_mask],
            use_unlabeled_data=use_unlabeled_data,
            chunk_size=chunk_size
        )
        tuning_param_per_context.append(tuning_param)
    return torch.stack(tuning_param_per_context, dim=0)
