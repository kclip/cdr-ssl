from typing import Tuple, List
from omegaconf import OmegaConf, DictConfig
import torch.optim as optim
import torch.nn as nn

from src.data.utils import CustomDataset
from src.data.beamforming import load_beamforming_dataset
from src.data.toy_example import load_toy_example_dataset
from src.models.net import SimpleNet
from src.models.ffnet import FFNet
from src.pretrained.beamforming import BeamformingLoSPretrainedModel
from src.pretrained.toy_example import ToyExamplePretrainedModel
from src.train.criterion import DetectionErrorLoss, AngleCosineLoss, MSESqueezeLoss, AccuracyMetric, BinaryRecallMetric, MacroRecallMetric, \
    BinaryPrecisionMetric, MacroPrecisionMetric, BinaryF1ScoreMetric, MacroF1ScoreMetric
from src.train.bias_estimate_schedule import BiasEstimateScheduleBase, BiasEstimateLinearSchedule, BiasEstimateConstantSchedule


# Models
# ------

def load_model(cfg: DictConfig) -> nn.Module:
    kwargs = OmegaConf.to_container(cfg.model.kwargs, resolve=True)
    
    if cfg.model.type == "FFNet":
        model = FFNet(**kwargs)
    elif cfg.model.type == "SimpleNet":
        model = SimpleNet(**kwargs)
    else:
        raise ValueError(f"Unknown model '{cfg.model.type}'")
    
    return model

def load_pretrained_model(cfg: DictConfig):
    kwargs = OmegaConf.to_container(cfg.pretrained_model.kwargs, resolve=True)

    if cfg.pretrained_model.type == "BeamformingLoSPretrainedModel":
        pretrained_model = BeamformingLoSPretrainedModel(**kwargs)
    elif cfg.pretrained_model.type == "ToyExamplePretrainedModel":
        pretrained_model = ToyExamplePretrainedModel(**kwargs)
    else:
        raise ValueError(f"Unknown pre-trained model '{cfg.pretrained_model.type}'")
    
    return pretrained_model


# Data
# ----

def load_dataset(cfg: DictConfig) -> Tuple[CustomDataset, CustomDataset, CustomDataset]:
    kwargs = OmegaConf.to_container(cfg.dataset.kwargs, resolve=True)
    
    # Load data
    if cfg.dataset.type == "BEAMFORMING":
        dataset_tr, dataset_val, dataset_te = load_beamforming_dataset(
            folder=cfg.dataset.folder,
            validation_ratio=cfg.dataset.validation_ratio,
            **kwargs
        )
    elif cfg.dataset.type == "TOY_EXAMPLE":
        dataset_tr, dataset_val, dataset_te = load_toy_example_dataset(
            folder=cfg.dataset.folder,
            validation_ratio=cfg.dataset.validation_ratio,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset '{cfg.dataset.type}'")
    
    return dataset_tr, dataset_val, dataset_te


# Training
# --------

def load_optimizer(cfg: DictConfig, parameters, outer_optim: bool = False) -> optim.Optimizer:
    cfg_optim = cfg.outer_optimizer if outer_optim else cfg.optimizer
    kwargs = OmegaConf.to_container(cfg_optim.kwargs, resolve=True)
    
    if cfg_optim.type == "SGD":
        optimizer = optim.SGD(
            parameters,
            lr=cfg_optim.lr,
            **kwargs
        )
    elif cfg_optim.type == "Adam":
        optimizer = optim.Adam(
            parameters,
            lr=cfg_optim.lr,
            **kwargs
        )
    elif cfg_optim.type == "AdamW":
        optimizer = optim.AdamW(
            parameters,
            lr=cfg_optim.lr,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer '{cfg_optim.type}'")
    
    return optimizer


def load_lr_schedule(cfg: DictConfig, optimizer: optim.Optimizer, outer_optim: bool = False) -> nn.Module:
    cfg_lr_schedule = cfg.outer_lr_schedule if outer_optim else cfg.lr_schedule
    kwargs = OmegaConf.to_container(cfg_lr_schedule.kwargs, resolve=True)
    
    if cfg_lr_schedule.type is None:
        return None
    elif cfg_lr_schedule.type == "CosineAnnealingLR":
        lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown lr-schedule '{cfg_lr_schedule.type}'")
    
    return lr_schedule


def load_bias_estimate_schedule(cfg: DictConfig) -> BiasEstimateScheduleBase:
    kwargs = OmegaConf.to_container(cfg.bias_estimate_schedule.kwargs, resolve=True)
    
    if cfg.bias_estimate_schedule.type == "BiasEstimateLinearSchedule":
        bias_estimate_schedule = BiasEstimateLinearSchedule(**kwargs)
    elif cfg.bias_estimate_schedule.type == "BiasEstimateConstantSchedule":
        bias_estimate_schedule = BiasEstimateConstantSchedule(**kwargs)
    else:
        raise ValueError(f"Unknown bias estimate-schedule type '{cfg.bias_estimate_schedule.type}'")

    return bias_estimate_schedule


# Losses and metrics
# ------------------

def _load_criterion(cfg_criterion: DictConfig, **kwargs) -> nn.Module:
    # Losses
    if cfg_criterion.type == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(**kwargs)
    if cfg_criterion.type == "MSELoss":
        criterion = nn.MSELoss(**kwargs)
    if cfg_criterion.type == "MSESqueezeLoss":
        criterion = MSESqueezeLoss(**kwargs)
    elif cfg_criterion.type == "DetectionErrorLoss":
        criterion = DetectionErrorLoss(**kwargs)
    elif cfg_criterion.type == "AngleCosineLoss":
        criterion = AngleCosineLoss(**kwargs)
    # Metrics
    elif cfg_criterion.type == "AccuracyMetric":
        criterion = AccuracyMetric(**kwargs)
    elif cfg_criterion.type == "BinaryRecallMetric":
        criterion = BinaryRecallMetric(**kwargs)
    elif cfg_criterion.type == "MacroRecallMetric":
        criterion = MacroRecallMetric(**kwargs)
    elif cfg_criterion.type == "BinaryPrecisionMetric":
        criterion = BinaryPrecisionMetric(**kwargs)
    elif cfg_criterion.type == "MacroPrecisionMetric":
        criterion = MacroPrecisionMetric(**kwargs)
    elif cfg_criterion.type == "BinaryF1ScoreMetric":
        criterion = BinaryF1ScoreMetric(**kwargs)
    elif cfg_criterion.type == "MacroF1ScoreMetric":
        criterion = MacroF1ScoreMetric(**kwargs)
    else:
        raise ValueError(f"Unknown loss '{cfg_criterion.type}'")
    
    return criterion


def load_loss(cfg: DictConfig, reduction: str = None, is_val_loss: bool = False) -> nn.Module:
    cfg_loss = cfg.val_loss if (is_val_loss and (cfg.val_loss is not None)) else cfg.loss
    
    if ("reduction" in cfg_loss.kwargs) and (reduction is not None):
        print(f"Overwritting reduction parameter '{cfg_loss.kwargs.reduction}' in config with value '{reduction}'...")
    
    kwargs = {
        # Config kwargs
        **OmegaConf.to_container(cfg_loss.kwargs, resolve=True),
        # Overwrite config
        **(dict() if reduction is None else {"reduction": reduction})
    }
    
    loss = _load_criterion(cfg_loss, **kwargs)
    
    return loss


def load_metrics(cfg: DictConfig) -> List[nn.Module]:
    metrics = []
    for cfg_metric in cfg.metrics:
        kwargs = OmegaConf.to_container(cfg_metric.kwargs, resolve=True)
        metrics.append(_load_criterion(cfg_metric, **kwargs))
    
    return metrics