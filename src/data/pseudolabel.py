from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from settings import DEVICE
from src.data.utils import CustomDataset, InputsTransformDataset, random_split_indices
from src.pretrained.utils import PretrainedModel


class PseudoLabeledDataset(Dataset):
    def __init__(
        self,
        original_dataset: Dataset,
        pretrain_dataset: Dataset,
        pretrained_model: PretrainedModel,
        return_label: bool,
        preload_pseudolabels: bool = False,
        preload_pseudolabels_batch_size: int = 1,
        preload_pseudolabels_num_workers: int = 1
    ):
        super().__init__()
        # Init
        self.original_dataset = original_dataset
        self._pretrain_dataset = pretrain_dataset
        self._pretrained_model = pretrained_model
        self._return_label = return_label
        
        # Preload pseudo-label predictions
        self._preloaded_pseudolabels = None
        if preload_pseudolabels:
            self.compute_preloaded_pseudolabels(
                batch_size=preload_pseudolabels_batch_size,
                num_workers=preload_pseudolabels_num_workers
            )
            
    
    def compute_preloaded_pseudolabels(self, batch_size: int = 1, num_workers: int = 1):
        pseudolabels = []
        dataloader = DataLoader(
            self._pretrain_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        print("Preloading pseudolabels...")
        for pretrain_inputs, pretrain_labels, pretrain_contexts in tqdm(dataloader):
            with torch.no_grad():
                pretrain_inputs, pretrain_labels, pretrain_contexts = pretrain_inputs.to(DEVICE), pretrain_labels.to(DEVICE), pretrain_contexts.to(DEVICE)
                pretrained_model_preds = self._pretrained_model(
                    inputs=pretrain_inputs,
                    # For label-noise models only
                    labels=pretrain_labels,
                    contexts=pretrain_contexts
                )
                pseudolabels.append(pretrained_model_preds)
        self._preloaded_pseudolabels = torch.cat(pseudolabels, dim=0).cpu()
    
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        input_sample, label_sample, context_sample = self.original_dataset[idx]
        if self._preloaded_pseudolabels is None:
            with torch.no_grad():
                pretrain_inputs, pretrain_labels, pretrain_contexts = self._pretrain_dataset[idx]
                pseudolabel_sample = self._pretrained_model(
                    inputs=pretrain_inputs.unsqueeze(0),
                    # For label-noise models only
                    labels=pretrain_labels.unsqueeze(0),
                    contexts=pretrain_contexts.unsqueeze(0)
                )[0]
        else:
            pseudolabel_sample = self._preloaded_pseudolabels[idx]
        
        if self._return_label:
            return input_sample, label_sample, context_sample, pseudolabel_sample
        else:
            return input_sample, context_sample, pseudolabel_sample


def split_and_pseudolabel_dataset(    
    original_dataset: CustomDataset,
    pretrained_model: PretrainedModel,
    labeled_ratio: float,
    preload_pseudolabels: bool = False,
    preload_pseudolabels_batch_size: int = 1,
    preload_pseudolabels_num_workers: int = 1,
    deterministic_labeled_split: bool = False
) -> Tuple[PseudoLabeledDataset, PseudoLabeledDataset]:
    # Build dataset for pretrained_model
    pretrain_dataset = InputsTransformDataset(
        original_dataset.get_raw_dataset(),
        transform=pretrained_model.inputs_transform
    )
    
    # Get labeled/unlabeled dataset split
    n_dataset = len(original_dataset)
    n_labeled = int(round(n_dataset * labeled_ratio))
    if deterministic_labeled_split:
        indices = np.arange(n_dataset)
        labeled_indices, unlabeled_indices = indices[:n_labeled], indices[n_labeled:]
    else:
        labeled_indices, unlabeled_indices = random_split_indices([n_labeled, n_dataset - n_labeled])
    
    # Labeled dataset with pseudo-labels
    if labeled_ratio > 0.0:
        labeled_original_dataset = Subset(original_dataset, indices=labeled_indices)
        labeled_pretrain_dataset = Subset(pretrain_dataset, indices=labeled_indices)
        labeled_pseudolabeled_dataset = PseudoLabeledDataset(
            original_dataset=labeled_original_dataset,
            pretrain_dataset=labeled_pretrain_dataset,
            pretrained_model=pretrained_model,
            return_label=True,
            preload_pseudolabels=preload_pseudolabels,
            preload_pseudolabels_batch_size=preload_pseudolabels_batch_size,
            preload_pseudolabels_num_workers=preload_pseudolabels_num_workers
        )
    else:
        labeled_pseudolabeled_dataset = None
    
    # Unlabeled dataset with pseudo-labels
    if labeled_ratio < 1.0:
        unlabeled_original_dataset = Subset(original_dataset, indices=unlabeled_indices)
        unlabeled_pretrain_dataset = Subset(pretrain_dataset, indices=unlabeled_indices)
        unlabeled_pseudolabeled_dataset = PseudoLabeledDataset(
            original_dataset=unlabeled_original_dataset,
            pretrain_dataset=unlabeled_pretrain_dataset,
            pretrained_model=pretrained_model,
            return_label=False,
            preload_pseudolabels=preload_pseudolabels,
            preload_pseudolabels_batch_size=preload_pseudolabels_batch_size,
            preload_pseudolabels_num_workers=preload_pseudolabels_num_workers
        )
    else:
        unlabeled_pseudolabeled_dataset = None
    
    return labeled_pseudolabeled_dataset, unlabeled_pseudolabeled_dataset


class GroundTruthPseudolabelsDataset(Dataset):
    def __init__(
        self,
        pseudolabeled_dataset: PseudoLabeledDataset
    ):
        super().__init__()
        self._pseudolabeled_dataset = pseudolabeled_dataset
    
    def __len__(self):
        return len(self._pseudolabeled_dataset)

    def __getitem__(self, idx):
        input_sample, context_sample, pseudolabel_sample = self._pseudolabeled_dataset[idx]
        return input_sample, pseudolabel_sample, context_sample, pseudolabel_sample