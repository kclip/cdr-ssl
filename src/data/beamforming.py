import os
from typing import Tuple
import numpy as np
import torch

from src.data.utils import CustomDataset, RawDataset, random_split_indices


class BeamformingDataset(CustomDataset):
    MIN_INPUTS = [10., 10., 35.]
    MAX_INPUTS = [361., 361., 60.]
        
    def __init__(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        contexts: np.ndarray
    ):
        super().__init__()

        # Data
        self._inputs = torch.from_numpy(inputs)
        self._labels = torch.from_numpy(labels)
        self._contexts = torch.from_numpy(contexts)
        self._context_names = ["NLoS", "LoS"]
        
        # Transform
        self._inputs_normalization_transform = self.get_inputs_normalization_transform()

    @classmethod
    def get_inputs_normalization_transform(cls):
        min_inputs = torch.tensor(cls.MIN_INPUTS, dtype=torch.float32)
        max_inputs = torch.tensor(cls.MAX_INPUTS, dtype=torch.float32)
        inputs_shift = (min_inputs + max_inputs) / 2
        inputs_scale = 1 / (max_inputs - min_inputs)
        return (lambda x: (x - inputs_shift) * inputs_scale)
    
    @property
    def label_names(self):
        return None

    @property
    def context_names(self):
        return self._context_names
    
    def get_raw_dataset(self) -> RawDataset:
        return RawDataset(inputs=self._inputs, labels=self._labels, contexts=self._contexts)

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, idx):
        input_sample = self._inputs_normalization_transform(self._inputs[idx])
        return input_sample, self._labels[idx], self._contexts[idx]


def load_beamforming_dataset(
    folder: str,
    validation_ratio: float = 0.0
) -> Tuple[BeamformingDataset, BeamformingDataset, BeamformingDataset]:
    # Load data
    dataset_tr_val = np.load(os.path.join(folder, "train_dataset.npz"))
    inputs_tr_val = dataset_tr_val["rx_positions"].astype(np.float32)
    labels_tr_val = dataset_tr_val["aod_strongest_path"].astype(np.float32)
    contexts_tr_val = dataset_tr_val["rx_los"].astype(np.int64).reshape(-1)
    
    dataset_te = np.load(os.path.join(folder, "test_dataset.npz"))
    inputs_te = dataset_te["rx_positions"].astype(np.float32)
    labels_te = dataset_te["aod_strongest_path"].astype(np.float32)
    contexts_te = dataset_te["rx_los"].astype(np.int64).reshape(-1)
    
    # Train/Validation/Test split
    n_tr_val = len(inputs_tr_val)
    n_val = round(validation_ratio * n_tr_val)
    n_train = n_tr_val - n_val
    indices_tr, indices_val = random_split_indices([n_train, n_val])
    inputs_tr, inputs_val = inputs_tr_val[indices_tr], inputs_tr_val[indices_val]
    labels_tr, labels_val = labels_tr_val[indices_tr], labels_tr_val[indices_val]
    contexts_tr, contexts_val = contexts_tr_val[indices_tr], contexts_tr_val[indices_val]
    
    # Init dataset classes
    dataset_tr = BeamformingDataset(
        inputs=inputs_tr,
        labels=labels_tr,
        contexts=contexts_tr
    )
    dataset_val = BeamformingDataset(
        inputs=inputs_val,
        labels=labels_val,
        contexts=contexts_val
    )
    dataset_te = BeamformingDataset(
        inputs=inputs_te,
        labels=labels_te,
        contexts=contexts_te
    )

    return dataset_tr, dataset_val, dataset_te
