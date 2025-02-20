import os
from typing import Tuple
import numpy as np
import torch

from src.data.utils import CustomDataset, RawDataset, random_split_indices


class ToyExampleDataset(CustomDataset):
    def __init__(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        contexts: np.ndarray
    ):
        super().__init__()

        # Data
        self._inputs = torch.from_numpy(inputs.reshape(-1, 1))
        self._labels = torch.from_numpy(labels.reshape(-1, 1))
        self._contexts = torch.from_numpy(contexts.reshape(-1))
        self._context_names = ["x < 0", "x >= 0"]
    
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
        return self._inputs[idx], self._labels[idx], self._contexts[idx]


def load_toy_example_dataset(
    folder: str,
    validation_ratio: float = 0.0,
    deterministic_validation_split: bool = True
) -> Tuple[ToyExampleDataset, ToyExampleDataset, ToyExampleDataset]:
    # Load data
    dataset = np.load(os.path.join(folder, "toy_example_dataset.npz"))
    
    inputs_tr_val = dataset["inputs_tr"].astype(np.float32)
    labels_tr_val = dataset["outputs_tr"].astype(np.float32)
    contexts_tr_val = (inputs_tr_val >= 0).astype(np.int64)
    
    inputs_te = dataset["inputs_te"].astype(np.float32)
    labels_te = dataset["outputs_te"].astype(np.float32)
    contexts_te = (inputs_te >= 0).astype(np.int64)
    
    # Train/Validation/Test split
    n_tr_val = len(inputs_tr_val)
    n_tr = n_tr_val - round(validation_ratio * n_tr_val)
    if deterministic_validation_split:
        indices = np.arange(n_tr_val)
        indices_tr, indices_val = indices[:n_tr], indices[n_tr:]
    else:
        indices_tr, indices_val = random_split_indices([n_tr, n_tr_val - n_tr])
    inputs_tr, inputs_val = inputs_tr_val[indices_tr], inputs_tr_val[indices_val]
    labels_tr, labels_val = labels_tr_val[indices_tr], labels_tr_val[indices_val]
    contexts_tr, contexts_val = contexts_tr_val[indices_tr], contexts_tr_val[indices_val]
    
    # Init dataset classes
    dataset_tr = ToyExampleDataset(
        inputs=inputs_tr,
        labels=labels_tr,
        contexts=contexts_tr
    )
    dataset_val = ToyExampleDataset(
        inputs=inputs_val,
        labels=labels_val,
        contexts=contexts_val
    )
    dataset_te = ToyExampleDataset(
        inputs=inputs_te,
        labels=labels_te,
        contexts=contexts_te
    )

    return dataset_tr, dataset_val, dataset_te
