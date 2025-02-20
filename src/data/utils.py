from typing import List, Any
import numpy as np
from torch.utils.data import Dataset

from settings import NUMPY_RNG


def random_split_indices(splits: List[int]) -> List[np.ndarray]:
    rand_perm = NUMPY_RNG.permutation(np.sum(splits))
    indices = []
    offset = 0
    for s in splits:
        indices.append(rand_perm[offset:(offset + s)])
        offset += s
    return indices


class RawDataset(Dataset):
    def __init__(self, inputs: Any, labels: Any, contexts: Any):
        super().__init__()
        self._inputs = inputs
        self._labels = labels
        self._contexts = contexts
    
    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, idx):
        return self._inputs[idx], self._labels[idx], self._contexts[idx]


class InputsTransformDataset(Dataset):
    def __init__(self, original_dataset, transform = None):
        super().__init__()
        self._original_dataset = original_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self._original_dataset)

    def __getitem__(self, idx):
        input_sample, label_sample, context_sample = self._original_dataset[idx]
        if self.transform:
            input_sample = self.transform(input_sample)
        return input_sample, label_sample, context_sample


class CustomDataset(Dataset):
    @property
    def label_names(self):
        raise NotImplementedError

    @property
    def context_names(self):
        raise NotImplementedError
    
    def get_raw_dataset(self) -> RawDataset:
        raise NotImplementedError
        
    @classmethod
    def inputs_augmentation_transform(cls):
        return None
    
    @classmethod
    def inputs_normalization_transform(cls):
        return None
