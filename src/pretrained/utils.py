import torch
import torch.nn as nn


class PretrainedModel(nn.Module):
    
    @property
    def inputs_transform(self):
        return None
    
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        # Note: `labels` and `contexts` are given to allow for pseudo-models that simulate a pretrained model
        # by adding noise to ground-truth labels
        raise NotImplementedError
