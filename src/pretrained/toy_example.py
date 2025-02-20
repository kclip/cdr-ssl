import numpy as np
import torch

from src.pretrained.utils import PretrainedModel


class ToyExamplePretrainedModel(PretrainedModel):
    
    def forward(self, inputs, labels=None, contexts=None) -> torch.Tensor:
        """
        :param inputs: torch.Tensor ; torch.float ; shape [N]
            Toy example input
        :param labels:  torch.Tensor ; torch.float ; shape [N]
            [PLACEHOLDER]
        :param contexts:  torch.Tensor ; torch.int ; shape [N]
            [PLACEHOLDER]
        :return: torch.Tensor ; torch.float ; shape [N, 1]
            Predicted output
        """
        w_neg = 2 * np.pi
        v_neg = 4 * np.pi
        w_pos = 3 * np.pi
        v_pos = 5 * np.pi
        return 0.5 * inputs * (
            (inputs < 0).type(torch.float32) * (
                torch.cos(w_neg * inputs) - torch.sin(v_neg * inputs)
            ) +
            (inputs >= 0).type(torch.float32) * (
                torch.cos(w_pos * inputs) - torch.sin(v_pos * inputs)
            )
        )
