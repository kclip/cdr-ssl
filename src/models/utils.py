import numpy as np
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, n_frequencies: int, max_frequency: float):
        super().__init__()
        self._max_frequency = torch.as_tensor(max_frequency)
        self._n_frequencies = n_frequencies
        self._frequencies = self._max_frequency ** (
            torch.arange(self._n_frequencies, dtype=torch.float32) / self._n_frequencies
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: torch.Tensor ; torch.float ; shape [..., N_FEATURES]
            Input tensor
        :return: torch.Tensor ; torch.float ; shape [..., 2 * N_FREQS * N_FEATURES]
            Fourier features embedding
        """
        inputs_shape = list(inputs.shape)
        frequencies = self._frequencies.to(inputs.device).reshape(([1] * len(inputs_shape)) + [self._n_frequencies])
        pi = torch.tensor(np.pi, device=inputs.device)
        embedding_base = (
            2 * pi * inputs.reshape([*inputs_shape, 1]) * frequencies
        ).reshape(
            inputs_shape[:-1] + [inputs_shape[-1] * self._n_frequencies]
        )
        return torch.cat(
            [torch.cos(embedding_base), torch.sin(embedding_base)],
            axis=-1
        )
