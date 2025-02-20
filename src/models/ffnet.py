import torch.nn as nn

from src.models.utils import FourierFeatures


class FFNet(nn.Module):
    def __init__(self, 
        n_inputs_dim: int = 3,
        n_outputs_dim: int = 2,
        n_freqs_emb: int = 20,
        max_freq_emb: float = 20.0
    ):
        n_outputs_embedding = 2 * n_freqs_emb * n_inputs_dim
        super().__init__()
        self.net = nn.Sequential(
            FourierFeatures(n_frequencies=n_freqs_emb, max_frequency=max_freq_emb),
            nn.Linear(n_outputs_embedding, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs_dim),
        )


    def forward(self, inputs):
        return self.net(inputs)