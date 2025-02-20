import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, 
        n_inputs_dim: int = 1,
        n_outputs_dim: int = 1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_outputs_dim),
        )

    def forward(self, inputs):
        return self.net(inputs)
