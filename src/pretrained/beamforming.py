import torch

from src.pretrained.utils import PretrainedModel


class BeamformingLoSPretrainedModel(PretrainedModel):
    BS_LOCATION = [173.0, 112.0, 73.3355]
    
    def __init__(self):
        super().__init__()
        self._bs_loc = torch.tensor(self.BS_LOCATION, dtype=torch.float32).reshape(1, -1)
    
    def forward(self, inputs, labels=None, contexts=None) -> torch.Tensor:
        """
        :param inputs: torch.Tensor ; torch.float ; shape [N, 3]
            Device locations
        :param labels:  torch.Tensor ; torch.float ; shape [N, 2]
            [PLACEHOLDER] Ground-truth AoDs. Each line is an AoD tuple (elevation, azimuth) in [rad]
        :param contexts:  torch.Tensor ; torch.int ; shape [N]
            [PLACEHOLDER] Binary context variable: 0 -> NLoS location ; 1 -> LoS location
        :return: torch.Tensor ; torch.float ; shape [N, 2]
            Noisy ground-truth AoDs
        """
        # Compute LoS angles
        los_directions = inputs - self._bs_loc.to(inputs.device)
        los_directions /= torch.linalg.vector_norm(los_directions, dim=1).reshape(-1, 1)
        los_elevations = torch.acos(los_directions[:, 2])
        los_azimuths = torch.atan2(los_directions[:, 1], los_directions[:, 0])
        los_labels = torch.stack([los_elevations, los_azimuths], dim=-1)
        
        return los_labels
