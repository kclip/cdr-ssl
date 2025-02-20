import os
import numpy as np
import torch


PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))


DEVICE = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
SEED = os.getenv("SEED", None)
NUMPY_RNG = np.random.default_rng(seed=SEED)
TORCH_RNG = torch.Generator(device=DEVICE)
TORCH_RNG_CPU = TORCH_RNG if DEVICE == "cpu" else torch.Generator(device="cpu")
if SEED is not None:
    TORCH_RNG.manual_seed(SEED)
    TORCH_RNG_CPU.manual_seed(SEED)
