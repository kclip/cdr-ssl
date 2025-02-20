import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.utils.save import SafeOpen


def create_tensorboard_writer(tb_dir: str) -> SummaryWriter:
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(tb_dir, current_time)
    return SummaryWriter(log_dir=log_dir)


class CheckpointManager(object):
    def __init__(self, directory: str, epoch: int = 0, checkpoint_freq: int = -1, log_best: bool = True):
        """
        :param directory: str
            Run directory. Checkpoints are saved in {directory}/checkpoints/checkpoint_*.pth
        :param epoch: int
            Initial epoch. Should be strictly positive only when loading from a previous checkpoint.
        :param checkpoint_freq: int
            Frequency of checkpoints in epochs. If set to `-1`, do not log checkpoints
        :param log_best: bool
            Log checkpoint at lowest validation-loss epoch
        """
        self._dir = directory
        self._checkpoint_dir = os.path.join(self._dir, "checkpoints")
        self._epoch = epoch
        self._checkpoint_freq = checkpoint_freq
        self._log_best = log_best
        self._best_loss = None
        self._best_epoch = None

    def _checkpoint_filepath(self, epoch: int = None, best_checkpoint: bool = False):
        checkpoint_filename = f"best_checkpoint.pth" if best_checkpoint else f"checkpoint_{epoch:0>3}.pth"
        return os.path.join(self._checkpoint_dir, checkpoint_filename)
    
    def load_current_checkpoint(self) -> dict:
        filepath = self._checkpoint_filepath(epoch=self._epoch)
        return torch.load(filepath, weights_only=True)
    
    def load_best_checkpoint(self) -> dict:
        if self._best_epoch is None:
            raise ValueError("Best model checkpoint was not found...")
        filepath = self._checkpoint_filepath(best_checkpoint=True)
        return torch.load(filepath, weights_only=True)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        outer_optimizer: optim.Optimizer = None,
        loss_tr: float = None,
        outer_loss_tr: float = None,
        loss_val: float = None,
        overwrite: bool = False
    ):
        # Gather data
        checkpoint = {
            "epoch": self._epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        if outer_optimizer is not None:
            checkpoint["outer_optimizer_state_dict"] = outer_optimizer.state_dict()
        if loss_tr is not None:
            checkpoint["loss_tr"] = loss_tr
        if outer_loss_tr is not None:
            checkpoint["outer_loss_tr"] = outer_loss_tr
        if loss_val is not None:
            checkpoint["loss_val"] = loss_val
            # Update best checkpoint
            if self._log_best and (self._best_loss is None) or (self._best_loss > loss_val):
                self._best_loss = loss_val
                self._best_epoch = self._epoch
                filepath = self._checkpoint_filepath(best_checkpoint=True)
                with SafeOpen(filepath, "wb", overwrite=True) as f:
                    torch.save(checkpoint, f)
        # Save
        if (self._checkpoint_freq > 0) and (((self._epoch + 1) % self._checkpoint_freq) == 0):
            filepath = self._checkpoint_filepath(epoch=self._epoch)
            with SafeOpen(filepath, "wb", overwrite=overwrite) as f:
                torch.save(checkpoint, f)
        # Update epoch counter
        self._epoch += 1
