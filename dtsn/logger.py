# dtsn/logger.py
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    """Thin wrapper so the rest of the code never imports SummaryWriter."""

    def __init__(self, run_name: str = "dtsn"):
        Path("runs").mkdir(exist_ok=True)
        self.w = SummaryWriter(log_dir=f"runs/{run_name}")

    def log(self, step: int, **scalars):
        """log(loss=..., q_loss=..., ...)"""
        for k, v in scalars.items():
            self.w.add_scalar(k, float(v), step)

    def close(self):
        self.w.flush()
        self.w.close()
