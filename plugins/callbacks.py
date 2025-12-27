from aetheria.core import Callback
from loguru import logger
import os

class ModelCheckpoint(Callback):
    def __init__(self, dir_path="checkpoints"):
        self.dir = dir_path
        os.makedirs(self.dir, exist_ok=True)

    def on_epoch_end(self, orchestrator, metrics, **kwargs):
        # Uses the new orchestrator.save_snapshot capability
        if orchestrator.accelerator.is_main_process:
            path = f"{self.dir}/snapshot_ep{orchestrator.current_epoch}.pt"
            orchestrator.save_snapshot(path)

class EarlyStopping(Callback):
    def __init__(self, monitor="val_loss", patience=3):
        self.monitor = monitor
        self.patience = patience
        self.best = float('inf')
        self.counter = 0

    def on_epoch_end(self, orchestrator, metrics, **kwargs):
        current = metrics.get(self.monitor)
        if current is None: return
        if current < self.best:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.warning("Early Stopping Triggered")
                orchestrator.stop_training = True
