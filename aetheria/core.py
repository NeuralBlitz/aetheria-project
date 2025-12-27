from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, Dict, Union, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .orchestrator import Orchestrator

# Optimizer can return just the optimizer, or (optimizer, scheduler)
OptimizerConfig = Union[torch.optim.Optimizer, Tuple[torch.optim.Optimizer, Optional[Any]]]

class AetherModel(nn.Module, ABC):
    @abstractmethod
    def training_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Must return dict with at least 'loss'."""
        pass

    @abstractmethod
    def validation_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Must return dict of validation metrics."""
        pass

    @abstractmethod
    def configure_optimizers(self) -> OptimizerConfig:
        """Model owns its hyperparameters."""
        pass

class Callback(ABC):
    def on_train_start(self, orchestrator: "Orchestrator"): pass
    def on_train_end(self, orchestrator: "Orchestrator"): pass
    def on_epoch_start(self, orchestrator: "Orchestrator"): pass
    def on_epoch_end(self, orchestrator: "Orchestrator", metrics: Dict[str, float]): pass
    def on_batch_end(self, orchestrator: "Orchestrator", batch_idx: int, metrics: Dict[str, float]): pass
    def on_validation_start(self, orchestrator: "Orchestrator"): pass
    def on_validation_end(self, orchestrator: "Orchestrator", metrics: Dict[str, float]): pass

class Logger(Callback):
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int): pass

    def on_batch_end(self, orchestrator, batch_idx, metrics):
        self.log_metrics(metrics, step=orchestrator.global_step)

    def on_epoch_end(self, orchestrator, metrics):
        # Prefix epoch metrics to distinguish from batch metrics
        epoch_metrics = {f"epoch_{k}": v for k, v in metrics.items()}
        epoch_metrics["epoch"] = orchestrator.current_epoch
        self.log_metrics(epoch_metrics, step=orchestrator.global_step)
