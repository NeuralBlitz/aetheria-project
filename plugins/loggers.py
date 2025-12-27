from aetheria.core import Logger
from loguru import logger

class ConsoleLogger(Logger):
    def log_metrics(self, metrics, step):
        msg = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Global Step {step} | {msg}")

class WandbLogger(Logger):
    def __init__(self, project):
        import wandb
        self.wandb = wandb
        self.project = project
        self._initialized = False

    def on_train_start(self, orchestrator, **kwargs):
        if orchestrator.accelerator.is_main_process and not self._initialized:
            self.wandb.init(project=self.project, config=orchestrator.config.dict())
            self._initialized = True

    def log_metrics(self, metrics, step):
        if self._initialized:
            self.wandb.log(metrics, step=step)
