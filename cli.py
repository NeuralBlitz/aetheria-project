import typer
import os
import torch
from aetheria.config import TrainingConfig
from aetheria.orchestrator import Orchestrator
from aetheria.accelerator import DDPAccelerator, GPUAccelerator
from aetheria.registry import Registry
from aetheria.inference import Predictor

# Import plugins for registration
import plugins.models
import plugins.loggers as loggers
import plugins.callbacks as cbs

app = typer.Typer()

@app.command()
def train(config_path: str, resume: str = ""):
    conf = TrainingConfig.from_yaml(config_path)
    
    ModelClass = Registry.get_model(conf.model_name)
    model = ModelClass(**conf.model_params)
    
    from plugins.models import RandomData 
    data = RandomData(**conf.data_params)

    acc = DDPAccelerator(mixed_precision=conf.mixed_precision) if "WORLD_SIZE" in os.environ \
          else GPUAccelerator(mixed_precision=conf.mixed_precision)

    callbacks = [
        loggers.ConsoleLogger(),
        cbs.ModelCheckpoint(),
        cbs.EarlyStopping()
    ]

    orch = Orchestrator(model, data, conf, accelerator=acc, callbacks=callbacks)
    
    resume_path = resume if resume else None
    orch.run(resume_path=resume_path)

@app.command()
def predict(config_path: str, checkpoint: str):
    conf = TrainingConfig.from_yaml(config_path)
    predictor = Predictor(conf.model_name, conf.model_params, checkpoint)
    print(predictor.predict(torch.randn(1, 10)))

if __name__ == "__main__":
    app()
