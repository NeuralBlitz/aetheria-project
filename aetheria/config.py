import yaml
from pydantic import BaseModel, Field
from typing import Dict, Any

class TrainingConfig(BaseModel):
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    val_interval: int = 1
    
    # Scaling & Safety
    mixed_precision: bool = False
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 0.0
    
    # Plugins
    model_name: str
    model_params: Dict[str, Any] = Field(default_factory=dict)
    data_params: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path, "r") as f:
            return cls(**yaml.safe_load(f))
