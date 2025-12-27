from typing import Dict, Type, Any

class Registry:
    _models: Dict[str, Type[Any]] = {}

    @classmethod
    def register_model(cls, name: str):
        def decorator(model_cls):
            cls._models[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get_model(cls, name: str) -> Type[Any]:
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not registered. Available: {list(cls._models.keys())}")
        return cls._models[name]
