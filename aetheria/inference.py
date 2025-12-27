import torch
from .registry import Registry
from .utils import recursive_to_device

class Predictor:
    def __init__(self, model_name: str, model_params: dict, checkpoint_path: str, device: str = "cpu"):
        self.device = device
        ModelClass = Registry.get_model(model_name)
        self.model = ModelClass(**model_params)
        
        # Load weights, stripping DDP prefix if needed
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Handle full snapshot vs just weights
        state = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt))
        
        clean_state = {k.replace("module.", ""): v for k, v in state.items()}
        
        self.model.load_state_dict(clean_state)
        self.model.to(device).eval()

    def predict(self, input_data):
        input_data = recursive_to_device(input_data, self.device)
        with torch.no_grad():
            return self.model(input_data)
