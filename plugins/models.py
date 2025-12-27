import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from aetheria.core import AetherModel
from aetheria.data import AetherDataModule
from aetheria.registry import Registry

@Registry.register_model("SimpleMLP")
class SimpleMLP(AetherModel):
    def __init__(self, input_dim: int, hidden_dim: int, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = nn.MSELoss()

    def training_step(self, batch):
        x, y = batch
        y_hat = self.net(x)
        return {"loss": self.loss_fn(y_hat, y)}

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss_fn(y_hat, y)
        mae = torch.mean(torch.abs(y_hat - y))
        return {"val_loss": loss, "val_mae": mae}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

class RandomData(AetherDataModule):
    def __init__(self, batch_size): self.bs = batch_size
    def prepare_data(self): pass
    def setup(self):
        self.train = TensorDataset(torch.randn(1000, 10), torch.randn(1000, 1))
        self.val = TensorDataset(torch.randn(200, 10), torch.randn(200, 1))
    def train_dataloader(self): return DataLoader(self.train, batch_size=self.bs)
    def val_dataloader(self): return DataLoader(self.val, batch_size=self.bs)
