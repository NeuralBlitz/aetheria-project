from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class AetherDataModule(ABC):
    @abstractmethod
    def prepare_data(self): 
        """Download or tokenize data. Runs only on main process."""
        pass
    
    @abstractmethod
    def setup(self): 
        """Split data or load artifacts. Runs on all processes."""
        pass
    
    @abstractmethod
    def train_dataloader(self) -> DataLoader: pass
    
    @abstractmethod
    def val_dataloader(self) -> DataLoader: pass
