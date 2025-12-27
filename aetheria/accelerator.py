import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from abc import ABC, abstractmethod
from contextlib import contextmanager
import os
from .utils import recursive_to_device

class Accelerator(ABC):
    @property
    @abstractmethod
    def is_main_process(self) -> bool: pass
    @abstractmethod
    def setup(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer): pass
    @abstractmethod
    def prepare_dataloader(self, loader: DataLoader) -> DataLoader: pass
    @abstractmethod
    def backward(self, loss: torch.Tensor): pass
    @abstractmethod
    def step(self, optimizer: torch.optim.Optimizer): pass
    @abstractmethod
    def reduce_metric(self, tensor: torch.Tensor) -> float: pass
    @abstractmethod
    def forward_context(self): pass
    @abstractmethod
    def clip_grad_norm(self, parameters, max_norm: float): pass
    @abstractmethod
    def check_nan(self, loss: torch.Tensor) -> bool: pass
    def cleanup(self): pass

class GPUAccelerator(Accelerator):
    def __init__(self, device_index: int = 0, mixed_precision: bool = False):
        self.device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    @property
    def is_main_process(self) -> bool: return True

    def setup(self, model, optimizer): return model.to(self.device), optimizer

    def prepare_dataloader(self, loader): return loader

    def process_batch(self, batch): return recursive_to_device(batch, self.device)

    @contextmanager
    def forward_context(self):
        with torch.cuda.amp.autocast(enabled=self.mixed_precision): yield

    def backward(self, loss): self.scaler.scale(loss).backward()
    
    def step(self, optimizer): 
        self.scaler.step(optimizer)
        self.scaler.update()

    def reduce_metric(self, tensor): return tensor.item()

    def clip_grad_norm(self, parameters, max_norm):
        if self.mixed_precision: self.scaler.unscale_(self.optimizer) # Assumes optimizer attached
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def check_nan(self, loss):
        return not torch.isfinite(loss).item()

class DDPAccelerator(Accelerator):
    def __init__(self, mixed_precision: bool = False):
        dist.init_process_group(backend="nccl")
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    @property
    def is_main_process(self) -> bool: return self.local_rank == 0

    def setup(self, model, optimizer):
        model = model.to(self.device)
        return DDP(model, device_ids=[self.local_rank]), optimizer

    def prepare_dataloader(self, loader):
        sampler = DistributedSampler(loader.dataset, shuffle=True)
        return DataLoader(loader.dataset, batch_size=loader.batch_size, 
                          sampler=sampler, num_workers=loader.num_workers)

    def process_batch(self, batch): return recursive_to_device(batch, self.device)

    @contextmanager
    def forward_context(self):
        with torch.cuda.amp.autocast(enabled=self.mixed_precision): yield

    def backward(self, loss): self.scaler.scale(loss).backward()

    def step(self, optimizer): 
        self.scaler.step(optimizer)
        self.scaler.update()

    def reduce_metric(self, tensor):
        rt = tensor.detach().clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt.item() / self.world_size

    def clip_grad_norm(self, parameters, max_norm):
        if self.mixed_precision: self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def check_nan(self, loss):
        # Synchronized check: if ANY rank has NaN, all must know
        is_nan = torch.tensor(1.0 if not torch.isfinite(loss) else 0.0, device=self.device)
        dist.all_reduce(is_nan, op=dist.ReduceOp.MAX)
        return is_nan.item() > 0.5

    def cleanup(self): dist.destroy_process_group()
