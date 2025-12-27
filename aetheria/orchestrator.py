import torch
from loguru import logger
from collections import defaultdict
from typing import List, Optional, Dict
from .core import AetherModel, Callback
from .data import AetherDataModule
from .config import TrainingConfig
from .accelerator import Accelerator, GPUAccelerator

class Orchestrator:
    def __init__(self, 
                 model: AetherModel, 
                 data: AetherDataModule, 
                 config: TrainingConfig, 
                 accelerator: Optional[Accelerator] = None,
                 callbacks: Optional[List[Callback]] = None):
        
        self.config = config
        self.accelerator = accelerator if accelerator else GPUAccelerator()
        self.callbacks = callbacks or []
        
        # Optimizer Injection (IoC)
        opt_conf = model.configure_optimizers()
        if isinstance(opt_conf, tuple): self.raw_optimizer, self.scheduler = opt_conf
        else: self.raw_optimizer, self.scheduler = opt_conf, None
            
        self.model, self.optimizer = self.accelerator.setup(model, self.raw_optimizer)
        
        # Attach optimizer to accelerator for unscaling logic
        # (A slight hack for simplicity in clip_grad_norm)
        self.accelerator.optimizer = self.optimizer 
        
        self.data = data
        self.current_epoch = 0
        self.global_step = 0
        self.stop_training = False

    def _run_hook(self, hook: str, **kwargs):
        """Safely broadcast events to callbacks."""
        kwargs['_is_main_process'] = self.accelerator.is_main_process
        for cb in self.callbacks:
            method = getattr(cb, hook, None)
            if method:
                try: method(self, **kwargs)
                except Exception as e: logger.error(f"Callback Error ({cb.__class__.__name__}): {e}")

    def save_snapshot(self, path: str):
        """Fault Tolerance: Saves the entire universe state."""
        if not self.accelerator.is_main_process: return
        snapshot = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'rng_torch': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
        torch.save(snapshot, path)
        logger.info(f"Snapshot saved: {path}")

    def resume_from(self, path: str):
        """Fault Tolerance: Restores universe."""
        logger.info(f"Resuming from: {path}")
        snap = torch.load(path, map_location=self.accelerator.device)
        self.current_epoch = snap['epoch']
        self.global_step = snap['global_step']
        self.model.load_state_dict(snap['model_state'])
        self.optimizer.load_state_dict(snap['optimizer_state'])
        if self.scheduler and snap['scheduler_state']:
            self.scheduler.load_state_dict(snap['scheduler_state'])
        torch.set_rng_state(snap['rng_torch'])
        if snap['rng_cuda'] is not None: torch.cuda.set_rng_state(snap['rng_cuda'])

    def _run_validation(self) -> Dict[str, float]:
        self.model.eval()
        self._run_hook('on_validation_start')
        loader = self.accelerator.prepare_dataloader(self.data.val_dataloader())
        metrics_agg = defaultdict(float)
        
        with torch.no_grad():
            for batch in loader:
                batch = self.accelerator.process_batch(batch)
                with self.accelerator.forward_context():
                    out = self.model.validation_step(batch)
                
                for k, v in out.items():
                    metrics_agg[k] += self.accelerator.reduce_metric(v)

        avg = {k: v / len(loader) for k, v in metrics_agg.items()}
        self._run_hook('on_validation_end', metrics=avg)
        self.model.train()
        return avg

    def run(self, resume_path: Optional[str] = None):
        if self.accelerator.is_main_process: self.data.prepare_data()
        self.data.setup()
        
        train_loader = self.accelerator.prepare_dataloader(self.data.train_dataloader())
        accum_steps = self.config.grad_accumulation_steps
        
        if resume_path: self.resume_from(resume_path)

        self._run_hook('on_train_start')
        self.model.train()

        for epoch in range(self.current_epoch, self.config.epochs):
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            if self.stop_training: break
            self.current_epoch = epoch
            self._run_hook('on_epoch_start')
            
            epoch_metrics = defaultdict(float)
            self.optimizer.zero_grad()
            
            for i, batch in enumerate(train_loader):
                batch = self.accelerator.process_batch(batch)
                
                with self.accelerator.forward_context():
                    out = self.model.training_step(batch)
                    loss = out["loss"] / accum_steps

                # --- RESILIENCE LAYER ---
                if self.accelerator.check_nan(loss):
                    logger.warning(f"NaN detected at Epoch {epoch} Step {i}. Skipping.")
                    self.optimizer.zero_grad()
                    continue
                # -----------------------

                self.accelerator.backward(loss)
                
                if (i + 1) % accum_steps == 0:
                    if self.config.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm(self.model.parameters(), self.config.max_grad_norm)
                    
                    self.accelerator.step(self.optimizer)
                    self.optimizer.zero_grad()

                reduced_loss = self.accelerator.reduce_metric(out["loss"])
                epoch_metrics["loss"] += reduced_loss
                self.global_step += 1
                
                safe_metrics = {k: v.detach().cpu().item() for k, v in out.items() if k != "loss"}
                safe_metrics["loss"] = reduced_loss
                for k, v in safe_metrics.items(): 
                    if k != "loss": epoch_metrics[k] += v

                if self.accelerator.is_main_process:
                    self._run_hook('on_batch_end', batch_idx=i, metrics=safe_metrics)

            if self.scheduler: self.scheduler.step()
            
            avg_train = {k: v / len(train_loader) for k, v in epoch_metrics.items()}
            
            if (epoch + 1) % self.config.val_interval == 0:
                val_metrics = self._run_validation()
                avg_train.update(val_metrics)

            if self.accelerator.is_main_process:
                self._run_hook('on_epoch_end', metrics=avg_train)

        self._run_hook('on_train_end')
        self.accelerator.cleanup()
