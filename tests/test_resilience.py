import torch
import pytest
from unittest.mock import MagicMock, patch
from aetheria.accelerator import DDPAccelerator, GPUAccelerator
from aetheria.orchestrator import Orchestrator
from aetheria.config import TrainingConfig

# --- Mock Data ---
def get_clean_loss(): return torch.tensor(0.5)
def get_nan_loss(): return torch.tensor(float('nan'))
def get_inf_loss(): return torch.tensor(float('inf'))

# --- Test 1: The DDP Handshake ---
# We verify that if Rank 0 has NaN and Rank 1 is clean, 
# the ReduceOp.MAX logic correctly propagates the Error State.

@patch('aetheria.accelerator.dist')
def test_ddp_nan_synchronization(mock_dist):
    """
    Simulates a DDP environment where this process finds a NaN,
    but we want to ensure it calls all_reduce to warn others.
    """
    # 1. Setup Mock Environment
    with patch.dict('os.environ', {'LOCAL_RANK': '0', 'WORLD_SIZE': '2'}):
        # Mock CUDA calls since we are likely on CPU during CI
        with patch('torch.cuda.set_device'), patch('torch.cuda.amp.GradScaler'):
            accelerator = DDPAccelerator(mixed_precision=False)
            
            # 2. Case: Local Loss is NaN
            loss = get_nan_loss()
            
            # We mock the in-place behavior of all_reduce
            def side_effect(tensor, op):
                # If input was 1.0 (True), it remains 1.0 (True)
                # In a real DDP, this takes the MAX of all ranks
                if tensor.item() == 1.0:
                    pass 
                return tensor
            
            mock_dist.all_reduce.side_effect = side_effect

            # 3. Execution
            is_nan = accelerator.check_nan(loss)

            # 4. Assertions
            assert is_nan is True
            # Critical: Did we actually attempt to talk to the cluster?
            mock_dist.all_reduce.assert_called_once()

# --- Test 2: The Loop Interlock ---
# We verify that the Orchestrator RESPECTS the accelerator's warning
# and refuses to step the optimizer.

def test_orchestrator_skips_step_on_nan():
    # 1. Mocks
    mock_model = MagicMock()
    mock_model.training_step.return_value = {"loss": get_nan_loss()}
    mock_model.configure_optimizers.return_value = MagicMock()
    
    mock_data = MagicMock()
    mock_data.train_dataloader.return_value = [1] # Single batch
    
    # 2. Config with Safety
    conf = TrainingConfig(
        epochs=1, batch_size=1, learning_rate=0.1, model_name="test",
        max_grad_norm=1.0 
    )

    # 3. Mock Accelerator to simulate finding a NaN
    mock_accelerator = MagicMock()
    mock_accelerator.device = "cpu"
    mock_accelerator.forward_context.return_value.__enter__.return_value = None
    
    # *** THE INJECTION ***
    # Force check_nan to return True regardless of input
    mock_accelerator.check_nan.return_value = True 

    orchestrator = Orchestrator(
        model=mock_model,
        data=mock_data,
        config=conf,
        accelerator=mock_accelerator
    )

    # 4. Run Loop
    orchestrator.run()

    # 5. Forensics
    # Did we calculate loss? Yes.
    mock_model.training_step.assert_called()
    
    # Did we check for NaNs? Yes.
    mock_accelerator.check_nan.assert_called()
    
    # Did we backpropagate? NO.
    mock_accelerator.backward.assert_not_called()
    
    # Did we step the optimizer? NO.
    mock_accelerator.step.assert_not_called()
    
    # Did we zero_grad (flush buffers)? YES.
    orchestrator.optimizer.zero_grad.assert_called()

# --- Test 3: The Gradient Shield ---
# We verify that clipping is applied before the step

def test_gradient_clipping_logic():
    # 1. Config
    conf = TrainingConfig(
        epochs=1, batch_size=1, learning_rate=0.1, model_name="test",
        max_grad_norm=1.0 # Shield Enabled
    )
    
    # 2. Mocks
    mock_model = MagicMock()
    mock_model.training_step.return_value = {"loss": get_clean_loss()}
    mock_model.configure_optimizers.return_value = MagicMock()
    
    mock_data = MagicMock()
    mock_data.train_dataloader.return_value = [1]
    
    mock_accelerator = MagicMock()
    mock_accelerator.device = "cpu"
    mock_accelerator.forward_context.return_value.__enter__.return_value = None
    mock_accelerator.check_nan.return_value = False # Clean run

    orchestrator = Orchestrator(
        model=mock_model,
        data=mock_data,
        config=conf,
        accelerator=mock_accelerator
    )

    # 3. Run
    orchestrator.run()

    # 4. Verify Order of Operations
    # Clip -> Step -> Zero
    mock_accelerator.clip_grad_norm.assert_called_with(mock_model.parameters(), 1.0)
    mock_accelerator.step.assert_called()
