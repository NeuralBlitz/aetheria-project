import torch
import os
from unittest.mock import MagicMock, patch
from aetheria.accelerator import DDPAccelerator
from aetheria.orchestrator import Orchestrator
from aetheria.config import TrainingConfig

@patch('aetheria.accelerator.dist')
@patch('aetheria.accelerator.torch.cuda')
def test_ddp_nan_synchronization(mock_cuda, mock_dist):
    """
    Verifies that if one GPU sees NaN, all_reduce is called to notify others.
    Mocks are used to simulate a GPU environment on a CPU-only CI runner.
    """
    # 1. Simulate Distributed Environment Variables
    with patch.dict('os.environ', {'LOCAL_RANK': '0', 'WORLD_SIZE': '2'}):
        
        # 2. Force the Accelerator to use CPU for the test to avoid runtime errors
        # preventing it from trying to allocate tensors on non-existent 'cuda:0'
        with patch('aetheria.accelerator.torch.device', return_value='cpu'):
            
            # Initialize DDP Accelerator (mocks prevent actual NCCL init)
            acc = DDPAccelerator(mixed_precision=False)
            
            # Create a NaN loss
            loss = torch.tensor(float('nan'))
            
            # 3. Execution
            # check_nan performs an all_reduce to check peers
            is_nan = acc.check_nan(loss)
            
            # 4. Assertions
            # It should return True (NaN detected)
            assert is_nan is True
            # Crucial: It must have communicated with the cluster
            mock_dist.all_reduce.assert_called_once()

def test_orchestrator_skips_step_on_nan():
    """
    Verifies that the Orchestrator halts optimization when a NaN is detected.
    """
    # 1. Setup Model Mocks
    mock_model = MagicMock()
    mock_model.training_step.return_value = {"loss": torch.tensor(float('nan'))}
    # Mocking configure_optimizers result
    mock_model.configure_optimizers.return_value = MagicMock()
    
    # 2. Setup Accelerator Mocks
    mock_acc = MagicMock()
    mock_acc.check_nan.return_value = True  # Simulate finding a NaN
    # Mock the context manager for forward pass
    mock_acc.forward_context.return_value.__enter__.return_value = None
    
    # *** FIX: Return tuple (model, optimizer) to satisfy unpacking in Orchestrator ***
    mock_acc.setup.return_value = (mock_model, MagicMock())

    # 3. Setup Config & Data
    conf = TrainingConfig(
        epochs=1, 
        batch_size=1, 
        learning_rate=0.1, 
        model_name="test",
        # Disable gradient accumulation so we step immediately
        grad_accumulation_steps=1 
    )
    
    # Mock the Data Loader to return one batch
    mock_data = MagicMock()
    mock_data.train_dataloader.return_value = [torch.randn(1, 10)] 

    # 4. Initialize Orchestrator
    orch = Orchestrator(mock_model, mock_data, conf, accelerator=mock_acc)
    
    # 5. Run
    orch.run()

    # 6. Assertions
    # Ensure backward was NOT called because of NaN
    mock_acc.backward.assert_not_called()
    # Ensure optimizer step was NOT called
    mock_acc.step.assert_not_called()
    # Ensure gradients were flushed (zeroed) to prevent pollution
    orch.optimizer.zero_grad.assert_called()
