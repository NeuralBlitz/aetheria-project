# The Aetheria Architecture Manifesto

> "Complexity is the enemy of reliability."

Aetheria is not just a deep learning framework; it is a rejection of spaghetti code in Machine Learning. It is built on a strict adherence to SOLID principles, ensuring that Research Logic (Science) never bleeds into Infrastructure Logic (Engineering).

## The 4 Immutable Laws of Aetheria

### I. The Law of Inversion (Dependency Inversion)
**The Orchestrator shall never know the implementation details of the Hardware.**

*   **Violation:** `if torch.cuda.is_available(): ...` inside the training loop.
*   **Aetheria Way:** The Orchestrator asks the `Accelerator` to `process_batch()`. Whether that happens on a CPU, a GPU, or a TPU Pod is irrelevant to the loop.

### II. The Law of Sovereignty (Encapsulation)
**The Model shall be the sole authority on its own Optimization.**

*   **Violation:** Passing `learning_rate` to the Orchestrator.
*   **Aetheria Way:** The Model defines `configure_optimizers()`. It owns its hyperparameters, schedulers, and weight decay strategies. The Orchestrator merely executes the strategy.

### III. The Law of Separation (Single Responsibility)
**The Training Loop shall not concern itself with Observability.**

*   **Violation:** Putting `wandb.log()` or `print()` statements inside `orchestrator.py`.
*   **Aetheria Way:** The Orchestrator emits signals (`on_batch_end`). Plugins (`MetricLogger`, `WandbLogger`) listen for signals. You can strip out all logging without breaking the training physics.

### IV. The Law of Resilience (Fault Tolerance)
**Time must be serializable.**

*   **Violation:** Saving only `model.state_dict()`.
*   **Aetheria Way:** We save the Universe. Epoch, Global Step, Optimizer State, and Random Number Generator (RNG) states are snapshotted. A crashed job must resume *deterministically*, or it is not valid science.

---

## The System Stack

| Layer | Component | Responsibility | Design Pattern |
| :--- | :--- | :--- | :--- |
| **User** | `CLI / Config` | Intent & Hyperparameters | Command |
| **App** | `Plugins` | Models, Datasets, Loggers | Factory / Adapter |
| **Engine** | `Orchestrator` | State Machine (Train/Val) | Mediator / Template |
| **Safety** | `Resilience` | NaN Checks, Grad Clipping | Chain of Responsibility |
| **Metal** | `Accelerator` | DDP, Mixed Precision, I/O | Strategy |

---

## Deployment Strategy

*   **Packaging:** `pyproject.toml` standardizes the build.
*   **Runtime:** Docker images pinned to specific CUDA versions ensure mathematical reproducibility.
*   **Scale:** `torchrun` manages the distributed process group; Aetheria manages the logic.

*Built with precision for the modern AI Architect.*
