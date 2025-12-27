
---

# 🌌 Aetheria v10.0

**A SOLID, Scale-Agnostic, and Resilient Deep Learning Framework.**

Aetheria is a modular AI framework designed for high-performance computing (HPC) and large-scale model training. It enforces a strict separation between **Research Logic** (the "Science") and **Infrastructure Logic** (the "Engineering"). 

By adhering to the principles of Dependency Inversion and the Strategy pattern, Aetheria allows you to scale from a single CPU to a distributed cluster of 1,000+ GPUs with zero changes to your model code.

---

## 📜 The Aetheria Manifesto

Complexity is the enemy of stability. Aetheria is built upon four immutable laws:

1.  **The Law of Inversion:** The training loop (Orchestrator) never knows the implementation details of the hardware.
2.  **The Law of Sovereignty:** The Model is the sole authority on its own optimization and hyperparameters.
3.  **The Law of Separation:** The training loop does not concern itself with observability; it emits signals, and plugins listen.
4.  **The Law of Resilience:** Time is serializable. Every state—from weights to random number seeds—must be snapshot-ready.

---

## 🚀 Key Features

*   **Zero-Code Scaling:** Switch between `CPU`, `GPU`, and `Multi-GPU DDP` via configuration.
*   **HPC Fortified:** Out-of-the-box support for **Automatic Mixed Precision (AMP)** and **Gradient Accumulation**.
*   **Numerical Safety:** Synchronized **NaN detection** across distributed ranks to prevent model corruption.
*   **Fault Tolerance:** Full-state serialization allows resuming training exactly where it left off, including optimizer momentum and RNG states.
*   **Plugin Architecture:** Decoupled Model, Data, Logger, and Callback registries.

---

## 🏛 Architecture

| Layer | Responsibility | Pattern |
| :--- | :--- | :--- |
| **User/CLI** | Hyperparameter definition and job execution. | Command |
| **Logic** | Neural Network architecture and loss calculation. | Template Method |
| **Engine** | State machine governing the training/validation loops. | Mediator |
| **Hardware** | Device placement, DDP syncing, and precision. | Strategy |
| **Observability**| Metric reporting and cloud integration (WandB). | Observer |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/aetheria-project.git
cd aetheria_project

# Install in editable mode with development dependencies
pip install -e .[dev]
```

---

## ⚡ Quick Start

### 1. Define your experiment (`config.yaml`)
```yaml
model_name: "SimpleMLP"
epochs: 10
batch_size: 64
learning_rate: 0.001

mixed_precision: true
grad_accumulation_steps: 4
max_grad_norm: 1.0  # Gradient Clipping
```

### 2. Launch Training

**Local Development:**
```bash
aetheria train config.yaml
```

**Massive Scale (8 GPUs via DDP):**
```bash
torchrun --nproc_per_node=8 cli.py train config.yaml
```

**Resume from Failure:**
```bash
aetheria train config.yaml --resume checkpoints/snapshot_ep5.pt
```

### 3. High-Performance Inference
```python
from aetheria.inference import Predictor

engine = Predictor(
    model_name="SimpleMLP",
    model_params={"input_dim": 10, "hidden_dim": 128},
    checkpoint_path="checkpoints/snapshot_ep10.pt"
)

# Optimized, thread-safe prediction
result = engine.predict(batch_data)
```

---

## 🧪 Verification

To ensure the resilience of the NaN guards and distributed logic, run the test suite:

```bash
pytest tests/ -v
```

---

## 🤝 Contributing

Aetheria is open for extension but closed for modification. 
*   Add new models to `plugins/models.py`.
*   Add new logging backends (e.g., MLFlow, Neptune) to `plugins/loggers.py`.
*   Maintain the core `aetheria/` library as a stable, verified engine.

---

