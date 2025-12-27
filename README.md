# Aetheria v10.0

A SOLID, Scale-Agnostic Deep Learning Framework designed for High-Performance Computing.

## Features
- **DDP/AMP:** Zero-code scaling from Laptop to Cluster.
- **Resilience:** Synchronized NaN detection and Graceful Resume.
- **Modularity:** Plugin-based Models, Loggers, and Callbacks.

## Quick Start
1. Install: `pip install .`
2. Train: `aetheria train config.yaml`
3. Predict: `aetheria predict config.yaml checkpoints/model_ep0.pt`
