## Model Checkpoints (Sample)

This folder contains code to load and use the trained model checkpoints.

### 📂 Files
- `OceanLens.py`: Main model architecture and training utilities
- `Oceanlens_inf.py`: Script to run inference using saved model checkpoints
- `Checkpoints/`: Folder containing saved model weights

  
These files are **only meant for running our provided checkpoints**

### How to Run

To run the model using checkpoints:

1. Make sure all dependencies are installed (see main `README.md`).
2. Use `Oceanlens_inf.py` for inference. It loads the models defined in `OceanLens.py` and the weights from `Checkpoints/`.

```bash
python Oceanlens_inf.py
