import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Test PyTorch installation
print(f"PyTorch version: {torch.__version__}")
print(f"TOrchvision version: {torchvision.__version__}")
print(f"MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}")

# Create a simple tensor
x = torch.randn(3, 3)
print(f"Sample tensor:\n{x}")