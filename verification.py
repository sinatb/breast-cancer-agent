## Run this code to test if you have the requirements for running this project

import torch; print(f'PyTorch: {torch.__version__}')
import torchvision; print(f'Torchvision: {torchvision.__version__}')
import numpy; print(f'NumPy: {numpy.__version__}')

import torch; print(f'CUDA available: {torch.cuda.is_available()}')