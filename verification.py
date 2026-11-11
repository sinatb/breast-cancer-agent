import cv2
import numpy as np
import pandas as pd
import torch

print("Python OK")
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Pandas available:", pd.__version__)
