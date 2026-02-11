import torch

print("torch version:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)   # <- new
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
