import torch
print("Version:", torch.__version__)
print("Is CUDA available?", torch.cuda.is_available())
print("Compiled with CUDA:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")