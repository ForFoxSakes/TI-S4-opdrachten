import torch

# Check if PyTorch is installed
print(f"PyTorch version: {torch.__version__}")

# Create two tensors and add them
a = torch.tensor([2.0, 3.0])
b = torch.tensor([4.0, 1.0])
c = a + b

print(f"Tensor a: {a}")
print(f"Tensor b: {b}")
print(f"Sum: {c}")

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")