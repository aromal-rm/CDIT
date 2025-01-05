import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current Device ID: {torch.cuda.current_device()}")
else:
    print("GPU is not available. Using CPU.")