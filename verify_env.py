import torch, transformers, timm, sklearn, cv2, einops

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print(
    "CUDA device:",
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
)
print("transformers:", transformers.__version__)
print("timm:", timm.__version__)
print("sklearn:", sklearn.__version__)
print("cv2:", cv2.__version__)
print("einops:", einops.__version__)
