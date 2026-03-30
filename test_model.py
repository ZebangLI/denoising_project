import torch
from models.simple_cnn import SimpleDenoiser

model = SimpleDenoiser()

x = torch.randn(4, 3, 128, 128)
y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)