from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

pair_dir = r"data/sidd/Data/0001_001_S6_00100_00060_3200_L"

gt_path = os.path.join(pair_dir, "GT_SRGB_010.PNG")
noisy_path = os.path.join(pair_dir, "NOISY_SRGB_010.PNG")

to_tensor = transforms.ToTensor()

gt_img = Image.open(gt_path).convert("RGB")
noisy_img = Image.open(noisy_path).convert("RGB")

gt_tensor = to_tensor(gt_img)
noisy_tensor = to_tensor(noisy_img)

print("GT shape:", gt_tensor.shape)
print("Noisy shape:", noisy_tensor.shape)
print("GT min/max:", gt_tensor.min().item(), gt_tensor.max().item())
print("Noisy min/max:", noisy_tensor.min().item(), noisy_tensor.max().item())

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(gt_tensor.permute(1, 2, 0))
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(noisy_tensor.permute(1, 2, 0))
plt.title("Noisy")
plt.axis("off")

plt.show()