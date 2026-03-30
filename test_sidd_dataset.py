from datasets.sidd_dataset import SIDDDataset
import matplotlib.pyplot as plt

dataset = SIDDDataset(
    root_dir="data/sidd/Data",
    patch_size=128
)

print("Dataset size:", len(dataset))

noisy_img, clean_img = dataset[0]

print("Clean patch shape:", clean_img.shape)
print("Noisy patch shape:", noisy_img.shape)
print("Clean min/max:", clean_img.min().item(), clean_img.max().item())
print("Noisy min/max:", noisy_img.min().item(), noisy_img.max().item())

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(clean_img.permute(1, 2, 0))
plt.title("Ground Truth Patch")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(noisy_img.permute(1, 2, 0))
plt.title("Noisy Patch")
plt.axis("off")

plt.show()