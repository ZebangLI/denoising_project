from datasets.synthetic_dataset import SyntheticDenoiseDataset
import matplotlib.pyplot as plt

dataset = SyntheticDenoiseDataset(
    image_dir="data/Berkeley Segmentation Dataset 500 (BSDS500)/images/train",
    sigma=25,
    patch_size=128
)

print("Number of training images:", len(dataset))

noisy_img, clean_img = dataset[0]

print("Clean patch shape:", clean_img.shape)
print("Noisy patch shape:", noisy_img.shape)
print("Clean min value:", clean_img.min().item())
print("Clean max value:", clean_img.max().item())
print("Noisy min value:", noisy_img.min().item())
print("Noisy max value:", noisy_img.max().item())

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(clean_img.permute(1, 2, 0))
plt.title("Clean Patch")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(noisy_img.permute(1, 2, 0))
plt.title("Noisy Patch")
plt.axis("off")

plt.show()