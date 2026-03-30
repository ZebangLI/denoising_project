from datasets.synthetic_dataset import SyntheticDenoiseDataset
from torch.utils.data import DataLoader

dataset = SyntheticDenoiseDataset(
    image_dir="data/Berkeley Segmentation Dataset 500 (BSDS500)/images/train",
    sigma=25,
    patch_size=128
)

train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

print("Dataset size:", len(dataset))
print("Number of batches:", len(train_loader))

for noisy_imgs, clean_imgs in train_loader:
    print("Noisy batch shape:", noisy_imgs.shape)
    print("Clean batch shape:", clean_imgs.shape)
    break