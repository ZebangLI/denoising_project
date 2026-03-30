from datasets.sidd_dataset import SIDDDataset
from torch.utils.data import DataLoader

dataset = SIDDDataset(
    root_dir="data/sidd/Data",
    patch_size=128
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

print("Dataset size:", len(dataset))
print("Number of batches:", len(loader))

for noisy_imgs, clean_imgs in loader:
    print("Noisy batch shape:", noisy_imgs.shape)
    print("Clean batch shape:", clean_imgs.shape)
    break