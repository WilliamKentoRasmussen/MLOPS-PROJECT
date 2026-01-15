"""Data loading and preprocessing for Chest X-Ray Pneumonia dataset."""

import shutil
from pathlib import Path


import kaggle
import torch
import typer
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXRayDataset(Dataset):
    """Chest X-Ray Pneumonia dataset."""

    def __init__(self, data_path: Path, split: str = "train") -> None:
        self.data_path = data_path / split

        # Collect image paths and labels
        self.samples = []
        self.labels = []

        # NORMAL = 0, PNEUMONIA = 1
        for label, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
            class_dir = self.data_path / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob("*.jpeg")):
                    self.samples.append(img_path)
                    self.labels.append(label)

        # Transforms for pretrained models (AlexNet, VGG16, etc.)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return a given sample from the dataset."""
        image = Image.open(self.samples[index]).convert("RGB")
        return self.transform(image), self.labels[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        raw_path = output_folder.parent / "raw"
        chest_xray_path = raw_path / "chest_xray"

        # Download if needed
        if not chest_xray_path.exists():
            print("Downloading from Kaggle (requires ~/.kaggle/kaggle.json)...")
            raw_path.mkdir(parents=True, exist_ok=True)
            kaggle.api.dataset_download_files("paultimothymooney/chest-xray-pneumonia", path=raw_path, unzip=True)

        # Copy to processed folder
        output_folder.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            src, dst = chest_xray_path / split, output_folder / split
            if src.exists() and not dst.exists():
                shutil.copytree(src, dst)

        print("✓ Data ready!")



def preprocess(data_path: Path, output_folder: Path) -> None:
    """Preprocess data."""
    print("Preprocessing data...")

    dataset = ChestXRayDataset(data_path)
    dataset.preprocess(output_folder)


    datasets = get_pneumonia_datasets()
    for dataset in datasets:
        dataset.preprocess(output_folder)


def get_pneumonia_datasets():
    """
    Returns a tuple of the pneumonia dataset in this order train_ds, test_ds, val_ds
    """""
    train_ds = PneumoniaDataset('data/chest_xray/train', transform=transform)
    test_ds = PneumoniaDataset('data/chest_xray/test', transform=transform)
    val_ds = PneumoniaDataset('data/chest_xray/val', transform=transform)

    return train_ds, test_ds, val_ds

if __name__ == "__main__":
    #typer.run(preprocess)

    train_ds = PneumoniaDataset('data/chest_xray/train', transform=transform)
    test_ds = PneumoniaDataset('data/chest_xray/test', transform=transform)
    val_ds = PneumoniaDataset('data/chest_xray/val', transform=transform)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True) ##Only gives one image if batch_size = 1
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    print(f"Loaded {len(train_ds)} training images, {len(val_ds)} validation images, and {len(test_ds)} testing images.")
    for images, labels in train_loader:
        print("Batch of images shape:", images.shape)  # Should be [batch_size, 3, 300, 300]
        print("Batch of labels shape:", labels.shape)  # Should be [batch_size]
        break
