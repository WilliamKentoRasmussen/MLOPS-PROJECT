"""Data loading and preprocessing for Chest X-Ray Pneumonia dataset."""

import shutil
from pathlib import Path

import kagglehub
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

        # Download (cached automatically)
        print("Downloading (or reusing cache) from KaggleHub...")
        dataset_path = Path(
            kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        )

        chest_xray_path = dataset_path / "chest_xray"

        if not chest_xray_path.exists():
            raise FileNotFoundError("Expected 'chest_xray' folder not found in dataset.")

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



if __name__ == "__main__":
    typer.run(preprocess)
    ## Run it using uv run src/main_project/data.py data/raw data/processed
