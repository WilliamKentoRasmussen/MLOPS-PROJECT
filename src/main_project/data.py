from pathlib import Path

from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class PneumoniaDataset(Dataset):
    """My custom dataset."""

    def __init__(self, root_dir: Path, transform = None) -> None:

        self.root_dir = root_dir #root_dir - root directory of either val and train, test
        self.image_paths = []
        self.labels = []
        self.transform = transform
        #Normal: 0, Bacterial: 1, Viral: 2
        self.classes = ['NORMAL', 'BACTERIAL', 'VIRAL'] #Phenumonia can be classified either as bacteiral or viral


        for label_dir in ['NORMAL', 'PNEUMONIA']:
            class_path = os.path.join(self.root_dir, label_dir)
            if not os.path.exists(class_path): continue

            #Goes through all the images of Normal and Phenumonia images
            for img_name in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, img_name))
                if label_dir == 'NORMAL':
                    self.labels.append(0)
                elif 'bacteria' in img_name.lower():
                    self.labels.append(1)
                else:
                    self.labels.append(2)


    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int):
        """Return a tuple consisting of a image and a label from the dataset."""
        img = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[index]
        #return self.image_paths[index], self.image_paths[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        images, targets = [], []

        for index in range(len(self.image_paths)):
            img = Image.open(self.image_paths[index]).convert('RGB')
            if self.transform: img = self.transform(img)

            images.append(torch.load(self.image_paths[index]))
            targets.append(torch.load(self.labels[index]))

        images = torch.cat(images)
        targets = torch.cat(targets)

        images = torch.cat(images).unsqueeze(1).float()
        targets = torch.cat(targets).unsqueeze(1).float()

        #Writes cleaned tensors to the processed_dir.
        torch.save(images, f"{output_folder}/train_images.pt")
        torch.save(targets, f"{output_folder}/train_target.pt")



# Image Transformations
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])






def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")

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
