"""Training script for Chest X-Ray classification (Vertex AI ready)."""

import os
from pathlib import Path

import hydra
import torch
from google.cloud import storage
from main_project.data import ChestXRayDataset
from main_project.model import get_model
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

import wandb


def download_folder_from_gcs(bucket_name: str, gcs_prefix: str, local_dest: Path):
    """Download all files from a GCS folder prefix to a local folder."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_prefix)  # list all objects under the folder

    for blob in blobs:
        if blob.name.endswith("/"):
            continue  # skip folders
        # compute relative path inside the folder
        rel_path = Path(blob.name).relative_to(gcs_prefix)
        local_file_path = local_dest / rel_path
        local_file_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_file_path)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("-" * 80)

    wandb.login()
    # Initialize Weights & Biases
    wandb.init(
        project="chest-xray-classification",
        name=f"{cfg.model.name}_{cfg.training.lr}_{cfg.training.batch_size}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[cfg.model.name, f"bs{cfg.training.batch_size}"],
    )

    # Setup device
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # --- DOWNLOAD DATA FROM GCS ---
    gcs_bucket = "mlops-project-pneumonia-bucket"
    local_data_root = Path("data/processed")

    # Download train and val folders
    print("Downloading train and val folders from GCS...")
    download_folder_from_gcs(gcs_bucket, "processed/train", local_data_root / "train")
    download_folder_from_gcs(gcs_bucket, "processed/val",   local_data_root / "val")
    print("✓ Data downloaded")

    print("=== TRAIN TREE ===")
    os.system("find data/processed/train | head -n 20")

    print("=== VAL TREE ===")
    os.system("find data/processed/val | head -n 20")

    print("VAL NORMAL:", os.listdir("data/processed/val/NORMAL") if os.path.exists("data/processed/val/NORMAL") else "MISSING")
    print("VAL PNEUMONIA:", os.listdir("data/processed/val/PNEUMONIA") if os.path.exists("data/processed/val/PNEUMONIA") else "MISSING")


    # --- CREATE DATASETS AND LOADERS ---
    train_dataset = ChestXRayDataset(local_data_root, split="train")
    val_dataset   = ChestXRayDataset(local_data_root, split="val")

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.training.batch_size)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- MODEL ---
    print(f"\nCreating {cfg.model.name} model...")
    model = get_model(cfg.model.name, num_classes=cfg.model.num_classes, pretrained=cfg.model.pretrained)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    # Training loop
    print(f"\nTraining for {cfg.training.epochs} epochs...")
    print(f"Early stopping patience: {cfg.training.patience} epochs\n")
    best_val_acc = 0.0
    epochs_without_improvement = 0

    # Profiling the training process

    for epoch in range(cfg.training.epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)


        train_acc = 100 * train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)

        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss_avg,
            "train/accuracy": train_acc,
            "val/loss": val_loss_avg,
            "val/accuracy": val_acc,
        })

        print(f"Epoch {epoch+1}/{cfg.training.epochs} | "
                f"Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%")


        # Early stopping & save best model directly to GCS

        #replace with the correct saving technic
        Path("/gcs/mlops-project-pneumonia-bucket/outputs/models").mkdir(parents=True, exist_ok=True)
        model_path = Path("/gcs/mlops-project-pneumonia-bucket/outputs/models") / f"{cfg.model.name}_best.pth"

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            wandb.log({"best_val_accuracy": best_val_acc})
            wandb.save(str(model_path))
            print(f"  → Saved best model (val_acc: {val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
            print(f"  → No improvement ({epochs_without_improvement}/{cfg.training.patience})")
            if epochs_without_improvement >= cfg.training.patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                break

    print(f"\n✓ Training complete! Best val accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_path}")

    # Profiling results


    wandb.finish()


if __name__ == "__main__":
    train()
