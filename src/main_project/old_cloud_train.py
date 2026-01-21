"""Training script for Chest X-Ray classification (Vertex AI ready)."""

import os
from pathlib import Path

import hydra
import torch
from google.cloud import storage
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, TensorDataset

import wandb
from main_project.model import get_model


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)





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

    # LOAD DATA DIRECTLY FROM GCS
    gcs_bucket = "mlops-project-pneumonia-bucket"

    local_train_path = "data/processed/train.pt"
    local_val_path   = "data/processed/val.pt"

    download_from_gcs(gcs_bucket, "processed/train.pt", local_train_path)
    download_from_gcs(gcs_bucket, "processed/val.pt",   local_val_path)

    train_data = torch.load(local_train_path)
    val_data   = torch.load(local_val_path)


    # Wrap in DataLoader
    train_loader = DataLoader(TensorDataset(*zip(*train_data)),
                              batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*zip(*val_data)),
                            batch_size=cfg.training.batch_size)






    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

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
    with profile(activities=[ProfilerActivity.CPU],
                 record_shapes=True) as prof:

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
                prof.step()

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
    print("\n== Profiling Results ==")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace(Path("/gcs/mlops-project-pneumonia-bucket/outputs/models") / "trace.json")
    print("\nChrome trace saved to /gcs/mlops-project-pneumonia-bucket/outputs/models/trace.json")
    wandb.finish()


if __name__ == "__main__":
    train()
