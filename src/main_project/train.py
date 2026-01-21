"""Training script for Chest X-Ray classification."""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

import wandb
from main_project.data import ChestXRayDataset
from main_project.model import get_model


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """
    Train a model on Chest X-Ray dataset using Hydra configuration.

    Args:
        cfg: Hydra configuration object containing all hyperparameters
    """
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("-" * 80)

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

    # Load datasets
    print("Loading datasets...")
    train_dataset = ChestXRayDataset(Path(cfg.data.root), split="train")
    val_dataset = ChestXRayDataset(Path(cfg.data.root), split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
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

                prof.step() # Step the profiler for each batch

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

            # Log metrics to W&B
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss_avg,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss_avg,
                    "val/accuracy": val_acc,
                }
            )

            print(
                f"Epoch {epoch+1}/{cfg.training.epochs} | "
                f"Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%"
            )

            # Save best model and check early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                Path(cfg.output.models_dir).mkdir(exist_ok=True)
                model_path = f"{cfg.output.models_dir}/{cfg.model.name}_best.pth"
                torch.save(model.state_dict(), model_path)

                # Log best model to W&B
                wandb.log({"best_val_accuracy": best_val_acc})
                wandb.save(model_path)

                print(f"  → Saved best model (val_acc: {val_acc:.2f}%)")
            else:
                epochs_without_improvement += 1
                print(f"  → No improvement ({epochs_without_improvement}/{cfg.training.patience})")

                if epochs_without_improvement >= cfg.training.patience:
                    print(
                        f"\n⚠ Early stopping triggered after {epoch+1} epochs (no improvement for {cfg.training.patience} epochs)"
                    )
                    break

    print(f"\n✓ Training complete! Best val accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {cfg.output.models_dir}/{cfg.model.name}_best.pth")

    # Add profiling results
    print("\n== Profiling Results ==")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Export trace for Chrome visualization
    trace_path = f"{cfg.output.models_dir}/trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved to: {trace_path}")
    print("Open in Chrome: ui.perfetto.dev → Load → select trace.json")

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    train()
