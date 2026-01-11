"""Training script for Chest X-Ray classification."""

from pathlib import Path

import torch
import typer
from torch import nn
from torch.utils.data import DataLoader

from main_project.data import ChestXRayDataset
from main_project.model import get_model


def train(
    model_name: str = "baseline",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = "auto",
    pretrained: bool = True
) -> None:
    """
    Train a model on Chest X-Ray dataset.
    
    Args:
        model_name: Model to use ('baseline', 'alexnet', 'vgg16')
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        pretrained: Use pretrained weights (only for alexnet/vgg16)
    """
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ChestXRayDataset(Path("data/processed"), split="train")
    val_dataset = ChestXRayDataset(Path("data/processed"), split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    use_pretrained = pretrained and (model_name != "baseline")
    print(f"\nCreating {model_name} model{'(pretrained)' if use_pretrained else ''}...")
    model = get_model(model_name, num_classes=2, pretrained=use_pretrained)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_val_acc = 0.0
    
    for epoch in range(epochs):
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
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")
            print(f"  → Saved best model (val_acc: {val_acc:.2f}%)")
    
    print(f"\n✓ Training complete! Best val accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: models/{model_name}_best.pth")


if __name__ == "__main__":
    typer.run(train)
