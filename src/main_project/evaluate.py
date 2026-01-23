"""Evaluation script for Chest X-Ray classification."""

import json
from pathlib import Path

import hydra
import torch
from main_project.data import ChestXRayDataset
from main_project.model import get_model
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """
    Evaluate a trained model on the test set and save results to JSON.

    Args:
        cfg: Hydra configuration object containing all hyperparameters
    """
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("-" * 80)

    # Setup device
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = ChestXRayDataset(Path(cfg.data.root), split="test")
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.data.num_workers
    )
    print(f"Test samples: {len(test_dataset)}")

    # Load model
    model_name = cfg.model.name
    print(f"\nLoading {model_name} model...")
    model = get_model(model_name, num_classes=cfg.model.num_classes, pretrained=False)

    model_path = Path(cfg.output.models_dir) / f"{model_name}_best.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded model from: {model_path}")

    # Evaluate
    print("\nEvaluating...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Calculate metrics
    num_params = sum(p.numel() for p in model.parameters())
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    # Print results
    print(f"\n{'='*50}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Parameters: {num_params:,}")

    # Classification report
    class_names = ["NORMAL", "PNEUMONIA"]
    print(f"\n{'='*50}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Save results to JSON
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    results = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "params": num_params,
        "test_samples": len(test_dataset),
    }

    output_path = reports_dir / f"{model_name}_eval.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    evaluate()
