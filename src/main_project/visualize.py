"""Visualization script for comparing Chest X-Ray classification models."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Models to compare
MODELS = ["baseline", "alexnet", "vgg16"]

# Reference benchmarks from literature (Chest X-Ray Pneumonia detection)
MODERN_BENCHMARKS = {
    "ResNet-50": {"accuracy": 94.0, "f1": 0.94},
    "DenseNet-121": {"accuracy": 95.5, "f1": 0.95},
    "EfficientNet-B0": {"accuracy": 96.0, "f1": 0.96},
    "ViT-Base": {"accuracy": 97.0, "f1": 0.97},
}


def load_results() -> dict:
    """Load evaluation results from reports/ directory."""
    reports_dir = Path("reports")
    results = {}

    for model in MODELS:
        path = reports_dir / f"{model}_eval.json"
        if path.exists():
            with open(path) as f:
                results[model] = json.load(f)
        else:
            print(f"⚠ Missing: {path}")

    return results


def plot_metrics_comparison(results: dict, output_dir: Path) -> None:
    """Create bar chart comparing all metrics across models."""
    if not results:
        print("No results to plot.")
        return

    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = [results[m][metric] if metric != "accuracy" else results[m][metric] / 100 for m in models]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize())
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Classification Metrics")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=150)
    print(f"✓ Saved: {output_dir / 'metrics_comparison.png'}")
    plt.close()


def plot_accuracy_vs_params(results: dict, output_dir: Path) -> None:
    """Create scatter plot of accuracy vs model size."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Our models
    for model, data in results.items():
        ax.scatter(data["params"] / 1e6, data["accuracy"], s=150, label=f"{model} (ours)", zorder=3)
        ax.annotate(model, (data["params"] / 1e6, data["accuracy"]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)

    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Efficiency: Accuracy vs. Size")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_params.png", dpi=150)
    print(f"✓ Saved: {output_dir / 'accuracy_vs_params.png'}")
    plt.close()


def plot_benchmark_comparison(results: dict, output_dir: Path) -> None:
    """Compare our models against modern architecture benchmarks."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Combine our results with benchmarks
    all_models = []
    accuracies = []
    colors = []

    # Our models (blue)
    for model, data in results.items():
        all_models.append(f"{model}\n(ours)")
        accuracies.append(data["accuracy"])
        colors.append("#3498db")

    # Modern benchmarks (gray)
    for model, data in MODERN_BENCHMARKS.items():
        all_models.append(model)
        accuracies.append(data["accuracy"])
        colors.append("#95a5a6")

    x = np.arange(len(all_models))
    bars = ax.bar(x, accuracies, color=colors)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Model Architecture")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Performance Comparison with Modern Architectures")
    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=45, ha="right")
    ax.axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90% threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_comparison.png", dpi=150)
    print(f"✓ Saved: {output_dir / 'benchmark_comparison.png'}")
    plt.close()


def print_summary_table(results: dict) -> None:
    """Print a summary comparison table."""
    print("\n" + "=" * 75)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 75)
    print(f"{'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Params':>12}")
    print("-" * 75)

    for model, data in results.items():
        print(f"{model:<12} {data['accuracy']:>9.2f}% {data['precision']:>10.4f} "
              f"{data['recall']:>10.4f} {data['f1']:>10.4f} {data['params']:>12,}")

    print("-" * 75)
    print("\nMODERN ARCHITECTURE BENCHMARKS (Literature)")
    print("-" * 75)
    for model, data in MODERN_BENCHMARKS.items():
        print(f"{model:<25} {data['accuracy']:>9.1f}%  F1: {data['f1']:.2f}")

    # Analysis
    if results:
        best_model = max(results, key=lambda x: results[x]["f1"])
        best_f1 = results[best_model]["f1"]
        modern_avg = sum(m["f1"] for m in MODERN_BENCHMARKS.values()) / len(MODERN_BENCHMARKS)
        gap = modern_avg - best_f1

        print("\n" + "=" * 75)
        print("ANALYSIS")
        print("=" * 75)
        print(f"• Best model: {best_model} (F1: {best_f1:.4f})")
        print(f"• Gap to modern architectures (avg): {gap:.2f}")

        if gap > 0.05:
            print("• Recommendation: Consider ResNet/DenseNet/EfficientNet for production")
        else:
            print("• Performance is competitive with modern architectures!")


def main() -> None:
    """Generate all visualizations and comparison report."""
    print("=" * 75)
    print("CHEST X-RAY CLASSIFICATION - MODEL COMPARISON VISUALIZATION")
    print("=" * 75)

    # Load results
    results = load_results()

    if not results:
        print("\n⚠ No evaluation results found in reports/")
        print("Run evaluate.py for each model first:")
        print("  uv run src/main_project/evaluate.py model=baseline")
        print("  uv run src/main_project/evaluate.py model=alexnet")
        print("  uv run src/main_project/evaluate.py model=vgg16")
        return

    print(f"\nFound results for: {', '.join(results.keys())}")

    # Create output directory
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_metrics_comparison(results, output_dir)
    plot_accuracy_vs_params(results, output_dir)
    plot_benchmark_comparison(results, output_dir)

    # Print summary
    print_summary_table(results)

    print(f"\n✓ All figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()
