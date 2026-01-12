"""Model architectures for Chest X-Ray classification."""

import torch
from torch import nn
from torchvision import models


class BaselineCNN(nn.Module):
    """Simple CNN baseline for chest X-ray classification."""

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_name: str = "baseline", num_classes: int = 2, pretrained: bool = True):
    """
    Get a model for chest X-ray classification.

    Args:
        model_name: 'baseline', 'alexnet', or 'vgg16'
        num_classes: Number of output classes (default: 2)
        pretrained: Use pretrained weights for transfer learning

    Returns:
        PyTorch model
    """
    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)

    elif model_name == "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model

    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'baseline', 'alexnet', or 'vgg16'")


if __name__ == "__main__":
    # Test models
    print("Testing model architectures...\n")

    models_to_test = ["baseline", "alexnet", "vgg16"]
    dummy_input = torch.randn(1, 3, 224, 224)

    for model_name in models_to_test:
        model = get_model(model_name, pretrained=False)
        output = model(dummy_input)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"{model_name:10s} | Params: {num_params:>10,} | Output: {output.shape}")
