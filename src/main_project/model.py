from torch import nn
import torch

class VGG16(nn.Module):
    def __init__(self, in_channels, num_classes, input_height, input_width):
        super().__init__()
        self.num_classes = num_classes
        
        # Calculate padding needed
        self.pad_height = (32 - (input_height % 32)) % 32
        self.pad_width = (32 - (input_width % 32)) % 32
        
        print(f"Input: {input_height}x{input_width}")
        print(f"Padding: {self.pad_height} pixels height, {self.pad_width} pixels width")
        print(f"Padded to: {input_height + self.pad_height}x{input_width + self.pad_width}")
        
        # Padding layer
        self.pad = nn.ZeroPad2d((0, self.pad_width, 0, self.pad_height))
        
        # Conv layers (same as before, with ReLU included)
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding="same", stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.flatten = nn.Flatten()
        self.classifier = None
    
    def calculate_features(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            x = self.pad(x)
            features = self.conv_layers(x)
            return features.numel() // x.shape[0]
    
    def build_classifier(self, x: torch.Tensor) -> None:
        features = self.calculate_features(x)
        print(f"Classifier input features: {features}")
        
        self.classifier = nn.Sequential(
            nn.Linear(features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad input
        x = self.pad(x)
        
        # Build classifier if needed
        if self.classifier is None:
            self.build_classifier(x)
        
        # Forward pass
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x


# Test with your dimensions
if __name__ == "__main__":
    model = VGG16(in_channels=1, num_classes=3, 
                             input_height=1240, input_width=840)
    
    # Correct input shape: [batch, channels, height, width]
    x = torch.rand(1, 1, 1240, 840)
    print(f"\nInput shape: {x.shape}")
    
    output = model(x)
    print(f"Output shape: {output.shape}") 
