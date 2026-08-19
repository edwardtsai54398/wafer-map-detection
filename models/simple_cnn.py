import torch.nn as nn


def conv_block(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    """Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d(2)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        
        nn.MaxPool2d(kernel_size=2),
    )


class SimpleCNN(nn.Module):
    """Basic 4-block CNN: 1(in) -> 16 -> 32 -> 64 -> 128(out)."""

    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.35),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)
