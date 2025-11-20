import torch
import torch.nn as nn
import torch.nn.functional as F  # may be useful if you extend things later


def count_parameters(model: nn.Module) -> int:
    """Utility for debugging / reporting, not used in marking."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _CNNSmall(nn.Module):
    """
    Very compact CNN (~26k parameters).
    Good for parameter-efficiency experiments.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(1, 16),          # 28x28 -> 28x28
            nn.MaxPool2d(2, 2),         # 14x14
            _ConvBlock(16, 32),         # 14x14
            nn.MaxPool2d(2, 2),         # 7x7
            _ConvBlock(32, 64),         # 7x7
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class _CNNMedium(nn.Module):
    """
    Medium CNN (~59k parameters).
    This is the one I’d use as your main submission model.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            _ConvBlock(1, 24),          # 28x28
            nn.MaxPool2d(2, 2),         # 14x14
            _ConvBlock(24, 48),         # 14x14
            nn.MaxPool2d(2, 2),         # 7x7
            _ConvBlock(48, 96),         # 7x7
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    """
    Main model class required by the coursework.

    Kept backwards-compatible with the original skeleton:
      - Keeps input_size, hidden_size, num_classes in the signature
      - ALSO accepts `variant`:
            "medium" (default) or "small"
    """
    def __init__(
        self,
        input_size: int = 28 * 28,   # kept for compatibility, not used
        hidden_size: int = 128,      # kept for compatibility, not used
        num_classes: int = 10,
        variant: str = "medium",
        **kwargs,
    ):
        super().__init__()

        if variant == "small":
            self.model = _CNNSmall(num_classes=num_classes)
        elif variant == "medium":
            self.model = _CNNMedium(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown variant '{variant}'. Use 'small' or 'medium'.")

        # Optional: initialise conv/linear weights with Kaiming normal
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either [B, 1, 28, 28] or [B, 28, 28]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.model(x)
    