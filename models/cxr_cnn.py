"""
Diseño de la Red Neuronal Convolucional para clasificación de Rayos X de tórax.

[Notebook] Celda: Diseño de Red Neuronal — Explicación y Arquitectura

Decisiones de diseño:
- Entradas en escala de grises (in_channels=1). Las CXR suelen ser monocromáticas; evitamos replicar canales.
- Filtros 3x3 con padding=1 en cada conv: capturan patrones locales (bordes, texturas) manteniendo resolución.
- Bloques Conv-BN-ReLU en pares: estabilidad (BatchNorm) y no linealidad (ReLU) para mayor capacidad.
- Pooling 2x2 tras cada bloque: reduce dimensión espacial y agrega invariancia local.
- Profundidad progresiva (32→64→128→256 canales): permite aprender desde bordes a patrones más complejos.
- AdaptiveAvgPool2d(1): agregación global espacial que reduce parámetros y sobreajuste.
- Cabeza fully-connected con Dropout: regulariza y mapea a las clases.

Entradas/salidas:
- Entrada: (N, 1, H, W), recomendado H=W=224
- Salida: logits (N, num_classes)k   
"""

from __future__ import annotations

from typing import Tuple
import torch
from torch import nn


class ConvBlock(nn.Module):
    """Bloque Conv-BN-ReLU"""

    def __init__(self, in_ch: int, out_ch: int, *, k: int = 3, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CxrSmallCNN(nn.Module):
    """
    Arquitectura CNN compacta para CXR.

    [Notebook] Celda: Código de la arquitectura (modelo)
    """

    def __init__(self, num_classes: int, in_channels: int = 1, dropout_p: float = 0.3):
        super().__init__()

        # [Notebook] Celda: Capas de la red (features)
        self.features = nn.Sequential(
            # Bloque 1: (N, C, 224, 224) -> (N, 32, 112, 112)
            ConvBlock(in_channels, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 2: (N, 32, 112, 112) -> (N, 64, 56, 56)
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 3: (N, 64, 56, 56) -> (N, 128, 28, 28)
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 4: (N, 128, 28, 28) -> (N, 256, 14, 14)
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Pool global: (N, 256, 14, 14) -> (N, 256, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # [Notebook] Celda: Cabeza (fully connected)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (N, 256)
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes),  # logits
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [Notebook] Celda: Forward y dimensiones intermedias
        x = self.features(x)  # -> (N, 256, 1, 1)
        logits = self.classifier(x)  # -> (N, num_classes)
        return logits


def build_cxr_small_cnn(num_classes: int, in_channels: int = 1) -> CxrSmallCNN:
    """Constructor utilitario del modelo."""
    return CxrSmallCNN(num_classes=num_classes, in_channels=in_channels)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (1, 224, 224)) -> str:
    """Resumen textual simple (sin dependencias externas)."""
    lines = []
    lines.append(repr(model))
    lines.append(f"\nTrainable params: {count_parameters(model):,}")
    lines.append(f"Input size: {input_size}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Ejemplo rápido de resumen
    m = build_cxr_small_cnn(num_classes=3, in_channels=1)
    print(model_summary(m))


