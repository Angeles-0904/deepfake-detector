"""
Definición del modelo DenseNet-121 con Transfer Learning para
clasificación binaria (REAL/FAKE).
"""

import torch
import torch.nn as nn
import torchvision.models as models

from src.config import NUM_CLASSES, DENSENET_FEATURES, DEVICE


def build_densenet121(
    freeze_until_block: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """
    Construye DenseNet-121 con Transfer Learning.

    Args:
        freeze_until_block: Nivel de congelamiento:
            - 0: no congelar nada (fine-tuning completo)
            - 1: congelar primeras capas (bloque 1)
            - 2: congelar hasta bloque 2
            - 3: congelar hasta bloque 3
            - 4: congelar todo excepto clasificador
        pretrained: Si usar pesos pre-entrenados en ImageNet

    Returns:
        model: Modelo DenseNet-121 modificado
    """
    # Cargar modelo pre-entrenado
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)

    # Congelar capas seleccionadas
    if freeze_until_block > 0:
        _freeze_blocks(model, freeze_until_block)

    # Reemplazar clasificador para salida binaria
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES),
    )

    return model


def _freeze_blocks(model: nn.Module, freeze_until_block: int) -> None:
    """
    Congela las capas convolucionales hasta un cierto bloque de DenseNet-121.

    La arquitectura DenseNet-121 tiene 4 bloques densos (features.denseblock1-4)
    con capas de transición entre ellos. También incluye features.conv0 y
    features.norm0 al inicio.
    """
    # DenseNet-121 feature blocks: conv0, norm0, relu0, pool0,
    #   denseblock1, transition1, denseblock2, transition2,
    #   denseblock3, transition3, denseblock4, norm5

    blocks_to_freeze = []

    # Siempre congelar las capas iniciales (conv0, norm0)
    blocks_to_freeze.append(model.features.conv0)
    blocks_to_freeze.append(model.features.norm0)

    if freeze_until_block >= 1:
        blocks_to_freeze.append(model.features.relu0)
        blocks_to_freeze.append(model.features.pool0)
        blocks_to_freeze.append(model.features.denseblock1)
        blocks_to_freeze.append(model.features.transition1)

    if freeze_until_block >= 2:
        blocks_to_freeze.append(model.features.denseblock2)
        blocks_to_freeze.append(model.features.transition2)

    if freeze_until_block >= 3:
        blocks_to_freeze.append(model.features.denseblock3)
        blocks_to_freeze.append(model.features.transition3)

    if freeze_until_block >= 4:
        blocks_to_freeze.append(model.features.denseblock4)
        blocks_to_freeze.append(model.features.norm5)

    for block in blocks_to_freeze:
        for param in block.parameters():
            param.requires_grad = False


def get_optimizer_and_scheduler(
    model: nn.Module,
    lr_last_layers: float = 1e-4,
    lr_finetune: float = 1e-5,
    weight_decay: float = 1e-4,
):
    """
    Configura el optimizador Adam con diferentes learning rates:
    - lr_finetune para las capas pre-entrenadas descongeladas (bloques densos)
    - lr_last_layers para el clasificador nuevo (cabeza)
    """
    # Separar parámetros por tipo: clasificador nuevo vs capas pre-entrenadas
    classifier_params = []
    pretrained_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            classifier_params.append(param)
        else:
            pretrained_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": classifier_params, "lr": lr_last_layers},
        {"params": pretrained_params, "lr": lr_finetune},
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
        min_lr=1e-7,
    )

    return optimizer, scheduler


def count_parameters(model: nn.Module) -> dict:
    """Cuenta parámetros totales y entrenables del modelo."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": 100.0 * trainable / total if total > 0 else 0,
    }


if __name__ == "__main__":
    # Prueba de construcción del modelo
    model = build_densenet121(freeze_until_block=3)
    model.to(DEVICE)

    counts = count_parameters(model)
    print(f"Total params:     {counts['total']:,}")
    print(f"Trainable params: {counts['trainable']:,} ({counts['trainable_percent']:.2f}%)")
    print(f"Frozen params:    {counts['frozen']:,}")

    # Prueba de forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(DEVICE)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Esperado: [2, 2]
