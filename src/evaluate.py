"""
Script de evaluación independiente.
Carga un modelo entrenado desde checkpoint y lo evalúa en el conjunto de prueba.
Genera todas las métricas y gráficos.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    CHECKPOINT_BEST,
    CHECKPOINT_LAST,
    PLOTS_DIR,
)
from src.data_pipeline import get_dataloaders
from src.model import build_densenet121
from src.train import evaluate
from src.utils import logger, set_seed


def load_model_from_checkpoint(
    checkpoint_path: Path,
    freeze_until_block: int = 3,
) -> nn.Module:
    """
    Carga un modelo desde un checkpoint guardado.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

    device = torch.device(DEVICE)
    model = build_densenet121(freeze_until_block=freeze_until_block)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        f"Modelo cargado desde: {checkpoint_path} "
        f"(época {checkpoint.get('epoch', '?')}, "
        f"val_acc={checkpoint.get('val_acc', '?'):.4f})"
    )

    return model


def run_evaluation(
    checkpoint_path: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Ejecuta la evaluación completa del modelo.

    Args:
        checkpoint_path: Ruta al checkpoint. Si es None, usa el mejor modelo.
        batch_size: Tamaño del batch para evaluación.

    Returns:
        results: Diccionario con métricas y gráficos
    """
    set_seed(42)

    if checkpoint_path is None:
        # Usar el mejor modelo disponible
        if CHECKPOINT_BEST.exists():
            checkpoint_path = CHECKPOINT_BEST
        elif CHECKPOINT_LAST.exists():
            checkpoint_path = CHECKPOINT_LAST
        else:
            raise FileNotFoundError(
                "No se encontró ningún checkpoint. "
                "Entrena el modelo primero con: python -m src.train"
            )

    logger.info("=== EVALUACIÓN DEL MODELO ===")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dispositivo: {DEVICE}")

    # Cargar modelo
    model = load_model_from_checkpoint(checkpoint_path)

    # Cargar datos de prueba
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )

    # Evaluar
    criterion = nn.CrossEntropyLoss()
    device = torch.device(DEVICE)
    test_results = evaluate(model, test_loader, criterion, device)

    logger.info("\n✅ Evaluación completada exitosamente.")
    logger.info(f"Resultados guardados en: {PLOTS_DIR}")

    return test_results


if __name__ == "__main__":
    results = run_evaluation()
