"""
Entrenamiento rápido con subconjunto de ~10,000 imágenes.
Útil para pruebas en CPU.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import random

from src.config import (
    DEVICE, BATCH_SIZE, NUM_WORKERS, EPOCHS,
    CHECKPOINT_BEST, CHECKPOINT_LAST, PLOTS_DIR,
    LEARNING_RATE_LAST_LAYERS, LEARNING_RATE_FINETUNE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
)
from src.data_pipeline import DeepFakeDataset, get_train_transforms, get_val_transforms
from src.model import build_densenet121, get_optimizer_and_scheduler, count_parameters
from src.train import train_one_epoch, validate_one_epoch, evaluate
from src.utils import logger, set_seed, plot_training_history


def train_subset(
    subset_size: int = 10000,
    epochs: int = 10,
    batch_size: int = 32,
):
    """Entrena con un subconjunto de imágenes."""
    set_seed(42)
    device = torch.device(DEVICE)
    logger.info(f"Usando dispositivo: {device}")
    logger.info(f"Subconjunto: {subset_size} imágenes de entrenamiento")

    # Cargar datasets completos
    train_full = DeepFakeDataset("train", transform=get_train_transforms())
    val_full = DeepFakeDataset("validation", transform=get_val_transforms())
    test_full = DeepFakeDataset("test", transform=get_val_transforms())

    # Crear subconjunto estratificado (mitad real, mitad fake)
    real_indices = [i for i, (_, l) in enumerate(train_full.samples) if l == 1]
    fake_indices = [i for i, (_, l) in enumerate(train_full.samples) if l == 0]

    half = subset_size // 2
    selected = random.sample(real_indices, min(half, len(real_indices))) + \
               random.sample(fake_indices, min(half, len(fake_indices)))

    train_sub = Subset(train_full, selected)

    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_full, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_full, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    logger.info(f"Train subset: {len(train_sub)} imágenes "
                f"({sum(1 for i in selected if train_full.samples[i][1]==1)} reales, "
                f"{sum(1 for i in selected if train_full.samples[i][1]==0)} falsas)")
    logger.info(f"Val: {len(val_full)} imágenes")
    logger.info(f"Test: {len(test_full)} imágenes")

    # Modelo
    model = build_densenet121(freeze_until_block=3)
    model.to(device)

    counts = count_parameters(model)
    logger.info(f"Params: {counts['trainable']:,} entrenables / {counts['total']:,} totales")

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, lr_last_layers=LEARNING_RATE_LAST_LAYERS,
        lr_finetune=LEARNING_RATE_FINETUNE, weight_decay=WEIGHT_DECAY,
    )

    # Entrenamiento
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    early_stop = 0

    logger.info(f"\n{'='*60}")
    logger.info(f"  INICIANDO ENTRENAMIENTO ({epochs} épocas, subset={subset_size})")
    logger.info(f"{'='*60}\n")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {train_loss:.4f} loss, {train_acc:.4f} acc | "
            f"Val: {val_loss:.4f} loss, {val_acc:.4f} acc"
        )

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, CHECKPOINT_BEST)
            logger.info(f"  ✓ Nuevo mejor modelo (val_acc={val_acc:.4f})")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
        }, CHECKPOINT_LAST)

        # Early stopping
        if epoch > 2 and val_loss > history["val_loss"][-3] + EARLY_STOPPING_MIN_DELTA:
            early_stop += 1
            if early_stop >= EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping en época {epoch+1}")
                break
        else:
            early_stop = 0

    logger.info(f"\n✓ Entrenamiento completado. Mejor val_acc: {best_val_acc:.4f}")

    # Graficar
    plot_training_history(history, save_path=PLOTS_DIR / "training_subset.png")
    logger.info(f"Gráfico guardado en: {PLOTS_DIR / 'training_subset.png'}")

    # Evaluación final
    logger.info("\nEvaluando en conjunto de prueba...")
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test Accuracy: {test_metrics['metrics']['accuracy']:.4f}")
    logger.info(f"Test AUC: {test_metrics['metrics']['auc']:.4f}")

    return history, test_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=10000, help="Tamaño del subconjunto")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    train_subset(subset_size=args.subset, epochs=args.epochs, batch_size=args.batch_size)
