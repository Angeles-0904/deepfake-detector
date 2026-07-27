"""
Entrenamiento ULTRA-RÁPIDO para CPU.
Configuración mínima para probar que el pipeline funciona.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import random
import time

from src.config import (
    DEVICE, CHECKPOINT_BEST, CHECKPOINT_LAST, PLOTS_DIR,
)
from src.data_pipeline import DeepFakeDataset, get_train_transforms, get_val_transforms
from src.model import build_densenet121, get_optimizer_and_scheduler, count_parameters
from src.train import train_one_epoch, validate_one_epoch, evaluate
from src.utils import logger, set_seed, plot_training_history


def train_quick(
    subset_size: int = 2000,
    epochs: int = 5,
    batch_size: int = 16,
):
    """Entrenamiento ultra-rápido con configuración mínima."""
    set_seed(42)
    device = torch.device(DEVICE)
    logger.info(f"🚀 MODO RÁPIDO - Dispositivo: {device}")
    logger.info(f"   Subconjunto: {subset_size} imágenes")
    logger.info(f"   Épocas: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    
    start_time = time.time()

    # Cargar datasets
    logger.info("Cargando datasets...")
    train_full = DeepFakeDataset("train", transform=get_train_transforms())
    val_full = DeepFakeDataset("validation", transform=get_val_transforms())
    test_full = DeepFakeDataset("test", transform=get_val_transforms())

    # Crear subconjunto estratificado
    real_indices = [i for i, (_, l) in enumerate(train_full.samples) if l == 1]
    fake_indices = [i for i, (_, l) in enumerate(train_full.samples) if l == 0]

    half = subset_size // 2
    selected = random.sample(real_indices, min(half, len(real_indices))) + \
               random.sample(fake_indices, min(half, len(fake_indices)))

    train_sub = Subset(train_full, selected)

    train_loader = DataLoader(
        train_sub, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,  # num_workers=0 para CPU
    )
    val_loader = DataLoader(
        val_full, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_full, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    logger.info(f"Train: {len(train_sub)} imágenes")
    logger.info(f"Val: {len(val_full)} imágenes")
    logger.info(f"Test: {len(test_full)} imágenes")

    # Modelo (más ligero: sin fine-tuning profundo)
    model = build_densenet121(freeze_until_block=4)  # Congelar todo excepto clasificador
    model.to(device)

    counts = count_parameters(model)
    logger.info(f"Params: {counts['trainable']:,} entrenables / {counts['total']:,} totales")

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, lr_last_layers=1e-3,  # LR más alto para converger rápido
        lr_finetune=1e-5, weight_decay=1e-4,
    )

    # Entrenamiento
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    logger.info(f"\n{'='*60}")
    logger.info(f"  🚀 ENTRENAMIENTO RÁPIDO ({epochs} épocas)")
    logger.info(f"{'='*60}\n")

    for epoch in range(epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
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

    total_time = time.time() - start_time
    logger.info(f"\n✅ Entrenamiento completado en {total_time:.1f} segundos")
    logger.info(f"   Mejor val_acc: {best_val_acc:.4f}")

    # Graficar
    plot_training_history(history, save_path=PLOTS_DIR / "training_quick.png")

    # Evaluación final
    logger.info("\nEvaluando en conjunto de prueba...")
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test Accuracy: {test_metrics['metrics']['accuracy']:.4f}")
    logger.info(f"Test AUC: {test_metrics['metrics']['auc']:.4f}")

    return history, test_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entrenamiento rápido para CPU")
    parser.add_argument("--subset", type=int, default=2000, help="Tamaño del subconjunto (default: 2000)")
    parser.add_argument("--epochs", type=int, default=5, help="Número de épocas (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    args = parser.parse_args()

    train_quick(subset_size=args.subset, epochs=args.epochs, batch_size=args.batch_size)
