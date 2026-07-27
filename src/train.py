"""
Pipeline completo de entrenamiento del modelo DenseNet-121.
Incluye early stopping, checkpoints, y logging de métricas.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import (
    DEVICE,
    EPOCHS,
    BATCH_SIZE,
    NUM_WORKERS,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    CHECKPOINT_BEST,
    CHECKPOINT_LAST,
    PLOTS_DIR,
    LEARNING_RATE_LAST_LAYERS,
    LEARNING_RATE_FINETUNE,
    WEIGHT_DECAY,
)
from src.data_pipeline import get_dataloaders
from src.model import build_densenet121, get_optimizer_and_scheduler, count_parameters
from src.utils import (
    logger,
    set_seed,
    plot_training_history,
    compute_metrics,
    print_metrics_table,
    plot_confusion_matrix,
    plot_roc_curve,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Entrena el modelo por una época y retorna pérdida y precisión promedio."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{correct / total:.4f}",
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evalúa el modelo en un conjunto de validación."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total:.4f}",
            })

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    freeze_until_block: int = 3,
    resume_from: Optional[Path] = None,
) -> dict:
    """
    Ejecuta el pipeline completo de entrenamiento.

    Args:
        epochs: Número máximo de épocas
        batch_size: Tamaño del batch
        num_workers: Trabajadores para DataLoader
        freeze_until_block: Nivel de congelamiento (0-4)
        resume_from: Ruta a checkpoint para reanudar entrenamiento

    Returns:
        history: Diccionario con el historial de entrenamiento
    """
    set_seed(42)
    device = torch.device(DEVICE)
    logger.info(f"Usando dispositivo: {device}")
    logger.info(f"Entrenando por {epochs} épocas con batch_size={batch_size}")

    # Preparar datos
    logger.info("Cargando datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Construir modelo
    logger.info(f"Construyendo DenseNet-121 (freeze_until_block={freeze_until_block})...")
    model = build_densenet121(freeze_until_block=freeze_until_block)
    model.to(device)

    param_counts = count_parameters(model)
    logger.info(
        f"Parámetros: {param_counts['total']:,} totales, "
        f"{param_counts['trainable']:,} entrenables "
        f"({param_counts['trainable_percent']:.2f}%)"
    )

    # Configurar optimización
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(
        model,
        lr_last_layers=LEARNING_RATE_LAST_LAYERS,
        lr_finetune=LEARNING_RATE_FINETUNE,
        weight_decay=WEIGHT_DECAY,
    )

    # Reanudar desde checkpoint si se especifica
    start_epoch = 0
    best_val_acc = 0.0
    if resume_from and resume_from.exists():
        logger.info(f"Reanudando desde checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        logger.info(f"Reanudando desde época {start_epoch}")

    # Entrenamiento
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr_history": [],
        "epochs_trained": 0,
    }

    early_stopping_counter = 0
    best_val_loss = float("inf")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  INICIANDO ENTRENAMIENTO")
    logger.info(f"{'=' * 60}\n")

    for epoch in range(start_epoch, epochs):
        logger.info(f"Época {epoch + 1}/{epochs}")

        # Entrenar
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validar
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Guardar historial
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr_history"].append(current_lr)
        history["epochs_trained"] = epoch + 1

        logger.info(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}  |  "
            f"LR: {current_lr:.2e}"
        )

        # Early Stopping
        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            logger.info(f"  Early stopping counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")

        # Checkpoint: guardar si mejora precisión de validación
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "best_val_acc": best_val_acc,
            }, CHECKPOINT_BEST)
            logger.info(f"  ✓ Nuevo mejor modelo guardado (val_acc={val_acc:.4f})")

        # Guardar último checkpoint siempre
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "best_val_acc": best_val_acc,
        }, CHECKPOINT_LAST)

        # Early stopping
        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping activado después de {epoch + 1} épocas")
            break

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  ENTRENAMIENTO COMPLETADO")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Mejor precisión de validación: {best_val_acc:.4f}")
    logger.info(f"  Épocas entrenadas: {history['epochs_trained']}")

    # Guardar historial
    history_path = PLOTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        # Convertir a serializable
        serializable_history = {
            k: [float(v) if isinstance(v, (float, int)) else v for v in vals]
            for k, vals in history.items()
        }
        json.dump(serializable_history, f, indent=2)
    logger.info(f"Historial guardado en {history_path}")

    # Generar gráficos de entrenamiento
    plot_path = PLOTS_DIR / "training_history.png"
    plot_training_history(history, save_path=plot_path)

    # Evaluación final en conjunto de prueba
    logger.info("\nEvaluando en conjunto de prueba...")
    test_metrics = evaluate(model, test_loader, criterion, device)

    # Guardar métricas finales
    metrics_path = PLOTS_DIR / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics["metrics"], f, indent=2)

    return history, test_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evalúa el modelo en un dataloader y retorna métricas detalladas.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    running_loss = 0.0
    total = 0

    for inputs, labels in tqdm(dataloader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilidad de clase REAL

    avg_loss = running_loss / total

    # Calcular métricas
    metrics, cm, fpr, tpr = compute_metrics(all_labels, all_preds, all_probs)
    metrics["test_loss"] = avg_loss

    print_metrics_table(metrics, title="Test Evaluation Metrics")

    # Guardar gráficos
    plot_confusion_matrix(cm, save_path=PLOTS_DIR / "confusion_matrix.png")
    plot_roc_curve(fpr, tpr, metrics["auc"], save_path=PLOTS_DIR / "roc_curve.png")

    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "all_labels": all_labels,
        "all_preds": all_preds,
        "all_probs": all_probs,
    }


if __name__ == "__main__":
    history, test_metrics = train()
