"""
Funciones utilitarias para el sistema de detección de deepfakes.
Incluye helpers para logging, visualización, y manejo de imágenes.
"""

import base64
import io
import json
import logging
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from src.config import (
    CLASS_NAMES,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PLOTS_DIR,
)

# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logger(name: str = "deepfake_detector") -> logging.Logger:
    """Configura y retorna un logger con formato estándar."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


# ─── Semilla ──────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fija la semilla aleatoria para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Procesamiento de imágenes ────────────────────────────────────────────────

def preprocess_image(
    image: Image.Image,
    img_size: int = IMG_SIZE,
) -> torch.Tensor:
    """
    Preprocesa una imagen PIL para el modelo DenseNet-121:
    - Redimensiona a img_size x img_size
    - Convierte a tensor
    - Normaliza con mean/std de ImageNet
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)  # batch dimension


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convierte un tensor normalizado de vuelta a imagen PIL."""
    img = tensor.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0).numpy()
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convierte una imagen PIL a string base64."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64_str: str) -> Image.Image:
    """Convierte un string base64 a imagen PIL."""
    buffer = io.BytesIO(base64.b64decode(b64_str))
    return Image.open(buffer)


# ─── Visualización ────────────────────────────────────────────────────────────

def plot_training_history(
    history: dict,
    save_path: Optional[Path] = None,
) -> None:
    """
    Genera gráficos de pérdida y precisión a lo largo de las épocas.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pérdida
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Precisión
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training history plot saved to {save_path}")

    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    Genera y guarda la matriz de confusión.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.close()


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    save_path: Optional[Path] = None,
) -> None:
    """
    Genera y guarda la curva ROC.
    """
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    plt.close()


def plot_robustness_curve(
    quality_factors: list,
    accuracies: list,
    save_path: Optional[Path] = None,
) -> None:
    """
    Genera gráfico de Precisión vs Factor de Calidad (robustez).
    """
    plt.figure(figsize=(8, 5))
    plt.plot(
        quality_factors, accuracies, "r-o", linewidth=2, markersize=8,
    )
    plt.xlabel("JPEG Quality Factor (QF)")
    plt.ylabel("Accuracy")
    plt.title("Model Robustness: Accuracy vs JPEG Compression Quality")
    # Eje X en orden descendente (100 → 50: de mayor a menor calidad)
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)

    # Anotar valores
    for qf, acc in zip(quality_factors, accuracies):
        plt.annotate(
            f"{acc:.2%}",
            (qf, acc),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10,
        )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Robustness curve saved to {save_path}")

    plt.close()


# ─── Métricas ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list, y_prob: list) -> dict:
    """
    Calcula Accuracy, Precision, Recall, F1-Score y AUC.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        confusion_matrix,
        roc_curve,
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary"),
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1_score": f1_score(y_true, y_pred, average="binary"),
        "auc": roc_auc_score(y_true, y_prob),
    }

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    return metrics, cm, fpr, tpr


def print_metrics_table(metrics: dict, title: str = "Evaluation Metrics") -> None:
    """Imprime una tabla formateada de métricas."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(f"  {'Metric':<20} {'Value':<10}")
    print(f"  {'-' * 30}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<20} {value:.4f}")
        else:
            print(f"  {key:<20} {value}")
    print(f"{'=' * 50}\n")
