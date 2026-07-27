"""
Pruebas de robustez: evalúa el modelo con imágenes comprimidas a diferentes
niveles de calidad JPEG (QF=100, 75, 50) y reporta el impacto en precisión.
"""

import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    QUALITY_FACTORS,
    PLOTS_DIR,
    DATA_DIR,
)
from src.evaluate import load_model_from_checkpoint
from src.model import build_densenet121
from src.utils import (
    logger,
    set_seed,
    plot_robustness_curve,
    compute_metrics,
    print_metrics_table,
)


class CompressedDataset(Dataset):
    """
    Dataset que aplica compresión JPEG en tiempo real.
    Útil para pruebas de robustez sin duplicar almacenamiento.
    """

    def __init__(self, split: str, quality_factor: int, transform=None):
        self.split = split
        self.quality_factor = quality_factor
        self.transform = transform

        self.data_dir = DATA_DIR / split

        self.samples = []
        for label, class_name in enumerate(["fake", "real"]):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.iterdir()):
                    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        self.samples.append((str(img_path), label))

        logger.info(
            f"CompressedDataset (QF={quality_factor}): {len(self.samples)} muestras"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Cargar imagen original
        image = Image.open(img_path).convert("RGB")

        # Aplicar compresión JPEG
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality_factor)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")

        if self.transform:
            compressed = self.transform(compressed)

        return compressed, label


def get_compressed_dataloaders(
    quality_factor: int,
    split: str = "test",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> DataLoader:
    """
    Crea un DataLoader con imágenes comprimidas a un factor de calidad dado.

    Args:
        quality_factor: Factor de calidad JPEG (1-100)
        split: split del dataset ('test', 'validation')
        batch_size: Tamaño del batch
        num_workers: Trabajadores para DataLoader

    Returns:
        DataLoader con imágenes comprimidas
    """
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = CompressedDataset(
        split=split,
        quality_factor=quality_factor,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate_compressed(
    model: nn.Module,
    quality_factor: int,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Evalúa el modelo en imágenes comprimidas a un factor de calidad dado.

    Args:
        model: Modelo a evaluar
        quality_factor: Factor de calidad JPEG
        batch_size: Tamaño del batch

    Returns:
        metrics: Diccionario con métricas para este factor de calidad
    """
    device = torch.device(DEVICE)
    model.eval()

    dataloader = get_compressed_dataloaders(
        quality_factor=quality_factor,
        batch_size=batch_size,
    )

    all_labels = []
    all_preds = []
    all_probs = []

    for inputs, labels in tqdm(
        dataloader,
        desc=f"Evaluating QF={quality_factor}",
    ):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    metrics, cm, fpr, tpr = compute_metrics(all_labels, all_preds, all_probs)

    return {
        "quality_factor": quality_factor,
        "metrics": metrics,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
    }


def run_robustness_test(
    checkpoint_path: Optional[Path] = None,
    quality_factors: Optional[List[int]] = None,
    baseline_acc: Optional[float] = None,
) -> dict:
    """
    Ejecuta pruebas de robustez completas.

    Args:
        checkpoint_path: Ruta al checkpoint del modelo
        quality_factors: Lista de factores de calidad a probar
        baseline_acc: Precisión base (sin compresión) para comparación

    Returns:
        results: Diccionario con resultados de todas las pruebas
    """
    set_seed(42)

    if quality_factors is None:
        quality_factors = QUALITY_FACTORS

    logger.info("=== PRUEBAS DE ROBUSTEZ ===")
    logger.info(f"Factores de calidad a probar: {quality_factors}")

    # Cargar modelo
    model = load_model_from_checkpoint(checkpoint_path)

    # Evaluar en cada nivel de compresión
    all_results = []

    for qf in quality_factors:
        logger.info(f"\nProbando con QF={qf}...")
        result = evaluate_compressed(model, qf)
        all_results.append(result)

        metrics = result["metrics"]
        logger.info(
            f"  QF={qf}: Accuracy={metrics['accuracy']:.4f}, "
            f"F1={metrics['f1_score']:.4f}, AUC={metrics['auc']:.4f}"
        )

    # Compilar resultados
    quality_values = [r["quality_factor"] for r in all_results]
    accuracy_values = [r["metrics"]["accuracy"] for r in all_results]
    f1_values = [r["metrics"]["f1_score"] for r in all_results]
    auc_values = [r["metrics"]["auc"] for r in all_results]

    # Mostrar tabla comparativa
    print("\n" + "=" * 70)
    print("  TABLA COMPARATIVA: ROBUSTEZ DEL MODELO")
    print("=" * 70)
    print(f"  {'Quality Factor':<20} {'Accuracy':<12} {'F1-Score':<12} {'AUC':<12}")
    print(f"  {'-' * 56}")

    for qf, acc, f1, auc in zip(quality_values, accuracy_values, f1_values, auc_values):
        acc_str = f"{acc:.4f}"
        f1_str = f"{f1:.4f}"
        auc_str = f"{auc:.4f}"

        # Calcular degradación si tenemos baseline
        if baseline_acc is not None:
            degradation = (baseline_acc - acc) / baseline_acc * 100
            acc_str += f" ({degradation:+.2f}%)"

        print(f"  {qf:<20} {acc_str:<12} {f1_str:<12} {auc_str:<12}")

    if baseline_acc is not None:
        print(f"\n  (*) Degradación relativa respecto a baseline (sin compresión)")
    print("=" * 70 + "\n")

    # Graficar curva de robustez
    plot_path = PLOTS_DIR / "robustness_curve.png"
    plot_robustness_curve(quality_values, accuracy_values, save_path=plot_path)

    logger.info(f"Curva de robustez guardada en: {plot_path}")

    return {
        "quality_factors": quality_values,
        "accuracies": accuracy_values,
        "f1_scores": f1_values,
        "auc_scores": auc_values,
        "detailed_results": all_results,
    }


if __name__ == "__main__":
    results = run_robustness_test()
