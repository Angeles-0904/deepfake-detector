"""
Pipeline de datos: descarga, preprocesamiento, aumento de datos y división
del dataset 140K Real and Fake Faces en entrenamiento/validación/prueba.
"""

import shutil
from pathlib import Path
from typing import Optional, Tuple

import kagglehub
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import (
    DATA_DIR,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    KAGGLE_DATASET,
)
from src.utils import logger


# ─── Dataset personalizado ────────────────────────────────────────────────────

class DeepFakeDataset(Dataset):
    """
    Dataset personalizado para imágenes de deepfakes.
    Espera estructura: data/{split}/real/ y data/{split}/fake/
    """

    def __init__(self, split: str, transform=None):
        self.split = split
        self.data_dir = DATA_DIR / split
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Directorio de datos no encontrado: {self.data_dir}. "
                f"Ejecute primero el pipeline de descarga."
            )

        self.samples = []
        for label, class_name in enumerate(["fake", "real"]):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.iterdir()):
                    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        self.samples.append((str(img_path), label))

        logger.info(
            f"Dataset '{split}': {len(self.samples)} muestras "
            f"({sum(1 for _, l in self.samples if l == 1)} reales, "
            f"{sum(1 for _, l in self.samples if l == 0)} falsas)"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ─── Transformaciones ─────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """Transformaciones para entrenamiento (con aumento de datos)."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Transformaciones para validación/prueba (sin aumento)."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── DataLoaders ──────────────────────────────────────────────────────────────

def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Retorna los DataLoaders para entrenamiento, validación y prueba.
    """
    train_dataset = DeepFakeDataset("train", transform=get_train_transforms())
    val_dataset = DeepFakeDataset("validation", transform=get_val_transforms())
    test_dataset = DeepFakeDataset("test", transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ─── Descarga y organización del dataset ──────────────────────────────────────

def download_dataset() -> Path:
    """
    Descarga el dataset 140K Real and Fake Faces usando kagglehub.
    Retorna la ruta donde se descargó.
    """
    logger.info(f"Descargando dataset '{KAGGLE_DATASET}' desde Kaggle...")
    logger.info("Esto puede tomar varios minutos dependiendo de tu conexión.")

    try:
        download_path = kagglehub.dataset_download(KAGGLE_DATASET)
        logger.info(f"Dataset descargado en: {download_path}")
        return Path(download_path)
    except Exception as e:
        logger.error(f"Error al descargar el dataset: {e}")
        logger.error(
            "Asegúrate de tener configurada la API de Kaggle correctamente. "
            "Visita: https://www.kaggle.com/docs/api"
        )
        raise


def _find_real_fake_dirs(directory: Path):
    """
    Recursively search for 'real' and 'fake' directories inside a directory.
    Returns (real_dir, fake_dir) or (None, None) if not found.
    """
    real_dir = fake_dir = None
    for item in directory.rglob("*"):
        if item.is_dir():
            if item.name == "real" and real_dir is None:
                real_dir = item
            elif item.name == "fake" and fake_dir is None:
                fake_dir = item
        if real_dir and fake_dir:
            break
    return real_dir, fake_dir


def organize_dataset(source_dir: Path) -> None:
    """
    Organiza las imágenes descargadas en las carpetas train/val/test
    con subcarpetas real/ y fake/.

    Maneja múltiples formatos del dataset:
    1. CSVs + real_vs_fake/ con subcarpeta extra (ej: real-vs-fake/train/real/)
    2. CSVs + real_vs_fake/train/real/ directamente
    3. Carpetas train/real/, valid/real/, test/real/ directamente
    4. Carpetas real/ y fake/ directamente
    """
    # ── Formato 1 y 2: Kaggle con CSVs ────────────────────────────────────
    train_csv = source_dir / "train.csv"
    test_csv = source_dir / "test.csv"
    valid_csv = source_dir / "valid.csv"

    if train_csv.exists() and test_csv.exists() and valid_csv.exists():
        logger.info("Detectado formato Kaggle con CSVs. Organizando dataset...")

        # Buscar la carpeta que contiene train/real/, train/fake/, etc.
        images_dir = _find_real_fake_images_dir(source_dir)
        if images_dir is None:
            raise FileNotFoundError(
                f"No se encontró una carpeta con train/real/ o train/fake/ en {source_dir}. "
                f"Contenido: {list(source_dir.iterdir())}"
            )
        logger.info(f"Carpeta de imágenes encontrada: {images_dir}")

        # Mapeo de splits: nombre en CSV -> (archivo CSV, carpeta destino)
        splits = {
            "train": (train_csv, DATA_DIR / "train"),
            "validation": (valid_csv, DATA_DIR / "validation"),
            "test": (test_csv, DATA_DIR / "test"),
        }

        for split_name, (csv_file, dest_base) in splits.items():
            # Detectar la subcarpeta correcta del split en images_dir
            split_dir = None
            for candidate in [split_name, "valid"]:
                candidate_dir = images_dir / candidate
                if candidate_dir.exists():
                    split_dir = candidate_dir
                    break

            if split_dir is None:
                logger.warning(f"  No se encontró carpeta '{split_name}' en {images_dir}")
                continue

            # Copiar directamente desde train/real/, train/fake/, etc.
            # (ya están organizados, no necesitamos parsear CSVs)
            for class_name in ["real", "fake"]:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    logger.warning(f"  No se encontró {class_dir}")
                    continue

                dest_dir = dest_base / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                imgs = [f for f in class_dir.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png")]

                for img in tqdm(imgs, desc=f"  {split_name}/{class_name}"):
                    dest = dest_dir / img.name
                    if not dest.exists():
                        shutil.copy2(img, dest)

                copied = len(list(dest_dir.glob("*")))
                logger.info(f"  {split_name}/{class_name}: {copied} imágenes")

    # ── Formato 3: Carpetas train/real/, valid/real/, test/real/ ────────────
    elif _has_split_dirs(source_dir):
        logger.info("Detectado formato organizado con subcarpetas train/valid/test.")
        _copy_split_dirs(source_dir)

    # ── Formato 4: Carpetas real/ y fake/ directamente ──────────────────────
    else:
        _copy_flat_dirs(source_dir)

    logger.info("Dataset organizado exitosamente:")
    for split_name in ["train", "validation", "test"]:
        real_count = len(list((DATA_DIR / split_name / "real").glob("*")))
        fake_count = len(list((DATA_DIR / split_name / "fake").glob("*")))
        logger.info(f"  {split_name}: {real_count} reales, {fake_count} falsas")


def _find_real_fake_images_dir(source_dir: Path) -> Optional[Path]:
    """
    Search for a directory that contains train/real/ or train/fake/ subdirectories.
    Handles nested structures like real_vs_fake/real-vs-fake/train/real/.
    Searches up to 3 levels deep.
    """
    # Level 1: check source_dir directly
    if (source_dir / "train" / "real").exists() or (source_dir / "train" / "fake").exists():
        return source_dir

    # Level 2: check subdirectories
    for subdir in source_dir.iterdir():
        if subdir.is_dir():
            if (subdir / "train" / "real").exists() or (subdir / "train" / "fake").exists():
                return subdir

    # Level 3: check sub-subdirectories
    for subdir in source_dir.iterdir():
        if subdir.is_dir():
            for subsubdir in subdir.iterdir():
                if subsubdir.is_dir():
                    if (subsubdir / "train" / "real").exists() or (subsubdir / "train" / "fake").exists():
                        return subsubdir

    return None


def _has_split_dirs(source_dir: Path) -> bool:
    """Check if source_dir has train/real or train/fake subdirectories."""
    return ((source_dir / "train" / "real").exists() or
            (source_dir / "train" / "fake").exists())


def _copy_split_dirs(source_dir: Path) -> None:
    """Copy images from organized split directories (train/real/, valid/fake/, etc.)."""
    split_mapping = [
        ("train", source_dir / "train"),
        ("validation", source_dir / "valid"),
        ("test", source_dir / "test"),
    ]
    for split_name, split_dir in split_mapping:
        if not split_dir.exists():
            continue
        for class_name in ["real", "fake"]:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            dest_dir = DATA_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for img_path in tqdm(class_dir.iterdir(), desc=f"Copiando {split_name}/{class_name}"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    dest = dest_dir / img_path.name
                    if not dest.exists():
                        shutil.copy2(img_path, dest)


def _copy_flat_dirs(source_dir: Path) -> None:
    """Copy from flat real/fake directories, splitting into train/val/test."""
    real_dir = source_dir / "real"
    fake_dir = source_dir / "fake"
    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            f"No se encontraron carpetas 'real' y 'fake' en {source_dir}. "
            f"Contenido: {list(source_dir.iterdir())}"
        )
    real_images = sorted([str(p) for p in real_dir.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    fake_images = sorted([str(p) for p in fake_dir.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    X = real_images + fake_images
    y = [1] * len(real_images) + [0] * len(fake_images)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y)
    val_ratio_adj = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adj, random_state=RANDOM_SEED, stratify=y_temp)
    for split_name, (images, labels) in [("train", (X_train, y_train)),
                                         ("validation", (X_val, y_val)),
                                         ("test", (X_test, y_test))]:
        for label, class_name in enumerate(["fake", "real"]):
            (DATA_DIR / split_name / class_name).mkdir(parents=True, exist_ok=True)
        for img_path, label in tqdm(zip(images, labels), desc=f"Organizando {split_name}", total=len(images)):
            class_name = "real" if label == 1 else "fake"
            dest = DATA_DIR / split_name / class_name / Path(img_path).name
            if not dest.exists():
                shutil.move(img_path, dest)

    logger.info("Dataset organizado exitosamente:")
    for split_name in ["train", "validation", "test"]:
        real_count = len(list((DATA_DIR / split_name / "real").glob("*")))
        fake_count = len(list((DATA_DIR / split_name / "fake").glob("*")))
        logger.info(f"  {split_name}: {real_count} reales, {fake_count} falsas")


def run_pipeline(download: bool = True) -> None:
    """
    Ejecuta el pipeline completo de datos:
    1. Descarga el dataset (si se solicita)
    2. Organiza las imágenes en train/val/test
    """
    if download:
        source_dir = download_dataset()
        organize_dataset(source_dir)
    else:
        # Intentar usar dataset ya descargado
        potential_dirs = [
            DATA_DIR.parent / "datasets" / "140k-real-and-fake-faces",
            Path.home() / ".cache" / "kagglehub" / "datasets" / KAGGLE_DATASET,
        ]
        for d in potential_dirs:
            if d.exists():
                logger.info(f"Usando dataset existente en: {d}")
                organize_dataset(d)
                return

        logger.warning(
            "No se encontró dataset descargado. "
            "Use download=True o descargue manualmente desde Kaggle."
        )


if __name__ == "__main__":
    run_pipeline(download=True)
