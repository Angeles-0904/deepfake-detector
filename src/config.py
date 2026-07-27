"""
Configuración central del sistema de detección de deepfakes.
Todas las rutas, hiperparámetros y constantes se definen aquí.
"""

import os
from pathlib import Path

# ─── Rutas del proyecto ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

# Crear directorios si no existen
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, CHECKPOINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME = "140k-real-and-fake-faces"
KAGGLE_DATASET = "xhlulu/140k-real-and-fake-faces"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ─── Preprocesamiento ─────────────────────────────────────────────────────────
IMG_SIZE = 224  # Tamaño de entrada para DenseNet-121
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
NUM_WORKERS = 4  # Ajustar según la CPU disponible

# ─── Aumento de datos ─────────────────────────────────────────────────────────
AUGMENTATION_CONFIG = {
    "rotation_range": 15,       # grados
    "horizontal_flip": True,
    "brightness_range": 0.1,
    "zoom_range": 0.1,
}

# ─── Modelo ───────────────────────────────────────────────────────────────────
NUM_CLASSES = 2
CLASS_NAMES = ["FAKE", "REAL"]
DENSENET_FEATURES = 1024

# ─── Estrategia de Transfer Learning ──────────────────────────────────────────
# Congelar hasta el bloque 3 de DenseNet-121, reentrenar bloque 4 y clasificador
FREEZE_UNTIL_BLOCK = 3  # 0=congelar nada, 4=congelar todo

# ─── Entrenamiento ────────────────────────────────────────────────────────────
EPOCHS = 30
LEARNING_RATE_LAST_LAYERS = 1e-4
LEARNING_RATE_FINETUNE = 1e-5
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# ─── Scheduler ────────────────────────────────────────────────────────────────
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 3
SCHEDULER_MIN_LR = 1e-7

# ─── Early Stopping ──────────────────────────────────────────────────────────
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 1e-4

# ─── Checkpoints ─────────────────────────────────────────────────────────────
CHECKPOINT_BEST = CHECKPOINTS_DIR / "best_model.pth"
CHECKPOINT_LAST = CHECKPOINTS_DIR / "last_model.pth"

# ─── Evaluación ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5

# ─── Robustez ─────────────────────────────────────────────────────────────────
QUALITY_FACTORS = [100, 75, 50]

# ─── Streamlit ───────────────────────────────────────────────────────────────
MAX_HISTORY = 5

# ─── Dispositivo ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"
# Alternativamente, detectar automáticamente:
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
