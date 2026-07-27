# 🔍 DeepFake Detector

Sistema completo de detección de deepfakes basado en **DenseNet-121** con **Transfer Learning** y **Grad-CAM** para explicabilidad.

## 📋 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Dataset](#-dataset)
- [Modelo](#-modelo)
- [Evaluación](#-evaluación)
- [Explicabilidad (Grad-CAM)](#-explicabilidad-grad-cam)
- [Pruebas de Robustez](#-pruebas-de-robustez)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Resultados Esperados](#-resultados-esperados)

## 📖 Descripción General

Este proyecto implementa un sistema de detección de deepfakes completamente funcional que:

1. **Descarga y prepara** el dataset [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) de Kaggle
2. **Entrena** un modelo DenseNet-121 con transfer learning
3. **Evalúa** el modelo con métricas exhaustivas (Accuracy, Precision, Recall, F1, AUC)
4. **Explica** las predicciones usando Grad-CAM
5. **Despliega** una interfaz web interactiva con Streamlit
6. **Prueba la robustez** del modelo ante compresión JPEG

## 🏗️ Arquitectura del Sistema

```
                    ┌─────────────────┐
                    │   Dataset 140K   │
                    │ (Kaggle)         │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Data Pipeline  │
                    │ train/val/test  │
                    │   70/15/15%     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼──────┐ ┌────▼────┐ ┌──────▼──────┐
    │   DenseNet-121 │ │ Grad-   │ │   Pruebas   │
    │ + Transfer     │ │ CAM     │ │   Robustez  │
    │ Learning       │ │         │ │  (QF: 100,  │
    │                │ │         │ │   75, 50)   │
    └────────┬───────┘ └────┬────┘ └──────┬──────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                    ┌───────▼────────┐
                    │   Streamlit    │
                    │  Web App UI    │
                    └────────────────┘
```

## 💻 Requisitos

- **Python 3.8+**
- **PyTorch 2.0+** (con o sin CUDA)
- ~8 GB de espacio en disco (para el dataset)
- Conexión a internet (para descargar el dataset y pesos pre-entrenados)
- **Opcional:** GPU con CUDA para entrenamiento más rápido

## 🔧 Instalación

### 1. Clonar o copiar el proyecto

```bash
cd detectorIA
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar API de Kaggle (opcional, para descarga automática)

Si deseas que el pipeline descargue el dataset automáticamente:

1. Crea una cuenta en [Kaggle](https://www.kaggle.com)
2. Ve a **Settings → API** y crea un API token
3. Coloca el archivo `kaggle.json` descargado en `~/.kaggle/kaggle.json`

**Alternativa:** Descarga el dataset manualmente desde [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) y coloca las carpetas `real/` y `fake/` en `data/raw/`.

## 🚀 Uso

### Pipeline completo (recomendado)

```bash
python run_pipeline.py --mode all
```

Esto ejecutará:
1. Descarga y preparación del dataset
2. Entrenamiento del modelo
3. Evaluación en el conjunto de prueba
4. Pruebas de robustez

### Solo preparar datos

```bash
python run_pipeline.py --mode data --download
```

### Solo entrenar

```bash
python run_pipeline.py --mode train
```

### Solo evaluar

```bash
python run_pipeline.py --mode evaluate
```

### Solo pruebas de robustez

```bash
python run_pipeline.py --mode robustness
```

### Interfaz web

```bash
python run_pipeline.py --mode app
```

Luego abre tu navegador en: **http://localhost:8501**

## 📊 Dataset

El dataset **140K Real and Fake Faces** contiene:

| Clase | Imágenes | Descripción |
|-------|----------|-------------|
| **REAL** | 70,000 | Rostros reales del dataset FFHQ |
| **FAKE** | 70,000 | Rostros generados con StyleGAN |

### División

| Split    | Porcentaje | Imágenes     |
|----------|-----------|--------------|
| Train    | 70%       | ~98,000      |
| Val      | 15%       | ~21,000      |
| Test     | 15%       | ~21,000      |

### Preprocesamiento

- Redimensionamiento a **224×224** píxeles
- Normalización con media/std de **ImageNet**
- Aumento de datos: rotación (±15°), volteo horizontal, ajuste de brillo/contraste

## 🧠 Modelo

### Arquitectura

```
DenseNet-121 (pre-entrenado en ImageNet)
├── features.conv0 ──────────────── Congelado
├── features.denseblock1 ────────── Congelado
├── features.transition1 ────────── Congelado
├── features.denseblock2 ────────── Congelado
├── features.transition2 ────────── Congelado
├── features.denseblock3 ────────── Congelado
├── features.transition3 ────────── Congelado
├── features.denseblock4 ────────── Re-entrenable
├── features.norm5 ──────────────── Re-entrenable
└── classifier ──────────────────── Re-entrenable
    ├── Linear(1024 → 512) + ReLU + Dropout(0.3)
    └── Linear(512 → 2)
```

### Configuración de Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| Épocas | 30 (con early stopping) |
| Batch size | 32 |
| Optimizador | Adam |
| LR (últimas capas) | 1×10⁻⁴ |
| LR (fine-tuning) | 1×10⁻⁵ |
| Scheduler | ReduceLROnPlateau |
| Early stopping | Paciencia=7 |
| Función de pérdida | CrossEntropyLoss |

## 📈 Evaluación

El sistema genera automáticamente:

1. **Métricas principales:**
   - Accuracy, Precision, Recall, F1-Score, AUC

2. **Gráficos:**
   - Curvas de entrenamiento (Loss y Accuracy por época)
   - Matriz de confusión
   - Curva ROC

## 🔥 Explicabilidad (Grad-CAM)

Grad-CAM genera mapas de calor que muestran qué regiones de la imagen fueron más relevantes para la decisión del modelo:

- **Áreas en rojo/amarillo**: Alta influencia en la decisión
- **Áreas en azul/verde**: Baja influencia

### Ejemplo de explicación textual

> "El modelo clasificó la imagen como **FAKE** con un **94.3%** de confianza. Se detectaron artefactos visuales principalmente en la región de ojos y boca, que son característicos de imágenes generadas por modelos generativos adversarios (GANs)."

## 🧪 Pruebas de Robustez

El sistema evalúa el modelo con imágenes comprimidas a diferentes niveles de calidad JPEG:

| Factor de Calidad | Nivel de Compresión |
|-------------------|-------------------|
| QF = 100 | Sin compresión apreciable |
| QF = 75 | Compresión moderada (típica en web) |
| QF = 50 | Compresión agresiva |

**Resultado esperado:** La precisión del modelo debería mantenerse estable incluso con QF=50, demostrando robustez ante compresión JPEG.

## 📁 Estructura del Proyecto

```
detectorIA/
├── data/                     # Dataset (se genera automáticamente)
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── validation/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
├── app/
│   └── streamlit_app.py      # Interfaz web Streamlit
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuración central
│   ├── data_pipeline.py      # Pipeline de datos
│   ├── model.py              # Definición del modelo DenseNet-121
│   ├── train.py              # Pipeline de entrenamiento
│   ├── evaluate.py           # Evaluación y métricas
│   ├── gradcam.py            # Implementación Grad-CAM
│   ├── robustness.py         # Pruebas de robustez
│   └── utils.py              # Funciones auxiliares
├── models/                   # Modelos guardados
├── outputs/
│   ├── plots/                # Gráficos generados
│   └── checkpoints/          # Checkpoints del modelo
├── requirements.txt
├── run_pipeline.py           # Script principal
└── README.md
```

## 📊 Resultados Esperados

Después del entrenamiento completo, puedes esperar:

| Métrica | Valor Esperado |
|---------|---------------|
| Accuracy | ≥ 95% |
| Precision | ≥ 95% |
| Recall | ≥ 95% |
| F1-Score | ≥ 95% |
| AUC | ≥ 0.99 |

### Ejemplo de salida de entrenamiento:

```
[2024-01-01 12:00:00] INFO - Época 15/30
[2024-01-01 12:00:05] INFO -   Train Loss: 0.0234 | Train Acc: 0.9921  |  Val Loss: 0.0456 | Val Acc: 0.9878  |  LR: 1.00e-04
[2024-01-01 12:00:05] INFO -   ✓ Nuevo mejor modelo guardado (val_acc=0.9878)
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un PR para discutir cambios mayores.

## 📄 Licencia

Este proyecto es educativo y está destinado a la investigación en detección de deepfakes.
