"""
Aplicación web Streamlit para detección de deepfakes.
Permite al usuario:
- Subir una imagen (JPEG/PNG)
- Obtener predicción REAL/FAKE con confianza
- Visualizar mapa de calor Grad-CAM
- Leer explicación textual
- Probar robustez con compresión JPEG
- Ver historial de predicciones
"""

import io
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

# Asegurar que el directorio raíz esté en el path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    CLASS_NAMES,
    DEVICE,
    IMG_SIZE,
    QUALITY_FACTORS,
    MAX_HISTORY,
    CHECKPOINT_BEST,
    CHECKPOINT_LAST,
    PLOTS_DIR,
)
from src.gradcam import GradCAM
from src.model import build_densenet121
from src.utils import (
    preprocess_image,
    image_to_base64,
    base64_to_image,
    logger,
)


# ─── Configuración de página ─────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Cargar modelo ───────────────────────────────────────────────────────────

@st.cache_resource
def load_model() -> Tuple[nn.Module, GradCAM]:
    """
    Carga el modelo y prepara Grad-CAM (cacheado para no recargar en cada interacción).
    """
    model = build_densenet121(freeze_until_block=3)

    # Buscar checkpoint
    checkpoint_path = CHECKPOINT_BEST
    if not checkpoint_path.exists():
        checkpoint_path = CHECKPOINT_LAST

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Modelo cargado desde: {checkpoint_path}")
        st.sidebar.success(f"✅ Modelo cargado: {checkpoint_path.name}")
    else:
        logger.warning("No se encontró checkpoint. Usando pesos pre-entrenados sin fine-tuning.")
        st.sidebar.warning(
            "⚠️ No se encontró modelo entrenado. "
            "Usando pesos pre-entrenados de ImageNet. "
            "Los resultados serán subóptimos."
        )

    model.to(DEVICE)
    model.eval()

    # Inicializar Grad-CAM
    gradcam = GradCAM(model)

    return model, gradcam


# ─── Funciones de procesamiento ──────────────────────────────────────────────

def predict_image(
    model: nn.Module,
    image: Image.Image,
) -> Tuple[int, float, torch.Tensor]:
    """
    Ejecuta la predicción del modelo sobre una imagen.

    Returns:
        class_idx: 0=FAKE, 1=REAL
        confidence: Probabilidad de la clase predicha (0-1)
        logits: Salida del modelo
    """
    input_tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    return predicted.item(), confidence.item(), outputs


def compress_image(image: Image.Image, quality: int) -> Image.Image:
    """
    Comprime una imagen JPEG con un factor de calidad dado.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def get_compression_info(quality: int) -> str:
    """Retorna una descripción del nivel de compresión."""
    if quality >= 95:
        return "Sin compresión apreciable"
    elif quality >= 75:
        return "Compresión moderada"
    elif quality >= 50:
        return "Compresión agresiva"
    else:
        return "Compresión muy agresiva"


# ─── Inicializar estado de sesión ────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

if "robustness_results" not in st.session_state:
    st.session_state.robustness_results = None

if "current_result" not in st.session_state:
    st.session_state.current_result = None


# ─── Interfaz de usuario ─────────────────────────────────────────────────────

def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("🔍 DeepFake Detector")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "### ¿Cómo funciona?\n"
        "1. Sube una imagen de rostro\n"
        "2. Haz clic en **Analizar**\n"
        "3. Obtén la predicción y el mapa de calor\n\n"
        "El modelo utiliza **DenseNet-121** con "
        "**Grad-CAM** para explicar sus decisiones."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Dispositivo:** {DEVICE.upper()}")
    st.sidebar.markdown(f"**Tamaño de entrada:** {IMG_SIZE}×{IMG_SIZE}")

    # Mostrar imágenes de ejemplo en sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📸 Prueba rápida")

    # ── Cargar modelo ────────────────────────────────────────────────────────
    with st.spinner("Cargando modelo..."):
        try:
            model, gradcam = load_model()
            model_ready = True
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
            model_ready = False
            gradcam = None

    # ── Área principal ────────────────────────────────────────────────────────
    st.title("🔍 Detector de DeepFakes")
    st.markdown(
        "Sube una imagen de un rostro para determinar si es **real** o "
        "**generada por IA**. El sistema te mostrará un mapa de calor "
        "con las regiones que influyeron en la decisión."
    )

    # ── Subida de imagen ─────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Selecciona una imagen",
            type=["jpg", "jpeg", "png"],
            help="Formatos soportados: JPG, JPEG, PNG",
        )

    # ── Botón de analizar ────────────────────────────────────────────────────
    analyze_button = st.button(
        "🚀 Analizar imagen",
        type="primary",
        disabled=not (model_ready and uploaded_file is not None),
        use_container_width=True,
    )

    # ── Procesar si hay imagen y se hizo clic ────────────────────────────────
    if uploaded_file is not None:
        # Mostrar la imagen subida
        image = Image.open(uploaded_file).convert("RGB")

        with col2:
            st.image(image, caption="Imagen subida", width=250)

        if analyze_button:
            with st.spinner("Analizando imagen..."):
                try:
                    # Predicción
                    class_idx, confidence, logits = predict_image(model, image)
                    class_name = CLASS_NAMES[class_idx]

                    # Grad-CAM
                    heatmap_raw, heatmap_resized = gradcam.generate(image, class_idx)
                    overlay = gradcam.overlay_heatmap(image, heatmap_resized)

                    # Explicación textual
                    explanation = gradcam.generate_explanation(
                        class_idx, confidence, heatmap_resized
                    )

                    # Resultado como base64 para el historial
                    overlay_b64 = image_to_base64(overlay)
                    image_b64 = image_to_base64(image)

                    # Almacenar resultado actual
                    st.session_state.current_result = {
                        "class_idx": class_idx,
                        "class_name": class_name,
                        "confidence": confidence,
                        "image": image,
                        "image_b64": image_b64,
                        "overlay": overlay,
                        "overlay_b64": overlay_b64,
                        "heatmap_raw": heatmap_raw,
                        "heatmap_resized": heatmap_resized,
                        "explanation": explanation,
                        "logits": logits,
                    }

                    # Agregar al historial
                    st.session_state.history.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "image_b64": image_b64,
                        "overlay_b64": overlay_b64,
                    })
                    # Mantener solo las últimas MAX_HISTORY
                    if len(st.session_state.history) > MAX_HISTORY:
                        st.session_state.history = st.session_state.history[-MAX_HISTORY:]

                except Exception as e:
                    st.error(f"Error durante el análisis: {e}")
                    logger.error(f"Error en análisis: {e}", exc_info=True)

    # ── Mostrar resultados ───────────────────────────────────────────────────
    if st.session_state.current_result is not None:
        result = st.session_state.current_result
        class_idx = result["class_idx"]
        class_name = result["class_name"]
        confidence = result["confidence"]
        overlay = result["overlay"]
        heatmap_raw = result["heatmap_raw"]
        explanation = result["explanation"]

        # Indicador visual
        is_fake = class_idx == 0
        color = "#ff4444" if is_fake else "#44bb44"
        emoji = "❌" if is_fake else "✅"

        st.markdown("---")
        st.markdown("## 📊 Resultados")

        # Métrica principal en grande
        col_result1, col_result2, col_result3 = st.columns(3)

        with col_result1:
            st.markdown(
                f"<div style='text-align: center; padding: 20px; "
                f"background-color: {color}22; border-radius: 10px; "
                f"border: 2px solid {color};'>"
                f"<h2 style='color: {color}; margin: 0;'>{emoji} {class_name}</h2>"
                f"<p style='font-size: 14px; margin: 5px 0 0 0;'>Predicción</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_result2:
            confidence_pct = confidence * 100
            bar_color = "#ff4444" if confidence_pct < 70 else "#ffaa00" if confidence_pct < 90 else "#44bb44"
            st.markdown(
                f"<div style='text-align: center; padding: 20px; "
                f"background-color: #f0f2f6; border-radius: 10px;'>"
                f"<h2 style='color: {bar_color}; margin: 0;'>{confidence_pct:.1f}%</h2>"
                f"<p style='font-size: 14px; margin: 5px 0 0 0;'>Confianza</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_result3:
            st.markdown(
                f"<div style='text-align: center; padding: 20px; "
                f"background-color: #f0f2f6; border-radius: 10px;'>"
                f"<h2 style='color: #333; margin: 0;'>{'FAKE' if is_fake else 'REAL'}</h2>"
                f"<p style='font-size: 14px; margin: 5px 0 0 0;'>"
                f"Score: {torch.softmax(result['logits'], dim=1)[0][1].item():.4f}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Mapa de calor y explicación
        st.markdown("---")
        col_heat, col_expl = st.columns(2)

        with col_heat:
            st.markdown("### 🔥 Mapa de Calor (Grad-CAM)")
            st.image(overlay, caption="Regiones que influyeron en la decisión", width=400)

            # Mostrar barra de colores del heatmap
            st.markdown(
                "<div style='display: flex; align-items: center; gap: 10px; "
                "justify-content: center; font-size: 12px; color: #666;'>"
                "<span>Menor influencia</span>"
                "<div style='width: 200px; height: 15px; "
                "background: linear-gradient(to right, blue, cyan, green, yellow, red); "
                "border-radius: 3px;'></div>"
                "<span>Mayor influencia</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        with col_expl:
            st.markdown("### 📝 Explicación")
            st.info(explanation)

            # Mostrar heatmap crudo
            st.markdown("### 🌡️ Mapa de Activación")
            heatmap_display = (heatmap_raw * 255).astype(np.uint8)
            heatmap_pil = Image.fromarray(heatmap_display, mode="L").resize((224, 224))
            st.image(heatmap_pil, caption="Activación pura (sin superposición)", width=200)

        # ── Pruebas de robustez ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 🧪 Pruebas de Robustez")
        st.markdown(
            "Evalúa cómo afecta la compresión JPEG a la predicción del modelo."
        )

        col_rob1, col_rob2, col_rob3 = st.columns(3)

        with col_rob1:
            test_qf_100 = st.button("📷 Probar QF=100 (Máxima calidad)", use_container_width=True)
        with col_rob2:
            test_qf_75 = st.button("📷 Probar QF=75 (Calidad media)", use_container_width=True)
        with col_rob3:
            test_qf_50 = st.button("📷 Probar QF=50 (Calidad baja)", use_container_width=True)

        robustness_results = []

        for qf, btn in [(100, test_qf_100), (75, test_qf_75), (50, test_qf_50)]:
            if btn:
                compressed = compress_image(result["image"], qf)
                c_idx, c_conf, _ = predict_image(model, compressed)
                c_name = CLASS_NAMES[c_idx]
                c_conf_pct = c_conf * 100

                robustness_results.append({
                    "qf": qf,
                    "class_name": c_name,
                    "confidence": c_conf,
                    "image": compressed,
                })

                st.markdown(
                    f"**QF={qf}** ({get_compression_info(qf)}): "
                    f"→ **{c_name}** con **{c_conf_pct:.1f}%** de confianza"
                )

        if robustness_results:
            st.session_state.robustness_results = robustness_results

            # Mostrar comparativa visual
            st.markdown("### 📊 Comparativa de compresión")
            cols = st.columns(len(robustness_results))
            for i, result in enumerate(robustness_results):
                with cols[i]:
                    st.image(
                        result["image"],
                        caption=f"QF={result['qf']}",
                        width=150,
                    )
                    c_name = result["class_name"]
                    c_conf = result["confidence"] * 100
                    emoji_rob = "❌" if c_name == "FAKE" else "✅"
                    st.markdown(
                        f"<p style='text-align: center; font-weight: bold; "
                        f"color: {'#ff4444' if c_name == 'FAKE' else '#44bb44'};'>"
                        f"{emoji_rob} {c_name}<br>{c_conf:.1f}%</p>",
                        unsafe_allow_html=True,
                    )

    # ── Historial ────────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("---")
        st.markdown("## 📋 Historial de predicciones")

        cols = st.columns(min(len(st.session_state.history), MAX_HISTORY))
        for i, entry in enumerate(
            reversed(st.session_state.history[:MAX_HISTORY])
        ):
            with cols[i]:
                # Decodificar imagen del historial
                hist_img = base64_to_image(entry["image_b64"])
                st.image(
                    hist_img,
                    caption=f"{'❌' if entry['class_name'] == 'FAKE' else '✅'} "
                            f"{entry['class_name']} "
                            f"({entry['confidence'] * 100:.1f}%)",
                    width=120,
                )

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 12px;'>"
        "DeepFake Detector v1.0 | Modelo: DenseNet-121 | "
        "Dataset: 140K Real and Fake Faces (Kaggle)"
        "</div>",
        unsafe_allow_html=True,
    )

    # Si no hay modelo entrenado, mostrar advertencia más prominente
    if not model_ready:
        st.error(
            "❌ **No se pudo cargar el modelo.**\n\n"
            "Asegúrate de haber entrenado el modelo primero ejecutando:\n"
            "```bash\npython -m src.train\n```"
        )


if __name__ == "__main__":
    main()
