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


# ─── CSS Personalizado ───────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* ── Variables globales ── */
    :root {
        --primary: #6C5CE7;
        --primary-dark: #5A4BD1;
        --secondary: #00CEC9;
        --accent: #FD79A8;
        --danger: #FF6B6B;
        --success: #00B894;
        --warning: #FDCB6E;
        --dark: #2D3436;
        --gray: #636E72;
        --light: #DFE6E9;
        --bg-gradient: linear-gradient(135deg, #0c0c1d 0%, #1a1a3e 50%, #0c0c1d 100%);
        --card-bg: rgba(255, 255, 255, 0.05);
        --card-border: rgba(255, 255, 255, 0.1);
        --glass: rgba(255, 255, 255, 0.08);
        --glass-border: rgba(255, 255, 255, 0.15);
    }

    /* ── Fondo general ── */
    .stApp {
        background: var(--bg-gradient);
        background-attachment: fixed;
    }

    /* ── Ocultar elementos por defecto de Streamlit ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ── Header / Hero ── */
    .hero-section {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        background: linear-gradient(135deg, rgba(108,92,231,0.15) 0%, rgba(0,206,201,0.1) 100%);
        border-radius: 20px;
        border: 1px solid rgba(108,92,231,0.2);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(108,92,231,0.08) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(0,206,201,0.08) 0%, transparent 50%);
        animation: heroGlow 8s ease-in-out infinite alternate;
    }
    @keyframes heroGlow {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-5%, -5%) rotate(3deg); }
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C5CE7, #00CEC9, #FD79A8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
        letter-spacing: -1px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.7);
        max-width: 600px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(108,92,231,0.2);
        border: 1px solid rgba(108,92,231,0.3);
        color: #A29BFE;
        padding: 0.3rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }

    /* ── Tarjetas (Glassmorphism) ── */
    .glass-card {
        background: var(--glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(108,92,231,0.3);
        box-shadow: 0 8px 32px rgba(108,92,231,0.15);
    }
    .glass-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
        transition: left 0.5s ease;
    }
    .glass-card:hover::after {
        left: 100%;
    }

    /* ── Resultado principal ── */
    .result-fake, .result-real {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: resultPop 0.5s ease-out;
    }
    @keyframes resultPop {
        0% { transform: scale(0.9); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    .result-fake {
        background: linear-gradient(135deg, rgba(255,107,107,0.2), rgba(214,48,49,0.1));
        border: 1px solid rgba(255,107,107,0.3);
    }
    .result-real {
        background: linear-gradient(135deg, rgba(0,184,148,0.2), rgba(85,239,196,0.1));
        border: 1px solid rgba(0,184,148,0.3);
    }
    .result-emoji {
        font-size: 4rem;
        display: block;
        margin-bottom: 0.5rem;
        animation: emojiBounce 0.6s ease-out;
    }
    @keyframes emojiBounce {
        0% { transform: scale(0); }
        50% { transform: scale(1.2); }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); }
    }
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 2px;
    }
    .result-label.fake { color: #FF6B6B; }
    .result-label.real { color: #00B894; }
    .result-subtext {
        color: rgba(255,255,255,0.6);
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    /* ── Medidor de confianza circular ── */
    .confidence-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1rem;
    }
    .confidence-ring {
        position: relative;
        width: 140px;
        height: 140px;
        margin: 0 auto 1rem;
    }
    .confidence-ring svg {
        transform: rotate(-90deg);
    }
    .confidence-ring .bg {
        fill: none;
        stroke: rgba(255,255,255,0.1);
        stroke-width: 8;
    }
    .confidence-ring .progress {
        fill: none;
        stroke-width: 8;
        stroke-linecap: round;
        transition: stroke-dashoffset 1s ease-out;
    }
    .confidence-value {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
    }
    .confidence-label {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.3rem;
    }

    /* ── Score ── */
    .score-card {
        text-align: center;
        padding: 1rem;
    }
    .score-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: rgba(255,255,255,0.9);
    }
    .score-label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* ── Botones personalizados ── */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6C5CE7, #a29bfe);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(108,92,231,0.3);
        position: relative;
        overflow: hidden;
    }
    div.stButton > button:first-child::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s ease;
    }
    div.stButton > button:first-child:hover::before {
        left: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108,92,231,0.4);
    }
    div.stButton > button:first-child:active {
        transform: translateY(0);
    }
    div.stButton > button:first-child:disabled {
        background: rgba(255,255,255,0.1);
        color: rgba(255,255,255,0.3);
        box-shadow: none;
    }

    /* ── Botón de robustez ── */
    div.stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.15);
        color: rgba(255,255,255,0.8);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: rgba(108,92,231,0.2);
        border-color: rgba(108,92,231,0.4);
        color: white;
    }

    /* ── File uploader ── */
    .stFileUploader {
        border: 2px dashed rgba(108,92,231,0.3);
        border-radius: 16px;
        padding: 1rem;
        background: rgba(108,92,231,0.05);
        transition: all 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: rgba(108,92,231,0.6);
        background: rgba(108,92,231,0.1);
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        border: none !important;
        padding: 1rem;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] div {
        color: rgba(255,255,255,0.6) !important;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] svg {
        fill: rgba(108,92,231,0.6) !important;
    }

    /* ── Sidebar ── */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(12, 12, 29, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    .css-1d391kg .stMarkdown, [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255,255,255,0.8);
    }

    /* ── Títulos ── */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: rgba(255,255,255,0.95);
        margin: 1.5rem 0 1rem;
        position: relative;
        padding-left: 1rem;
    }
    .section-title::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 3px;
        background: linear-gradient(135deg, #6C5CE7, #00CEC9);
        border-radius: 3px;
    }

    /* ── Explicación ── */
    .explanation-box {
        background: rgba(108,92,231,0.1);
        border: 1px solid rgba(108,92,231,0.2);
        border-radius: 12px;
        padding: 1.2rem;
        color: rgba(255,255,255,0.85);
        line-height: 1.7;
        font-size: 0.95rem;
    }
    .explanation-box strong {
        color: #A29BFE;
    }

    /* ── Heatmap legend ── */
    .heatmap-legend {
        display: flex;
        align-items: center;
        gap: 12px;
        justify-content: center;
        margin-top: 0.5rem;
    }
    .heatmap-legend span {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.5);
    }
    .heatmap-bar {
        width: 180px;
        height: 12px;
        border-radius: 6px;
        background: linear-gradient(to right, #0984E3, #00CEC9, #00B894, #FDCB6E, #FF6B6B);
    }

    /* ── Historial ── */
    .history-item {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .history-item:hover {
        transform: translateY(-2px);
        border-color: rgba(108,92,231,0.3);
    }
    .history-label {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: rgba(255,255,255,0.3);
        font-size: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 2rem;
    }
    .footer strong {
        color: rgba(255,255,255,0.5);
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-color: #6C5CE7 !important;
        border-top-color: transparent !important;
    }

    /* ── Alertas ── */
    .stAlert {
        background: rgba(108,92,231,0.1) !important;
        border: 1px solid rgba(108,92,231,0.2) !important;
        border-radius: 12px !important;
        color: rgba(255,255,255,0.85) !important;
    }
    .stAlert [data-testid="baseButton-secondary"] {
        color: #A29BFE !important;
    }

    /* ── Error ── */
    .stError {
        background: rgba(255,107,107,0.15) !important;
        border: 1px solid rgba(255,107,107,0.3) !important;
        border-radius: 12px !important;
    }

    /* ── Success ── */
    .stSuccess {
        background: rgba(0,184,148,0.15) !important;
        border: 1px solid rgba(0,184,148,0.3) !important;
        border-radius: 12px !important;
    }

    /* ── Warning ── */
    .stWarning {
        background: rgba(253,203,110,0.15) !important;
        border: 1px solid rgba(253,203,110,0.3) !important;
        border-radius: 12px !important;
    }

    /* ── Info ── */
    .stInfo {
        background: rgba(108,92,231,0.1) !important;
        border: 1px solid rgba(108,92,231,0.2) !important;
        border-radius: 12px !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1rem;
        color: rgba(255,255,255,0.6);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(108,92,231,0.3) !important;
        color: white !important;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .hero-section { padding: 1.5rem 1rem; }
        .result-emoji { font-size: 3rem; }
        .result-label { font-size: 1.5rem; }
        .confidence-ring { width: 100px; height: 100px; }
        .confidence-value { font-size: 1.5rem; }
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(108,92,231,0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(108,92,231,0.5);
    }

    /* ── Stats row ── */
    .stat-item {
        text-align: center;
        padding: 0.5rem;
    }
    .stat-value {
        font-size: 1.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C5CE7, #00CEC9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── Spinning ring for loading ── */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .loading-ring {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid rgba(108,92,231,0.3);
        border-top: 2px solid #6C5CE7;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin-right: 8px;
        vertical-align: middle;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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


def render_confidence_gauge(confidence_pct: float, color: str) -> str:
    """Renderiza un medidor circular SVG de confianza."""
    radius = 54
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - confidence_pct / 100)
    return f"""
    <div class="confidence-ring">
        <svg width="140" height="140" viewBox="0 0 140 140">
            <circle class="bg" cx="70" cy="70" r="{radius}"/>
            <circle class="progress" cx="70" cy="70" r="{radius}"
                    stroke="{color}" stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}"/>
        </svg>
        <div class="confidence-value" style="color: {color}">{confidence_pct:.1f}%</div>
    </div>
    """


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
    with st.sidebar:
        st.markdown(
            "<div style='text-align: center; padding: 1rem 0;'>"
            "<span style='font-size: 2.5rem;'>🔍</span>"
            "<h2 style='margin: 0.5rem 0 0; font-weight: 700; "
            "background: linear-gradient(135deg, #6C5CE7, #00CEC9); "
            "-webkit-background-clip: text; -webkit-text-fill-color: transparent; "
            "background-clip: text;'>DeepFake Detector</h2>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown(
            "<div style='padding: 0.5rem 0;'>"
            "<h4 style='color: rgba(255,255,255,0.9); margin-bottom: 0.8rem;'>📖 ¿Cómo funciona?</h4>"
            "<ol style='color: rgba(255,255,255,0.7); font-size: 0.9rem; line-height: 1.8; padding-left: 1.2rem;'>"
            "<li>Sube una imagen de rostro</li>"
            "<li>Haz clic en <strong>Analizar</strong></li>"
            "<li>Obtén la predicción y el mapa de calor</li>"
            "</ol>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='padding: 0.5rem 0; font-size: 0.85rem; color: rgba(255,255,255,0.6);'>"
            "El modelo utiliza <strong style='color: #A29BFE;'>DenseNet-121</strong> con "
            "<strong style='color: #A29BFE;'>Grad-CAM</strong> para explicar sus decisiones."
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Stats en sidebar
        st.markdown(
            "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 0.5rem 0;'>"
            "<div class='stat-item'><div class='stat-value'>" + DEVICE.upper() + "</div>"
            "<div class='stat-label'>Dispositivo</div></div>"
            "<div class='stat-item'><div class='stat-value'>" + str(IMG_SIZE) + "×" + str(IMG_SIZE) + "</div>"
            "<div class='stat-label'>Entrada</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Cargar modelo ────────────────────────────────────────────────────────
    with st.spinner(""):
        placeholder = st.empty()
        placeholder.markdown(
            "<div style='text-align: center; padding: 3rem;'>"
            "<div class='loading-ring'></div>"
            "<span style='color: rgba(255,255,255,0.6);'>Cargando modelo...</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        try:
            model, gradcam = load_model()
            model_ready = True
            placeholder.empty()
        except Exception as e:
            placeholder.empty()
            st.error(f"Error al cargar el modelo: {e}")
            model_ready = False
            gradcam = None

    # ── Hero Section ─────────────────────────────────────────────────────────
    st.markdown(
        "<div class='hero-section'>"
        "<div class='hero-badge'>🧠 Inteligencia Artificial · Visión Computacional</div>"
        "<h1 class='hero-title'>Detector de DeepFakes</h1>"
        "<p class='hero-subtitle'>"
        "Sube una imagen de un rostro y determina si es "
        "<strong>real</strong> o <strong>generada por IA</strong> "
        "con mapas de calor explicativos."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Subida de imagen ─────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📁 Arrastra una imagen aquí",
        type=["jpg", "jpeg", "png"],
        help="Formatos soportados: JPG, JPEG, PNG",
    )

    col_btn, col_preview = st.columns([3, 2])

    with col_btn:
        analyze_button = st.button(
            "🚀 Analizar imagen",
            type="primary",
            disabled=not (model_ready and uploaded_file is not None),
            use_container_width=True,
        )

    # ── Procesar si hay imagen y se hizo clic ────────────────────────────────
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with col_preview:
            st.markdown(
                "<div style='text-align: center;'>"
                "<p style='color: rgba(255,255,255,0.5); font-size: 0.8rem; "
                "text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;'>Previsualización</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.image(image, width=200)

        if analyze_button:
            with st.spinner(""):
                analysis_placeholder = st.empty()
                analysis_placeholder.markdown(
                    "<div style='text-align: center; padding: 2rem;'>"
                    "<div class='loading-ring' style='width: 40px; height: 40px; border-width: 3px;'></div>"
                    "<p style='color: rgba(255,255,255,0.6); margin-top: 1rem; font-size: 1.1rem;'>"
                    "Analizando imagen...</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
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

                    analysis_placeholder.empty()

                except Exception as e:
                    analysis_placeholder.empty()
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
        color = "#FF6B6B" if is_fake else "#00B894"
        emoji = "❌" if is_fake else "✅"
        label_class = "fake" if is_fake else "real"

        st.markdown("<div class='section-title'>📊 Resultados</div>", unsafe_allow_html=True)

        # ── Cards de resultados ──────────────────────────────────────────────
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.markdown(
                f"<div class='result-{label_class}'>"
                f"<span class='result-emoji'>{emoji}</span>"
                f"<h2 class='result-label {label_class}'>{class_name}</h2>"
                f"<p class='result-subtext'>Predicción del modelo</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_r2:
            confidence_pct = confidence * 100
            gauge_color = "#FF6B6B" if confidence_pct < 70 else "#FDCB6E" if confidence_pct < 90 else "#00B894"
            st.markdown(
                "<div class='glass-card confidence-container'>"
                + render_confidence_gauge(confidence_pct, gauge_color)
                + "<div class='confidence-label'>Confianza</div>"
                + "</div>",
                unsafe_allow_html=True,
            )

        with col_r3:
            score = torch.softmax(result['logits'], dim=1)[0][1].item()
            st.markdown(
                f"<div class='glass-card score-card'>"
                f"<div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>{'🤖' if is_fake else '👤'}</div>"
                f"<div class='score-value'>{score:.4f}</div>"
                f"<div class='score-label'>Score REAL</div>"
                f"<div style='margin-top: 1rem; padding: 0.5rem; background: rgba(255,255,255,0.05); "
                f"border-radius: 8px;'>"
                f"<div style='font-size: 0.75rem; color: rgba(255,255,255,0.5);'>"
                f"0.0000 ·················· 1.0000</div>"
                f"<div style='width: 100%; height: 4px; background: rgba(255,255,255,0.1); "
                f"border-radius: 2px; margin-top: 4px;'>"
                f"<div style='width: {score * 100}%; height: 100%; "
                f"background: linear-gradient(90deg, #FF6B6B, #FDCB6E, #00B894); "
                f"border-radius: 2px; transition: width 1s ease-out;'></div>"
                f"</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Mapa de calor y explicación ──────────────────────────────────────
        st.markdown("<div class='section-title'>🔥 Análisis del Modelo</div>", unsafe_allow_html=True)

        col_heat, col_expl = st.columns(2)

        with col_heat:
            st.markdown(
                "<div class='glass-card'>"
                "<h4 style='color: rgba(255,255,255,0.9); margin: 0 0 1rem;'>"
                "🌡️ Mapa de Calor Grad-CAM</h4>",
                unsafe_allow_html=True,
            )
            st.image(overlay, caption="Regiones que influyeron en la decisión", use_container_width=True)

            # Leyenda
            st.markdown(
                "<div class='heatmap-legend'>"
                "<span>Menor influencia</span>"
                "<div class='heatmap-bar'></div>"
                "<span>Mayor influencia</span>"
                "</div>",
                unsafe_allow_html=True,
            )

            # Heatmap crudo
            st.markdown(
                "<h4 style='color: rgba(255,255,255,0.7); margin: 1.5rem 0 0.5rem; font-size: 0.9rem;'>"
                "🌡️ Activación pura</h4>",
                unsafe_allow_html=True,
            )
            heatmap_display = (heatmap_raw * 255).astype(np.uint8)
            heatmap_pil = Image.fromarray(heatmap_display, mode="L").resize((224, 224))
            st.image(heatmap_pil, width=150)

            st.markdown("</div>", unsafe_allow_html=True)

        with col_expl:
            st.markdown(
                "<div class='glass-card'>"
                "<h4 style='color: rgba(255,255,255,0.9); margin: 0 0 1rem;'>"
                "📝 Explicación</h4>"
                f"<div class='explanation-box'>{explanation}</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        # ── Pruebas de robustez ──────────────────────────────────────────────
        st.markdown("<div class='section-title'>🧪 Pruebas de Robustez</div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-bottom: 1rem;'>"
            "Evalúa cómo afecta la compresión JPEG a la predicción del modelo.</p>",
            unsafe_allow_html=True,
        )

        col_rob1, col_rob2, col_rob3 = st.columns(3)

        with col_rob1:
            test_qf_100 = st.button("📷 QF=100 · Máxima calidad", use_container_width=True, key="qf100")
        with col_rob2:
            test_qf_75 = st.button("📷 QF=75 · Calidad media", use_container_width=True, key="qf75")
        with col_rob3:
            test_qf_50 = st.button("📷 QF=50 · Calidad baja", use_container_width=True, key="qf50")

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

                emoji_rob = "❌" if c_name == "FAKE" else "✅"
                st.markdown(
                    f"<div style='background: rgba(108,92,231,0.1); border: 1px solid rgba(108,92,231,0.2); "
                    f"border-radius: 10px; padding: 0.8rem 1rem; margin: 0.5rem 0; "
                    f"color: rgba(255,255,255,0.8);'>"
                    f"<strong>QF={qf}</strong> ({get_compression_info(qf)}): "
                    f"{emoji_rob} → <strong>{c_name}</strong> con "
                    f"<strong>{c_conf_pct:.1f}%</strong> de confianza"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if robustness_results:
            st.session_state.robustness_results = robustness_results

            st.markdown(
                "<h4 style='color: rgba(255,255,255,0.8); margin: 1.5rem 0 1rem;'>"
                "📊 Comparativa visual</h4>",
                unsafe_allow_html=True,
            )
            cols = st.columns(len(robustness_results))
            for i, rob_result in enumerate(robustness_results):
                with cols[i]:
                    st.markdown(
                        "<div class='glass-card' style='padding: 1rem; text-align: center;'>",
                        unsafe_allow_html=True,
                    )
                    st.image(
                        rob_result["image"],
                        caption=f"QF={rob_result['qf']}",
                        use_container_width=True,
                    )
                    c_name = rob_result["class_name"]
                    c_conf = rob_result["confidence"] * 100
                    emoji_rob = "❌" if c_name == "FAKE" else "✅"
                    rob_color = "#FF6B6B" if c_name == "FAKE" else "#00B894"
                    st.markdown(
                        f"<p style='font-weight: 700; color: {rob_color}; margin: 0.5rem 0 0; "
                        f"font-size: 1.1rem;'>{emoji_rob} {c_name}</p>"
                        f"<p style='color: rgba(255,255,255,0.6); font-size: 0.9rem; "
                        f"margin: 0;'>{c_conf:.1f}%</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

    # ── Historial ────────────────────────────────────────────────────────────
    if st.session_state.history:
        st.markdown("<div class='section-title'>📋 Historial de predicciones</div>", unsafe_allow_html=True)

        cols = st.columns(min(len(st.session_state.history), MAX_HISTORY))
        for i, entry in enumerate(reversed(st.session_state.history[:MAX_HISTORY])):
            with cols[i]:
                hist_img = base64_to_image(entry["image_b64"])
                is_fake_hist = entry["class_name"] == "FAKE"
                hist_color = "#FF6B6B" if is_fake_hist else "#00B894"
                hist_emoji = "❌" if is_fake_hist else "✅"
                st.markdown(
                    "<div class='glass-card history-item'>",
                    unsafe_allow_html=True,
                )
                st.image(hist_img, width=100)
                st.markdown(
                    f"<p class='history-label' style='color: {hist_color};'>"
                    f"{hist_emoji} {entry['class_name']}<br>"
                    f"<span style='font-size: 0.75rem; color: rgba(255,255,255,0.5);'>"
                    f"{entry['confidence'] * 100:.1f}%</span></p>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='footer'>"
        "🔍 <strong>DeepFake Detector</strong> · "
        "Modelo: <strong>DenseNet-121</strong> · "
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
