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
    page_icon="\U0001F50D",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── CSS Personalizado ───────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,800;1,400&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg: #0F0D0A;
        --bg-card: #1A1612;
        --bg-card-hover: #221E18;
        --border: #2A2520;
        --border-light: #3A3430;
        --text: #E8E2D5;
        --text-muted: #9A9488;
        --text-dim: #6A6458;
        --accent: #D4935A;
        --accent-glow: rgba(212, 147, 90, 0.12);
        --accent-border: rgba(212, 147, 90, 0.25);
        --teal: #5C8A7A;
        --teal-glow: rgba(92, 138, 122, 0.12);
        --red: #C2514A;
        --red-glow: rgba(194, 81, 74, 0.12);
        --red-border: rgba(194, 81, 74, 0.25);
        --green: #6B9F71;
        --green-glow: rgba(107, 159, 113, 0.12);
        --green-border: rgba(107, 159, 113, 0.25);
        --amber: #C8A06A;
        --font-serif: 'Playfair Display', Georgia, serif;
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    }

    .stApp {
        background: var(--bg);
        font-family: var(--font-sans);
        color: var(--text);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ── Tipografía ── */
    h1, h2, h3, h4, .serif {
        font-family: var(--font-serif);
        color: var(--text);
    }
    .mono {
        font-family: var(--font-mono);
    }

    /* ── Hero Section ── */
    .hero {
        padding: 3rem 1.5rem 2rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
        position: relative;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 120px;
        height: 2px;
        background: var(--accent);
    }
    .hero-label {
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 0.75rem;
    }
    .hero-title {
        font-family: var(--font-serif);
        font-size: 3.2rem;
        font-weight: 800;
        color: var(--text);
        line-height: 1.1;
        margin: 0 0 0.75rem;
        letter-spacing: -0.5px;
    }
    .hero-title em {
        font-style: italic;
        color: var(--accent);
    }
    .hero-sub {
        font-size: 1rem;
        color: var(--text-muted);
        max-width: 580px;
        line-height: 1.7;
        font-weight: 300;
    }
    .hero-sub strong {
        color: var(--text);
        font-weight: 500;
    }

    /* ── Cards ── */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        padding: 1.5rem;
        transition: all 0.25s ease;
    }
    .card:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-light);
    }
    .card-title {
        font-family: var(--font-serif);
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
        margin: 0 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }

    /* ── Result Cards ── */
    .result-card {
        padding: 2rem 1.5rem;
        text-align: center;
        border: 1px solid;
        transition: all 0.3s ease;
    }
    .result-card.fake {
        background: var(--red-glow);
        border-color: var(--red-border);
    }
    .result-card.real {
        background: var(--green-glow);
        border-color: var(--green-border);
    }
    .result-card .icon {
        font-size: 3.5rem;
        display: block;
        margin-bottom: 0.75rem;
        line-height: 1;
    }
    .result-card .label {
        font-family: var(--font-serif);
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 1px;
    }
    .result-card .label.fake { color: var(--red); }
    .result-card .label.real { color: var(--green); }
    .result-card .sub {
        font-size: 0.8rem;
        color: var(--text-dim);
        margin-top: 0.4rem;
        font-family: var(--font-mono);
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* ── Confidence Ring ── */
    .gauge-wrap {
        text-align: center;
        padding: 0.75rem;
    }
    .gauge-ring {
        position: relative;
        width: 130px;
        height: 130px;
        margin: 0 auto 0.75rem;
    }
    .gauge-ring svg { transform: rotate(-90deg); }
    .gauge-ring .track {
        fill: none;
        stroke: rgba(255,255,255,0.06);
        stroke-width: 6;
    }
    .gauge-ring .fill {
        fill: none;
        stroke-width: 6;
        stroke-linecap: round;
        transition: stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .gauge-val {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-family: var(--font-serif);
        font-size: 1.8rem;
        font-weight: 700;
    }
    .gauge-label {
        font-size: 0.7rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 2.5px;
        font-family: var(--font-mono);
    }

    /* ── Score Bar ── */
    .score-box {
        text-align: center;
        padding: 0.75rem;
    }
    .score-box .emoji-icon {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .score-box .value {
        font-family: var(--font-mono);
        font-size: 1.8rem;
        font-weight: 500;
        color: var(--text);
    }
    .score-box .label {
        font-size: 0.7rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 2.5px;
        font-family: var(--font-mono);
        margin-top: 0.15rem;
    }
    .score-track {
        margin-top: 0.75rem;
        padding: 0.6rem 0.75rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
    }
    .score-track .ticks {
        display: flex;
        justify-content: space-between;
        font-family: var(--font-mono);
        font-size: 0.6rem;
        color: var(--text-dim);
        margin-bottom: 0.35rem;
    }
    .score-track .bar {
        width: 100%;
        height: 3px;
        background: rgba(255,255,255,0.08);
    }
    .score-track .bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--red), var(--amber), var(--green));
        transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Section Titles ── */
    .sec-title {
        font-family: var(--font-serif);
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text);
        margin: 2rem 0 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
        position: relative;
    }
    .sec-title::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 60px;
        height: 2px;
        background: var(--accent);
    }

    /* ── Explanation ── */
    .explain-box {
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--border);
        padding: 1.25rem;
        font-size: 0.95rem;
        line-height: 1.8;
        color: var(--text-muted);
    }
    .explain-box strong {
        color: var(--accent);
        font-weight: 600;
    }

    /* ── Heatmap Legend ── */
    .heat-legend {
        display: flex;
        align-items: center;
        gap: 10px;
        justify-content: center;
        margin-top: 0.75rem;
    }
    .heat-legend span {
        font-size: 0.7rem;
        color: var(--text-dim);
        font-family: var(--font-mono);
    }
    .heat-bar {
        width: 160px;
        height: 8px;
        background: linear-gradient(to right, #2A4A6A, #3A7A6A, #5A8A4A, #B08A3A, #B04A3A);
    }

    /* ── Buttons ── */
    div.stButton > button:first-child {
        background: transparent;
        color: var(--text);
        border: 1px solid var(--accent-border);
        border-radius: 0;
        padding: 0.6rem 1.8rem;
        font-family: var(--font-sans);
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.25s ease;
        letter-spacing: 0.3px;
    }
    div.stButton > button:first-child:hover {
        background: var(--accent-glow);
        border-color: var(--accent);
        color: var(--accent);
    }
    div.stButton > button:first-child:active {
        transform: none;
    }
    div.stButton > button:first-child:disabled {
        opacity: 0.3;
        border-color: var(--border);
        color: var(--text-dim);
    }

    /* ── File Uploader ── */
    .stFileUploader {
        border: 1px dashed var(--border-light);
        padding: 0.75rem;
        background: rgba(255,255,255,0.01);
        transition: all 0.25s ease;
    }
    .stFileUploader:hover {
        border-color: var(--accent-border);
        background: var(--accent-glow);
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        border: none !important;
        padding: 0.75rem;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] div {
        color: var(--text-muted) !important;
        font-family: var(--font-sans) !important;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] svg {
        fill: var(--accent) !important;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] small {
        color: var(--text-dim) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0A0907 !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-muted);
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
    }

    .sidebar-title {
        font-family: var(--font-serif);
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text);
        text-align: center;
        margin: 0;
    }
    .sidebar-sub {
        font-size: 0.75rem;
        color: var(--text-dim);
        text-align: center;
        font-family: var(--font-mono);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.3rem;
    }
    .sidebar-heading {
        font-family: var(--font-serif);
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.6rem;
    }
    .sidebar-text {
        font-size: 0.85rem;
        color: var(--text-muted);
        line-height: 1.7;
    }
    .sidebar-text strong, .sidebar-text b {
        color: var(--accent);
        font-weight: 500;
    }
    .sidebar-ol {
        color: var(--text-muted);
        font-size: 0.85rem;
        line-height: 1.9;
        padding-left: 1.1rem;
        margin: 0;
    }

    /* ── Stats ── */
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
    }
    .stat-item {
        text-align: center;
        padding: 0.6rem 0.3rem;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.02);
    }
    .stat-item .val {
        font-family: var(--font-mono);
        font-size: 0.95rem;
        font-weight: 500;
        color: var(--accent);
    }
    .stat-item .lbl {
        font-size: 0.6rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.2rem;
    }

    /* ── Robustness Result ── */
    .rob-result {
        border: 1px solid var(--border);
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        color: var(--text-muted);
        font-size: 0.9rem;
        background: rgba(255,255,255,0.015);
    }
    .rob-result strong {
        color: var(--text);
    }

    /* ── History ── */
    .hist-item {
        border: 1px solid var(--border);
        padding: 0.75rem;
        text-align: center;
        transition: all 0.25s ease;
        background: rgba(255,255,255,0.015);
    }
    .hist-item:hover {
        border-color: var(--border-light);
        background: var(--bg-card-hover);
    }
    .hist-label {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: var(--text-dim);
        font-size: 0.75rem;
        border-top: 1px solid var(--border);
        margin-top: 2.5rem;
        font-family: var(--font-mono);
        letter-spacing: 0.5px;
    }
    .footer strong {
        color: var(--text-muted);
        font-weight: 500;
    }

    /* ── Loading ── */
    .loading-dots {
        display: inline-block;
    }
    .loading-dots::after {
        content: '';
        animation: dots 1.2s steps(4, end) infinite;
    }
    @keyframes dots {
        0% { content: ''; }
        25% { content: '.'; }
        50% { content: '..'; }
        75% { content: '...'; }
        100% { content: ''; }
    }
    .loading-text {
        color: var(--text-dim);
        font-family: var(--font-mono);
        font-size: 0.85rem;
    }

    /* ── Alerts / Messages ── */
    .stAlert, .stError, .stWarning, .stSuccess, .stInfo {
        border-radius: 0 !important;
        border-width: 1px !important;
        font-family: var(--font-sans) !important;
    }
    .stAlert {
        background: rgba(255,255,255,0.02) !important;
        border-color: var(--border) !important;
        color: var(--text-muted) !important;
    }
    .stError {
        background: var(--red-glow) !important;
        border-color: var(--red-border) !important;
        color: var(--red) !important;
    }
    .stSuccess {
        background: var(--green-glow) !important;
        border-color: var(--green-border) !important;
        color: var(--green) !important;
    }
    .stWarning {
        background: rgba(200, 160, 106, 0.08) !important;
        border-color: rgba(200, 160, 106, 0.2) !important;
        color: var(--amber) !important;
    }
    .stInfo {
        background: var(--accent-glow) !important;
        border-color: var(--accent-border) !important;
        color: var(--accent) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-color: var(--accent) !important;
        border-top-color: transparent !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border-light); }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

    /* ── Preview label ── */
    .preview-label {
        font-family: var(--font-mono);
        font-size: 0.65rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 2px;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* ── Sub section title ── */
    .sub-title {
        font-family: var(--font-serif);
        font-size: 1rem;
        font-weight: 600;
        color: var(--text);
        margin: 1.5rem 0 0.75rem;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .hero { padding: 2rem 1rem; }
        .result-card .label { font-size: 1.5rem; }
        .result-card .icon { font-size: 2.5rem; }
        .gauge-ring { width: 100px; height: 100px; }
        .gauge-val { font-size: 1.3rem; }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── Cargar modelo ───────────────────────────────────────────────────────────

@st.cache_resource
def load_model() -> Tuple[nn.Module, GradCAM]:
    model = build_densenet121(freeze_until_block=3)
    checkpoint_path = CHECKPOINT_BEST
    if not checkpoint_path.exists():
        checkpoint_path = CHECKPOINT_LAST
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Modelo cargado desde: {checkpoint_path}")
        st.sidebar.success(f"Modelo cargado: {checkpoint_path.name}")
    else:
        logger.warning("No se encontró checkpoint. Usando pesos pre-entrenados sin fine-tuning.")
        st.sidebar.warning(
            "No se encontró modelo entrenado. "
            "Usando pesos pre-entrenados de ImageNet. "
            "Los resultados serán subóptimos."
        )
    model.to(DEVICE)
    model.eval()
    gradcam = GradCAM(model)
    return model, gradcam


# ─── Funciones de procesamiento ──────────────────────────────────────────────

def predict_image(
    model: nn.Module,
    image: Image.Image,
) -> Tuple[int, float, torch.Tensor]:
    input_tensor = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
    return predicted.item(), confidence.item(), outputs


def compress_image(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def get_compression_info(quality: int) -> str:
    if quality >= 95:
        return "Sin compresion apreciable"
    elif quality >= 75:
        return "Compresion moderada"
    elif quality >= 50:
        return "Compresion agresiva"
    else:
        return "Compresion muy agresiva"


def render_gauge(pct: float, color: str) -> str:
    r = 52
    c = 2 * 3.14159 * r
    offset = c * (1 - pct / 100)
    return f"""
    <div class="gauge-ring">
        <svg width="130" height="130" viewBox="0 0 130 130">
            <circle class="track" cx="65" cy="65" r="{r}"/>
            <circle class="fill" cx="65" cy="65" r="{r}"
                    stroke="{color}" stroke-dasharray="{c}" stroke-dashoffset="{offset}"/>
        </svg>
        <div class="gauge-val" style="color: {color}">{pct:.1f}%</div>
    </div>
    """


# ─── Estado de sesion ─────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "robustness_results" not in st.session_state:
    st.session_state.robustness_results = None
if "current_result" not in st.session_state:
    st.session_state.current_result = None


# ─── Interfaz ────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:1.5rem 0 0.5rem;">'
            '<span style="font-size:2rem;">\U0001F50D</span>'
            '<p class="sidebar-title">DeepFake Detector</p>'
            '<p class="sidebar-sub">Forensic Analysis</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            '<p class="sidebar-heading">Protocolo</p>'
            '<ol class="sidebar-ol">'
            '<li>Cargar una imagen facial</li>'
            '<li>Ejecutar el analisis</li>'
            '<li>Revisar prediccion y Activacion</li>'
            '</ol>'
            '<p class="sidebar-text" style="margin-top:0.75rem;">'
            'Arquitectura: <b>DenseNet-121</b> con <b>Grad-CAM</b> '
            'para mapas de activacion.</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            f'<div class="stat-grid">'
            f'<div class="stat-item"><div class="val">{DEVICE.upper()}</div><div class="lbl">Dispositivo</div></div>'
            f'<div class="stat-item"><div class="val">{IMG_SIZE}x{IMG_SIZE}</div><div class="lbl">Entrada</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Cargar modelo ──
    loading = st.empty()
    loading.markdown(
        '<div style="text-align:center;padding:3rem;">'
        '<p class="loading-text">Inicializando modelo<span class="loading-dots"></span></p>'
        '</div>',
        unsafe_allow_html=True,
    )
    try:
        model, gradcam = load_model()
        model_ready = True
        loading.empty()
    except Exception as e:
        loading.empty()
        st.error(f"Error al cargar el modelo: {e}")
        model_ready = False
        gradcam = None

    # ── Hero ──
    st.markdown(
        '<div class="hero">'
        '<p class="hero-label">Computer Vision &bull; Deep Learning</p>'
        '<h1 class="hero-title">Detector de<br><em>DeepFakes</em></h1>'
        '<p class="hero-sub">'
        'Determine si una imagen facial es <strong>autentica</strong> o '
        '<strong>generada por inteligencia artificial</strong> '
        'mediante analisis espectral con mapas de activacion.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Upload ──
    uploaded_file = st.file_uploader(
        "Cargar imagen facial",
        type=["jpg", "jpeg", "png"],
        help="Formatos: JPG, JPEG, PNG",
    )

    col_btn, col_preview = st.columns([3, 2])

    with col_btn:
        analyze = st.button(
            "Ejecutar Analisis",
            type="primary",
            disabled=not (model_ready and uploaded_file is not None),
            use_container_width=True,
        )

    # ── Procesar ──
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with col_preview:
            st.markdown('<p class="preview-label">Vista previa</p>', unsafe_allow_html=True)
            st.image(image, width=200)

        if analyze:
            analyzing = st.empty()
            analyzing.markdown(
                '<div style="text-align:center;padding:2rem;">'
                '<p class="loading-text">Procesando<span class="loading-dots"></span></p>'
                '</div>',
                unsafe_allow_html=True,
            )
            try:
                class_idx, confidence, logits = predict_image(model, image)
                class_name = CLASS_NAMES[class_idx]
                heatmap_raw, heatmap_resized = gradcam.generate(image, class_idx)
                overlay = gradcam.overlay_heatmap(image, heatmap_resized)
                explanation = gradcam.generate_explanation(class_idx, confidence, heatmap_resized)
                overlay_b64 = image_to_base64(overlay)
                image_b64 = image_to_base64(image)

                st.session_state.current_result = {
                    "class_idx": class_idx, "class_name": class_name,
                    "confidence": confidence, "image": image,
                    "image_b64": image_b64, "overlay": overlay,
                    "overlay_b64": overlay_b64, "heatmap_raw": heatmap_raw,
                    "heatmap_resized": heatmap_resized, "explanation": explanation,
                    "logits": logits,
                }
                st.session_state.history.append({
                    "class_name": class_name, "confidence": confidence,
                    "image_b64": image_b64, "overlay_b64": overlay_b64,
                })
                if len(st.session_state.history) > MAX_HISTORY:
                    st.session_state.history = st.session_state.history[-MAX_HISTORY:]

                analyzing.empty()
            except Exception as e:
                analyzing.empty()
                st.error(f"Error durante el analisis: {e}")
                logger.error(f"Error en analisis: {e}", exc_info=True)

    # ── Resultados ──
    if st.session_state.current_result is not None:
        res = st.session_state.current_result
        is_fake = res["class_idx"] == 0
        color = "#C2514A" if is_fake else "#6B9F71"
        icon = "\u2718" if is_fake else "\u2714"
        cls = "fake" if is_fake else "real"

        st.markdown('<p class="sec-title">Dictamen</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f'<div class="result-card {cls}">'
                f'<span class="icon">{icon}</span>'
                f'<p class="label {cls}">{res["class_name"]}</p>'
                f'<p class="sub">Prediccion</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col2:
            conf_pct = res["confidence"] * 100
            gc = "#C2514A" if conf_pct < 70 else "#C8A06A" if conf_pct < 90 else "#6B9F71"
            st.markdown(
                f'<div class="card gauge-wrap">'
                + render_gauge(conf_pct, gc) +
                '<p class="gauge-label">Confianza</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        with col3:
            score = torch.softmax(res["logits"], dim=1)[0][1].item()
            emoji_icon = "\U0001F916" if is_fake else "\U0001F464"
            st.markdown(
                f'<div class="card score-box">'
                f'<div class="emoji-icon">{emoji_icon}</div>'
                f'<div class="value">{score:.4f}</div>'
                f'<div class="label">Score Real</div>'
                f'<div class="score-track">'
                f'<div class="ticks"><span>0.0000</span><span>1.0000</span></div>'
                f'<div class="bar"><div class="bar-fill" style="width:{score*100}%"></div></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Heatmap + Explanation ──
        st.markdown('<p class="sec-title">Activacion del Modelo</p>', unsafe_allow_html=True)

        hcol, ecol = st.columns(2)

        with hcol:
            st.markdown(
                '<div class="card">'
                '<p class="card-title">Mapa de Activacion Grad-CAM</p>',
                unsafe_allow_html=True,
            )
            st.image(res["overlay"], caption="Regiones de mayor influencia en la decision",
                     use_container_width=True)
            st.markdown(
                '<div class="heat-legend">'
                '<span>Baja</span>'
                '<div class="heat-bar"></div>'
                '<span>Alta</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="sub-title" style="font-size:0.9rem;margin-top:1.5rem;">'
                'Activacion pura</p>',
                unsafe_allow_html=True,
            )
            hd = (res["heatmap_raw"] * 255).astype(np.uint8)
            hp = Image.fromarray(hd, mode="L").resize((224, 224))
            st.image(hp, width=140)
            st.markdown('</div>', unsafe_allow_html=True)

        with ecol:
            st.markdown(
                f'<div class="card">'
                f'<p class="card-title">Informe</p>'
                f'<div class="explain-box">{res["explanation"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Robustez ──
        st.markdown('<p class="sec-title">Pruebas de Robustez</p>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.85rem;color:var(--text-muted);margin-bottom:1rem;">'
            'Efecto de la compresion JPEG en la prediccion.</p>',
            unsafe_allow_html=True,
        )

        r1, r2, r3 = st.columns(3)
        with r1:
            b100 = st.button("QF 100  Maxima", use_container_width=True, key="qf100")
        with r2:
            b75 = st.button("QF 75  Media", use_container_width=True, key="qf75")
        with r3:
            b50 = st.button("QF 50  Baja", use_container_width=True, key="qf50")

        rob_results = []
        for qf, btn in [(100, b100), (75, b75), (50, b50)]:
            if btn:
                comp = compress_image(res["image"], qf)
                cidx, cconf, _ = predict_image(model, comp)
                cname = CLASS_NAMES[cidx]
                rob_results.append({"qf": qf, "class_name": cname, "confidence": cconf, "image": comp})
                rm = "\u2718" if cname == "FAKE" else "\u2714"
                st.markdown(
                    f'<div class="rob-result">'
                    f'<b>QF {qf}</b> &mdash; {get_compression_info(qf)}: '
                    f'{rm} <b>{cname}</b> con <b>{cconf*100:.1f}%</b></div>',
                    unsafe_allow_html=True,
                )

        if rob_results:
            st.session_state.robustness_results = rob_results
            st.markdown(
                '<p class="sub-title">Comparativa</p>',
                unsafe_allow_html=True,
            )
            cols = st.columns(len(rob_results))
            for i, rr in enumerate(rob_results):
                with cols[i]:
                    rf = rr["class_name"] == "FAKE"
                    rc = "#C2514A" if rf else "#6B9F71"
                    rico = "\u2718" if rf else "\u2714"
                    st.markdown(
                        f'<div class="card" style="text-align:center;padding:1rem;">',
                        unsafe_allow_html=True,
                    )
                    st.image(rr["image"], caption=f"QF {rr['qf']}", use_container_width=True)
                    st.markdown(
                        f'<p style="font-weight:600;color:{rc};margin:0.5rem 0 0;'
                        f'font-size:1rem;">{rico} {rr["class_name"]}</p>'
                        f'<p style="color:var(--text-dim);font-size:0.85rem;margin:0;">'
                        f'{rr["confidence"]*100:.1f}%</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

    # ── Historial ──
    if st.session_state.history:
        st.markdown('<p class="sec-title">Historial</p>', unsafe_allow_html=True)
        cols = st.columns(min(len(st.session_state.history), MAX_HISTORY))
        for i, entry in enumerate(reversed(st.session_state.history[:MAX_HISTORY])):
            with cols[i]:
                himg = base64_to_image(entry["image_b64"])
                hf = entry["class_name"] == "FAKE"
                hc = "#C2514A" if hf else "#6B9F71"
                hico = "\u2718" if hf else "\u2714"
                st.markdown(
                    '<div class="hist-item">',
                    unsafe_allow_html=True,
                )
                st.image(himg, width=90)
                st.markdown(
                    f'<p class="hist-label" style="color:{hc};">'
                    f'{hico} {entry["class_name"]}<br>'
                    f'<span style="font-size:0.7rem;color:var(--text-dim);">'
                    f'{entry["confidence"]*100:.1f}%</span></p>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(
        '<div class="footer">'
        '<strong>DeepFake Detector</strong> &nbsp;&middot;&nbsp; '
        'DenseNet-121 &nbsp;&middot;&nbsp; '
        'Dataset 140K Real and Fake Faces'
        '</div>',
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error(
            "No se pudo cargar el modelo.\n\n"
            "Entrene el modelo primero ejecutando:\n"
            "```bash\npython -m src.train\n```"
        )


if __name__ == "__main__":
    main()
