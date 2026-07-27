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
    MAX_HISTORY,
    CHECKPOINT_BEST,
    CHECKPOINT_LAST,
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
    initial_sidebar_state="collapsed",
)


# ─── CSS Personalizado - Premium Dark AI Dashboard ───────────────────────────

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Variables ── */
    :root {
        --bg-primary: #08080F;
        --bg-secondary: #0D0D18;
        --bg-card: #111122;
        --bg-card-hover: #161629;
        --border: #1E1E32;
        --border-light: #2A2A44;
        --text: #EEEEF8;
        --text-secondary: #8888B0;
        --text-muted: #55557A;
        --accent-1: #7C5CFC;
        --accent-2: #5C9DFC;
        --accent-gradient: linear-gradient(135deg, #7C5CFC, #5C9DFC);
        --glow-purple: rgba(124, 92, 252, 0.15);
        --glow-blue: rgba(92, 157, 252, 0.1);
        --success: #2ED47A;
        --success-bg: rgba(46, 212, 122, 0.1);
        --success-border: rgba(46, 212, 122, 0.2);
        --danger: #FF6B6B;
        --danger-bg: rgba(255, 107, 107, 0.1);
        --danger-border: rgba(255, 107, 107, 0.2);
        --warning: #FFB545;
        --warning-bg: rgba(255, 181, 69, 0.1);
        --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    }

    /* ── Base ── */
    .stApp {
        background: var(--bg-primary);
        font-family: var(--font-sans);
        color: var(--text);
    }
    #MainMenu, footer, .stDeployButton { display: none; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary); }
    [data-testid="stSidebar"] hr { border-color: var(--border) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

    /* ── Hero / Header ── */
    .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.5rem 0 1rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    .header-left { display: flex; align-items: center; gap: 1rem; }
    .header-logo {
        width: 40px; height: 40px;
        background: var(--accent-gradient);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem;
    }
    .header-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.3px;
        margin: 0;
    }
    .header-sub {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: var(--font-mono);
        margin: 0;
    }
    .header-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    .status-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--success);
        box-shadow: 0 0 8px rgba(46, 212, 122, 0.4);
    }
    .status-text {
        font-size: 0.75rem;
        color: var(--text-secondary);
        font-family: var(--font-mono);
    }

    /* ── Hero section ── */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 2rem;
        position: relative;
    }
    .hero-glow {
        position: absolute;
        top: -30%;
        left: 50%;
        transform: translateX(-50%);
        width: 400px;
        height: 400px;
        background: radial-gradient(ellipse, rgba(124,92,252,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 1rem;
        background: var(--glow-purple);
        border: 1px solid rgba(124,92,252,0.2);
        border-radius: 20px;
        font-size: 0.75rem;
        color: var(--accent-1);
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -1px;
        margin: 0 0 0.75rem;
        line-height: 1.15;
    }
    .hero-title span {
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-size: 1rem;
        color: var(--text-secondary);
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.7;
        font-weight: 400;
    }
    .hero-sub strong { color: var(--text); font-weight: 500; }

    /* ── Cards ── */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }
    .card:hover {
        border-color: var(--border-light);
        background: var(--bg-card-hover);
    }
    .card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin: 0 0 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .card-icon {
        width: 36px; height: 36px;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
    }

    /* ── Result Card ├-- */
    .result-card {
        padding: 1.75rem;
        border-radius: 12px;
        text-align: center;
    }
    .result-card.fake {
        background: linear-gradient(135deg, rgba(255,107,107,0.12), rgba(255,107,107,0.04));
        border: 1px solid var(--danger-border);
    }
    .result-card.real {
        background: linear-gradient(135deg, rgba(46,212,122,0.12), rgba(46,212,122,0.04));
        border: 1px solid var(--success-border);
    }
    .result-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin: 0;
    }
    .result-label.fake { color: var(--danger); }
    .result-label.real { color: var(--success); }
    .result-sub {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: var(--font-mono);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
    }

    /* ── Gauge ── */
    .gauge-wrap { text-align: center; padding: 0.5rem; }
    .gauge-ring {
        position: relative;
        width: 120px; height: 120px;
        margin: 0 auto 0.75rem;
    }
    .gauge-ring svg { transform: rotate(-90deg); }
    .gauge-track {
        fill: none; stroke: rgba(255,255,255,0.05);
        stroke-width: 6;
    }
    .gauge-fill {
        fill: none; stroke-width: 6;
        stroke-linecap: round;
        transition: stroke-dashoffset 1s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .gauge-val {
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.5rem;
        font-weight: 700;
        font-family: var(--font-sans);
    }
    .gauge-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: var(--font-mono);
    }

    /* ── Score ── */
    .score-wrap { text-align: center; padding: 0.5rem; }
    .score-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .score-num {
        font-size: 1.8rem;
        font-weight: 600;
        font-family: var(--font-mono);
        color: var(--text);
    }
    .score-lbl {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: var(--font-mono);
        margin-top: 0.1rem;
    }
    .score-bar {
        margin-top: 0.75rem;
        padding: 0.5rem 0.75rem;
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--border);
        border-radius: 6px;
    }
    .score-bar-labels {
        display: flex; justify-content: space-between;
        font-family: var(--font-mono);
        font-size: 0.55rem;
        color: var(--text-muted);
        margin-bottom: 0.3rem;
    }
    .score-bar-track {
        width: 100%; height: 3px;
        background: rgba(255,255,255,0.06);
        border-radius: 2px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--danger), var(--warning), var(--success));
        border-radius: 2px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Seccion Title ── */
    .sec-title {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--text);
        margin: 2rem 0 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border);
    }
    .sec-title-dot {
        width: 4px; height: 20px;
        background: var(--accent-gradient);
        border-radius: 2px;
    }

    /* ── Explanation ── */
    .explain-box {
        padding: 1.25rem;
        border: 1px solid var(--border);
        border-radius: 8px;
        font-size: 0.9rem;
        line-height: 1.8;
        color: var(--text-secondary);
        background: rgba(255,255,255,0.015);
    }
    .explain-box strong { color: var(--accent-1); font-weight: 500; }

    /* ── Heatmap Legend ── */
    .heat-legend {
        display: flex; align-items: center;
        gap: 10px; justify-content: center;
        margin-top: 0.75rem;
    }
    .heat-legend span {
        font-size: 0.65rem;
        color: var(--text-muted);
        font-family: var(--font-mono);
    }
    .heat-bar {
        width: 160px; height: 8px;
        border-radius: 4px;
        background: linear-gradient(to right, #1A3A6A, #2A6A8A, #3A8A5A, #8A8A3A, #9A3A3A);
    }

    /* ── Uploader ── */
    .stFileUploader {
        border: 1.5px dashed rgba(124,92,252,0.25);
        border-radius: 10px;
        padding: 0.5rem;
        background: rgba(124,92,252,0.03);
        transition: all 0.25s ease;
    }
    .stFileUploader:hover {
        border-color: rgba(124,92,252,0.5);
        background: rgba(124,92,252,0.06);
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        border: none !important;
        padding: 0.75rem;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] div {
        color: var(--text-secondary) !important;
        font-family: var(--font-sans) !important;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] svg {
        fill: var(--accent-1) !important;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] small {
        color: var(--text-muted) !important;
    }

    /* ── Buttons ── */
    div.stButton > button:first-child {
        background: var(--accent-gradient);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.5rem;
        font-family: var(--font-sans);
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 20px rgba(124,92,252,0.25);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 25px rgba(124,92,252,0.35);
    }
    div.stButton > button:first-child:active { transform: translateY(0); }
    div.stButton > button:first-child:disabled {
        opacity: 0.35;
        box-shadow: none;
    }

    /* ── Secondary buttons ── */
    div.stButton > button[kind="secondary"] {
        background: transparent;
        border: 1px solid var(--border);
        color: var(--text-secondary);
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: var(--glow-purple);
        border-color: rgba(124,92,252,0.3);
        color: var(--accent-1);
    }

    /* ── Alerts ── */
    .stAlert, .stError, .stWarning, .stSuccess, .stInfo {
        border-radius: 8px !important;
        border-width: 1px !important;
        font-family: var(--font-sans) !important;
    }
    .stAlert {
        background: var(--bg-card) !important;
        border-color: var(--border) !important;
        color: var(--text-secondary) !important;
    }
    .stError {
        background: var(--danger-bg) !important;
        border-color: var(--danger-border) !important;
        color: var(--danger) !important;
    }
    .stSuccess {
        background: var(--success-bg) !important;
        border-color: var(--success-border) !important;
        color: var(--success) !important;
    }
    .stWarning {
        background: var(--warning-bg) !important;
        border-color: rgba(255,181,69,0.2) !important;
        color: var(--warning) !important;
    }
    .stInfo {
        background: var(--glow-purple) !important;
        border-color: rgba(124,92,252,0.2) !important;
        color: var(--accent-1) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-color: var(--accent-1) !important;
        border-top-color: transparent !important;
    }

    /* ── Loading ── */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        gap: 1rem;
    }
    .loading-spinner {
        width: 32px; height: 32px;
        border: 2px solid rgba(124,92,252,0.15);
        border-top: 2px solid var(--accent-1);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loading-text {
        font-size: 0.85rem;
        color: var(--text-muted);
        font-family: var(--font-mono);
    }

    /* ── Stats in sidebar ── */
    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
    }
    .stat-box {
        padding: 0.6rem;
        text-align: center;
        border: 1px solid var(--border);
        border-radius: 6px;
        background: rgba(255,255,255,0.015);
    }
    .stat-box .val {
        font-family: var(--font-mono);
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--accent-1);
    }
    .stat-box .lbl {
        font-size: 0.55rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.15rem;
    }

    /* ── Result item ── */
    .result-line {
        padding: 0.7rem 1rem;
        border: 1px solid var(--border);
        border-radius: 6px;
        margin: 0.35rem 0;
        font-size: 0.85rem;
        color: var(--text-secondary);
        background: rgba(255,255,255,0.01);
    }
    .result-line strong { color: var(--text); }

    /* ── History ── */
    .hist-card {
        padding: 0.75rem;
        text-align: center;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: rgba(255,255,255,0.01);
        transition: all 0.2s ease;
    }
    .hist-card:hover {
        border-color: var(--border-light);
        background: var(--bg-card-hover);
    }
    .hist-label {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }

    /* ── Preview label ── */
    .preview-lbl {
        font-family: var(--font-mono);
        font-size: 0.6rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-align: center;
        margin-bottom: 0.4rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: var(--text-muted);
        font-size: 0.7rem;
        border-top: 1px solid var(--border);
        margin-top: 3rem;
        font-family: var(--font-mono);
    }
    .footer strong { color: var(--text-secondary); font-weight: 500; }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .header { flex-direction: column; gap: 0.75rem; align-items: flex-start; }
        .result-label { font-size: 1.5rem; }
        .gauge-ring { width: 100px; height: 100px; }
        .gauge-val { font-size: 1.2rem; }
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
        logger.warning("Usando pesos pre-entrenados sin fine-tuning.")
        st.sidebar.warning("Usando pesos ImageNet. Resultados suboptimos.")
    model.to(DEVICE)
    model.eval()
    gradcam = GradCAM(model)
    return model, gradcam


# ─── Funciones ───────────────────────────────────────────────────────────────

def predict_image(model: nn.Module, image: Image.Image) -> Tuple[int, float, torch.Tensor]:
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
    if quality >= 95: return "Sin compresion apreciable"
    elif quality >= 75: return "Compresion moderada"
    elif quality >= 50: return "Compresion agresiva"
    else: return "Compresion muy agresiva"


def render_gauge(pct: float, color: str) -> str:
    r = 48
    c = 2 * 3.14159 * r
    offset = c * (1 - pct / 100)
    return f"""
    <div class="gauge-ring">
        <svg width="120" height="120" viewBox="0 0 120 120">
            <circle class="gauge-track" cx="60" cy="60" r="{r}"/>
            <circle class="gauge-fill" cx="60" cy="60" r="{r}"
                    stroke="{color}" stroke-dasharray="{c}" stroke-dashoffset="{offset}"/>
        </svg>
        <div class="gauge-val" style="color:{color}">{pct:.1f}%</div>
    </div>
    """


# ─── Estado ──────────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "robustness_results" not in st.session_state:
    st.session_state.robustness_results = None


# ─── UI Principal ────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ──
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:1.5rem 0.5rem 0.5rem;">'
            '<div style="width:44px;height:44px;margin:0 auto 0.75rem;'
            'background:var(--accent-gradient);border-radius:12px;'
            'display:flex;align-items:center;justify-content:center;font-size:1.4rem;">'
            '\U0001F50D</div>'
            '<p style="font-weight:700;color:var(--text);margin:0;font-size:1.05rem;">'
            'DeepFake Detector</p>'
            '<p style="font-size:0.65rem;color:var(--text-muted);font-family:var(--font-mono);'
            'text-transform:uppercase;letter-spacing:1.5px;margin-top:0.2rem;">'
            'Forensic Analysis</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.8rem;font-weight:600;color:var(--text);margin:0 0 0.5rem;">'
            'Como funciona</p>'
            '<ol style="color:var(--text-secondary);font-size:0.8rem;line-height:1.9;'
            'padding-left:1rem;margin:0;">'
            '<li>Carga una imagen facial</li>'
            '<li>Ejecuta el analisis</li>'
            '<li>Revisa la prediccion y activacion</li>'
            '</ol>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:0.75rem;color:var(--text-muted);margin-top:0.75rem;line-height:1.6;">'
            'Modelo: <strong style="color:var(--accent-1);">DenseNet-121</strong><br>'
            'Explicabilidad: <strong style="color:var(--accent-1);">Grad-CAM</strong></p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            f'<div class="stat-grid">'
            f'<div class="stat-box"><div class="val">{DEVICE.upper()}</div><div class="lbl">Dispositivo</div></div>'
            f'<div class="stat-box"><div class="val">{IMG_SIZE}x{IMG_SIZE}</div><div class="lbl">Entrada</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Cargar modelo ──
    loading = st.empty()
    loading.markdown(
        '<div class="loading-container">'
        '<div class="loading-spinner"></div>'
        '<div class="loading-text">Inicializando modelo...</div>'
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

    # ── Header ──
    st.markdown(
        '<div class="header">'
        '<div class="header-left">'
        '<div class="header-logo">\U0001F50D</div>'
        '<div>'
        '<p class="header-title">DeepFake Detector</p>'
        '<p class="header-sub">Computer Vision &bull; Deep Learning</p>'
        '</div>'
        '</div>'
        '<div class="header-status">'
        '<div class="status-dot"></div>'
        '<span class="status-text">Sistema activo</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Hero ──
    st.markdown(
        '<div class="hero">'
        '<div class="hero-glow"></div>'
        '<div class="hero-badge">'
        '\U0001F52C Inteligencia Artificial en accion</div>'
        '<h1 class="hero-title">Deteccion de<br><span>DeepFakes</span></h1>'
        '<p class="hero-sub">'
        'Determina si una imagen facial es <strong>autentica</strong> o ha sido '
        '<strong>generada por IA</strong> mediante analisis profundo '
        'con mapas de activacion.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Upload ──
    uploaded_file = st.file_uploader(
        "Arrastra una imagen facial aqui",
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
            st.markdown('<p class="preview-lbl">Previsualizacion</p>', unsafe_allow_html=True)
            st.image(image, width=200)

        if analyze:
            analyzing = st.empty()
            analyzing.markdown(
                '<div class="loading-container">'
                '<div class="loading-spinner"></div>'
                '<div class="loading-text">Analizando imagen...</div>'
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
        cls = "fake" if is_fake else "real"
        color = "#FF6B6B" if is_fake else "#2ED47A"
        icon = "\u2718" if is_fake else "\u2714"

        st.markdown(
            '<div class="sec-title">'
            '<div class="sec-title-dot"></div>'
            'Resultados del Analisis'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f'<div class="result-card {cls}">'
                f'<div class="result-icon">{icon}</div>'
                f'<p class="result-label {cls}">{res["class_name"]}</p>'
                f'<p class="result-sub">Prediccion</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c2:
            conf = res["confidence"] * 100
            gc = "#FF6B6B" if conf < 70 else "#FFB545" if conf < 90 else "#2ED47A"
            st.markdown(
                f'<div class="card gauge-wrap">'
                + render_gauge(conf, gc) +
                '<p class="gauge-label">Confianza</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        with c3:
            score = torch.softmax(res["logits"], dim=1)[0][1].item()
            ic = "\U0001F916" if is_fake else "\U0001F464"
            st.markdown(
                f'<div class="card score-wrap">'
                f'<div class="score-icon">{ic}</div>'
                f'<div class="score-num">{score:.4f}</div>'
                f'<div class="score-lbl">Score Real</div>'
                f'<div class="score-bar">'
                f'<div class="score-bar-labels"><span>0</span><span>1</span></div>'
                f'<div class="score-bar-track">'
                f'<div class="score-bar-fill" style="width:{score*100}%"></div>'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )

        # ── Heatmap + Explanacion ──
        st.markdown(
            '<div class="sec-title">'
            '<div class="sec-title-dot"></div>'
            'Activacion del Modelo'
            '</div>',
            unsafe_allow_html=True,
        )

        hcol, ecol = st.columns(2)

        with hcol:
            st.markdown(
                '<div class="card">'
                '<p class="card-title">\U0001F3AF Grad-CAM</p>',
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
            hd = (res["heatmap_raw"] * 255).astype(np.uint8)
            hp = Image.fromarray(hd, mode="L").resize((224, 224))
            st.markdown(
                '<p style="font-size:0.8rem;color:var(--text-muted);margin:1rem 0 0.5rem;'
                'font-family:var(--font-mono);">Activacion pura</p>',
                unsafe_allow_html=True,
            )
            st.image(hp, width=130)
            st.markdown('</div>', unsafe_allow_html=True)

        with ecol:
            st.markdown(
                f'<div class="card">'
                f'<p class="card-title">\U0001F4DD Informe</p>'
                f'<div class="explain-box">{res["explanation"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Robustez ──
        st.markdown(
            '<div class="sec-title">'
            '<div class="sec-title-dot"></div>'
            'Pruebas de Robustez'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:0.8rem;color:var(--text-secondary);margin-bottom:0.75rem;">'
            'Evalua el impacto de la compresion JPEG en la prediccion.</p>',
            unsafe_allow_html=True,
        )

        rb1, rb2, rb3 = st.columns(3)
        with rb1: b100 = st.button("QF 100  Maxima calidad", use_container_width=True, key="qf100")
        with rb2: b75 = st.button("QF 75  Calidad media", use_container_width=True, key="qf75")
        with rb3: b50 = st.button("QF 50  Calidad baja", use_container_width=True, key="qf50")

        rob_res = []
        for qf, btn in [(100, b100), (75, b75), (50, b50)]:
            if btn:
                comp = compress_image(res["image"], qf)
                cidx, cconf, _ = predict_image(model, comp)
                cname = CLASS_NAMES[cidx]
                rob_res.append({"qf": qf, "class_name": cname, "confidence": cconf, "image": comp})
                rm = "\u2718" if cname == "FAKE" else "\u2714"
                st.markdown(
                    f'<div class="result-line">'
                    f'<b>QF {qf}</b> &mdash; {get_compression_info(qf)}: '
                    f'{rm} <b>{cname}</b> con <b>{cconf*100:.1f}%</b></div>',
                    unsafe_allow_html=True,
                )

        if rob_res:
            st.session_state.robustness_results = rob_res
            st.markdown(
                '<p style="font-size:0.9rem;font-weight:600;color:var(--text);margin:1rem 0 0.75rem;">'
                'Comparativa visual</p>',
                unsafe_allow_html=True,
            )
            cols = st.columns(len(rob_res))
            for i, rr in enumerate(rob_res):
                with cols[i]:
                    rf2 = rr["class_name"] == "FAKE"
                    rc2 = "#FF6B6B" if rf2 else "#2ED47A"
                    ric2 = "\u2718" if rf2 else "\u2714"
                    st.markdown(
                        '<div class="card" style="text-align:center;padding:1rem;">',
                        unsafe_allow_html=True,
                    )
                    st.image(rr["image"], caption=f"QF {rr['qf']}", use_container_width=True)
                    st.markdown(
                        f'<p style="font-weight:600;color:{rc2};margin:0.5rem 0 0;'
                        f'font-size:1rem;">{ric2} {rr["class_name"]}</p>'
                        f'<p style="color:var(--text-muted);font-size:0.8rem;margin:0;">'
                        f'{rr["confidence"]*100:.1f}%</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

    # ── Historial ──
    if st.session_state.history:
        st.markdown(
            '<div class="sec-title">'
            '<div class="sec-title-dot"></div>'
            'Historial'
            '</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(min(len(st.session_state.history), MAX_HISTORY))
        for i, entry in enumerate(reversed(st.session_state.history[:MAX_HISTORY])):
            with cols[i]:
                himg = base64_to_image(entry["image_b64"])
                hf = entry["class_name"] == "FAKE"
                hc = "#FF6B6B" if hf else "#2ED47A"
                hico = "\u2718" if hf else "\u2714"
                st.markdown('<div class="hist-card">', unsafe_allow_html=True)
                st.image(himg, width=90)
                st.markdown(
                    f'<p class="hist-label" style="color:{hc};">'
                    f'{hico} {entry["class_name"]}<br>'
                    f'<span style="font-size:0.65rem;color:var(--text-muted);'
                    f'font-family:var(--font-mono);">'
                    f'{entry["confidence"]*100:.1f}%</span></p>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(
        '<div class="footer">'
        '<strong>DeepFake Detector</strong> &nbsp;&#183;&nbsp; '
        'DenseNet-121 &nbsp;&#183;&nbsp; '
        'Dataset 140K Real and Fake Faces'
        '</div>',
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error("No se pudo cargar el modelo. Entrenelo primero ejecutando: python -m src.train")


if __name__ == "__main__":
    main()
