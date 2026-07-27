"""
Aplicacion web Streamlit para deteccion de deepfakes.
"""

import io
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

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


st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="\U0001F98B",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Nunito:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #0A0808;
        --bg-soft: #0D0A0A;
        --card: #141010;
        --card-hover: #1A1414;
        --border: #221A1A;
        --border-light: #2E2424;
        --text: #F0E8E8;
        --text-soft: #C8B0B0;
        --text-muted: #887070;
        --rose: #D4878F;
        --rose-soft: #E8B4B8;
        --rose-glow: rgba(212, 135, 143, 0.1);
        --rose-border: rgba(212, 135, 143, 0.2);
        --gold: #C9A96E;
        --gold-soft: #E0C88A;
        --gold-glow: rgba(201, 169, 110, 0.1);
        --success: #A8C8A0;
        --success-bg: rgba(168, 200, 160, 0.08);
        --success-border: rgba(168, 200, 160, 0.2);
        --danger: #D4878F;
        --danger-bg: rgba(212, 135, 143, 0.08);
        --danger-border: rgba(212, 135, 143, 0.2);
        --warning: #E0C88A;
        --font-serif: 'Playfair Display', Georgia, serif;
        --font-sans: 'Nunito', -apple-system, sans-serif;
    }

    .stApp {
        background: var(--bg);
        font-family: var(--font-sans);
        color: var(--text);
    }
    #MainMenu, footer, .stDeployButton { display: none; }

    [data-testid="stSidebar"] {
        background: var(--bg-soft) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stMarkdown { color: var(--text-soft); }
    [data-testid="stSidebar"] hr { border-color: var(--border) !important; }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }

    /* ── Tipografia ── */
    h1, h2, h3, h4, .serif { font-family: var(--font-serif); color: var(--text); }
    .mono { font-family: 'Courier New', monospace; }

    /* ── Header ── */
    .header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 1.25rem 0 0.75rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    .header-left { display: flex; align-items: center; gap: 0.9rem; }
    .header-logo {
        width: 38px; height: 38px;
        border: 1.5px solid var(--rose-border);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
    }
    .header-title {
        font-family: var(--font-serif);
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text);
        margin: 0;
    }
    .header-sub {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-family: var(--font-sans);
        font-weight: 300;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .header-status {
        display: flex; align-items: center; gap: 0.5rem;
        padding: 0.35rem 0.9rem;
        border: 1px solid var(--border);
        border-radius: 6px;
    }
    .status-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: var(--gold);
        box-shadow: 0 0 10px rgba(201, 169, 110, 0.3);
    }
    .status-text {
        font-size: 0.7rem;
        color: var(--text-soft);
        font-family: var(--font-sans);
        font-weight: 300;
    }

    /* ── Hero ── */
    .hero {
        text-align: center;
        padding: 2rem 1rem 1.75rem;
        position: relative;
    }
    .hero-glow {
        position: absolute;
        top: -40%;
        left: 50%;
        transform: translateX(-50%);
        width: 500px;
        height: 500px;
        background: radial-gradient(ellipse, rgba(212,135,143,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 1rem;
        background: var(--rose-glow);
        border: 1px solid var(--rose-border);
        border-radius: 20px;
        font-size: 0.7rem;
        color: var(--rose);
        font-weight: 500;
        margin-bottom: 1rem;
        font-family: var(--font-sans);
    }
    .hero-title {
        font-family: var(--font-serif);
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.5px;
        margin: 0 0 0.75rem;
        line-height: 1.15;
    }
    .hero-title span {
        color: var(--rose);
        font-style: italic;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: var(--text-soft);
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.7;
        font-weight: 300;
    }
    .hero-sub strong { color: var(--gold); font-weight: 500; }

    /* ── Cards ── */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.25rem;
        transition: all 0.2s ease;
    }
    .card:hover {
        border-color: var(--border-light);
        background: var(--card-hover);
    }
    .card-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-muted);
        margin: 0 0 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: var(--font-sans);
    }

    /* ── Result Cards ── */
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .result-box.real {
        background: var(--success-bg);
        border: 1px solid var(--success-border);
    }
    .result-box.fake {
        background: var(--danger-bg);
        border: 1px solid var(--danger-border);
    }
    .result-symbol {
        font-size: 2rem;
        margin-bottom: 0.4rem;
    }
    .result-name {
        font-family: var(--font-serif);
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .result-name.real { color: var(--success); }
    .result-name.fake { color: var(--danger); }
    .result-foot {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
        font-family: var(--font-sans);
        font-weight: 300;
    }

    /* ── Gauge ── */
    .gauge-box { text-align: center; padding: 0.25rem; }
    .gauge-ring {
        position: relative;
        width: 110px; height: 110px;
        margin: 0 auto 0.5rem;
    }
    .gauge-ring svg { transform: rotate(-90deg); }
    .gauge-track {
        fill: none; stroke: rgba(255,255,255,0.04);
        stroke-width: 5;
    }
    .gauge-fill {
        fill: none; stroke-width: 5;
        stroke-linecap: round;
        transition: stroke-dashoffset 1s ease;
    }
    .gauge-val {
        position: absolute;
        top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        font-family: var(--font-serif);
        font-size: 1.4rem;
        font-weight: 600;
    }
    .gauge-foot {
        font-size: 0.6rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: var(--font-sans);
        font-weight: 300;
    }

    /* ── Score ── */
    .score-box { text-align: center; padding: 0.25rem; }
    .score-icon { font-size: 1.7rem; margin-bottom: 0.4rem; }
    .score-num {
        font-family: var(--font-serif);
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--text);
    }
    .score-foot {
        font-size: 0.6rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-family: var(--font-sans);
        font-weight: 300;
        margin-top: 0.1rem;
    }
    .score-track {
        margin-top: 0.6rem;
        padding: 0.4rem 0.6rem;
        border: 1px solid var(--border);
        border-radius: 5px;
        background: rgba(255,255,255,0.015);
    }
    .score-labels {
        display: flex; justify-content: space-between;
        font-size: 0.5rem;
        color: var(--text-muted);
        margin-bottom: 0.25rem;
        font-family: var(--font-sans);
    }
    .score-bar {
        width: 100%; height: 2px;
        background: rgba(255,255,255,0.05);
        border-radius: 2px;
        overflow: hidden;
    }
    .score-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--danger), var(--gold), var(--success));
        border-radius: 2px;
        transition: width 1s ease;
    }

    /* ── Section Title ── */
    .sec-title {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-family: var(--font-serif);
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
        margin: 1.75rem 0 0.75rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }
    .sec-accent {
        width: 3px; height: 18px;
        background: linear-gradient(180deg, var(--rose), var(--gold));
        border-radius: 2px;
    }

    /* ── Explanation ── */
    .explain-box {
        padding: 1rem 1.1rem;
        border: 1px solid var(--border);
        border-radius: 8px;
        font-size: 0.85rem;
        line-height: 1.8;
        color: var(--text-soft);
        background: rgba(255,255,255,0.01);
    }
    .explain-box strong { color: var(--gold); font-weight: 500; }

    /* ── Heat legend ── */
    .heat-legend {
        display: flex; align-items: center;
        gap: 10px; justify-content: center;
        margin-top: 0.6rem;
    }
    .heat-legend span {
        font-size: 0.6rem;
        color: var(--text-muted);
        font-family: var(--font-sans);
    }
    .heat-bar {
        width: 150px; height: 6px;
        border-radius: 3px;
        background: linear-gradient(to right, #2A1A4A, #4A1A5A, #5A3A3A, #6A4A2A, #7A2A2A);
    }

    /* ── Uploader ── */
    .stFileUploader {
        border: 1.5px dashed rgba(212,135,143,0.2);
        border-radius: 10px;
        padding: 0.4rem;
        background: rgba(212,135,143,0.02);
        transition: all 0.2s ease;
    }
    .stFileUploader:hover {
        border-color: rgba(212,135,143,0.4);
        background: rgba(212,135,143,0.05);
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        border: none !important;
        padding: 0.6rem;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] div {
        color: var(--text-soft) !important;
        font-family: var(--font-sans) !important;
        font-weight: 300 !important;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] svg {
        fill: var(--rose) !important;
    }
    .stFileUploader [data-testid="stFileUploaderDropzone"] small {
        color: var(--text-muted) !important;
    }

    /* ── Buttons ── */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, var(--rose), var(--gold));
        color: #0A0808;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-family: var(--font-sans);
        font-weight: 600;
        font-size: 0.85rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 20px rgba(212,135,143,0.2);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 25px rgba(212,135,143,0.3);
    }
    div.stButton > button:first-child:active { transform: translateY(0); }
    div.stButton > button:first-child:disabled {
        opacity: 0.3;
        box-shadow: none;
    }

    div.stButton > button[kind="secondary"] {
        background: transparent;
        border: 1px solid var(--border);
        color: var(--text-soft);
        border-radius: 8px;
        font-family: var(--font-sans);
        font-weight: 400;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: var(--rose-glow);
        border-color: var(--rose-border);
        color: var(--rose);
    }

    /* ── Alerts ── */
    .stAlert, .stError, .stWarning, .stSuccess, .stInfo {
        border-radius: 8px !important;
        border-width: 1px !important;
        font-family: var(--font-sans) !important;
    }
    .stAlert { background: var(--card) !important; border-color: var(--border) !important; color: var(--text-soft) !important; }
    .stError { background: var(--danger-bg) !important; border-color: var(--danger-border) !important; color: var(--danger) !important; }
    .stSuccess { background: var(--success-bg) !important; border-color: var(--success-border) !important; color: var(--success) !important; }
    .stWarning { background: rgba(224,200,138,0.08) !important; border-color: rgba(224,200,138,0.2) !important; color: var(--gold) !important; }
    .stInfo { background: var(--rose-glow) !important; border-color: var(--rose-border) !important; color: var(--rose) !important; }

    .stSpinner > div {
        border-color: var(--rose) !important;
        border-top-color: transparent !important;
    }

    .loading-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2.5rem;
        gap: 0.75rem;
    }
    .loading-spin {
        width: 28px; height: 28px;
        border: 2px solid rgba(212,135,143,0.12);
        border-top: 2px solid var(--rose);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loading-txt {
        font-size: 0.8rem;
        color: var(--text-muted);
        font-family: var(--font-sans);
        font-weight: 300;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
    }
    .stat-item {
        padding: 0.5rem;
        text-align: center;
        border: 1px solid var(--border);
        border-radius: 6px;
        background: rgba(255,255,255,0.01);
    }
    .stat-item .v {
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--rose);
        font-family: var(--font-sans);
    }
    .stat-item .l {
        font-size: 0.5rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.15rem;
    }

    .rob-line {
        padding: 0.6rem 0.9rem;
        border: 1px solid var(--border);
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 0.8rem;
        color: var(--text-soft);
        background: rgba(255,255,255,0.01);
    }
    .rob-line strong { color: var(--text); }

    .hist-item {
        padding: 0.6rem;
        text-align: center;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: rgba(255,255,255,0.01);
        transition: all 0.2s ease;
    }
    .hist-item:hover {
        border-color: var(--border-light);
        background: var(--card-hover);
    }
    .hist-lbl {
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }

    .preview-lbl {
        font-size: 0.6rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-align: center;
        margin-bottom: 0.4rem;
        font-family: var(--font-sans);
        font-weight: 300;
    }

    .footer {
        text-align: center;
        padding: 1.25rem;
        color: var(--text-muted);
        font-size: 0.65rem;
        border-top: 1px solid var(--border);
        margin-top: 2.5rem;
        font-family: var(--font-sans);
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    .footer strong { color: var(--text-soft); font-weight: 500; }

    @media (max-width: 768px) {
        .hero-title { font-size: 1.8rem; }
        .header { flex-direction: column; gap: 0.5rem; align-items: flex-start; }
        .result-name { font-size: 1.3rem; }
        .gauge-ring { width: 90px; height: 90px; }
        .gauge-val { font-size: 1.1rem; }
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model() -> Tuple[nn.Module, GradCAM]:
    model = build_densenet121(freeze_until_block=3)
    cp = CHECKPOINT_BEST
    if not cp.exists():
        cp = CHECKPOINT_LAST
    if cp.exists():
        ckpt = torch.load(cp, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Modelo cargado desde: {cp}")
        st.sidebar.success(f"Modelo cargado: {cp.name}")
    else:
        logger.warning("Usando pesos pre-entrenados sin fine-tuning.")
        st.sidebar.warning("Usando pesos ImageNet. Resultados suboptimos.")
    model.to(DEVICE)
    model.eval()
    gradcam = GradCAM(model)
    return model, gradcam


def predict_image(model: nn.Module, image: Image.Image) -> Tuple[int, float, torch.Tensor]:
    t = preprocess_image(image).to(DEVICE)
    with torch.no_grad():
        o = model(t)
        p = torch.softmax(o, dim=1)
        conf, pred = torch.max(p, dim=1)
    return pred.item(), conf.item(), o


def compress_image(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def get_compression_info(q: int) -> str:
    if q >= 95: return "Sin compresion apreciable"
    elif q >= 75: return "Compresion moderada"
    elif q >= 50: return "Compresion agresiva"
    else: return "Compresion muy agresiva"


def render_gauge(pct: float, color: str) -> str:
    r = 45
    c = 2 * 3.14159 * r
    off = c * (1 - pct / 100)
    return f"""
    <div class="gauge-ring">
        <svg width="110" height="110" viewBox="0 0 110 110">
            <circle class="gauge-track" cx="55" cy="55" r="{r}"/>
            <circle class="gauge-fill" cx="55" cy="55" r="{r}"
                    stroke="{color}" stroke-dasharray="{c}" stroke-dashoffset="{off}"/>
        </svg>
        <div class="gauge-val" style="color:{color}">{pct:.1f}%</div>
    </div>
    """


if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "robustness_results" not in st.session_state:
    st.session_state.robustness_results = None


def main():
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:1.25rem 0.5rem 0.25rem;">'
            '<div style="width:42px;height:42px;margin:0 auto 0.75rem;'
            'border:1.5px solid var(--rose-border);border-radius:10px;'
            'display:flex;align-items:center;justify-content:center;font-size:1.2rem;">'
            '\U0001F98B</div>'
            '<p style="font-family:var(--font-serif);font-weight:600;color:var(--text);'
            'margin:0;font-size:1rem;">DeepFake Detector</p>'
            '<p style="font-size:0.6rem;color:var(--text-muted);text-transform:uppercase;'
            'letter-spacing:1.5px;margin-top:0.2rem;">Analisis Forense</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.75rem;font-weight:500;color:var(--text);'
            'margin:0 0 0.5rem;">Procedimiento</p>'
            '<ol style="color:var(--text-soft);font-size:0.75rem;line-height:1.9;'
            'padding-left:1rem;margin:0;font-weight:300;">'
            '<li>Cargar una imagen facial</li>'
            '<li>Ejecutar el analisis</li>'
            '<li>Revisar prediccion y mapa de activacion</li>'
            '</ol>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:0.7rem;color:var(--text-muted);margin-top:0.75rem;'
            'line-height:1.6;font-weight:300;">'
            'Modelo: <strong style="color:var(--rose);font-weight:500;">DenseNet-121</strong><br>'
            'Explicabilidad: <strong style="color:var(--rose);font-weight:500;">Grad-CAM</strong></p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(
            f'<div class="stat-grid">'
            f'<div class="stat-item"><div class="v">{DEVICE.upper()}</div>'
            f'<div class="l">Dispositivo</div></div>'
            f'<div class="stat-item"><div class="v">{IMG_SIZE}x{IMG_SIZE}</div>'
            f'<div class="l">Entrada</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    loading = st.empty()
    loading.markdown(
        '<div class="loading-box">'
        '<div class="loading-spin"></div>'
        '<div class="loading-txt">Inicializando modelo...</div>'
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

    st.markdown(
        '<div class="header">'
        '<div class="header-left">'
        '<div class="header-logo">\U0001F98B</div>'
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

    st.markdown(
        '<div class="hero">'
        '<div class="hero-glow"></div>'
        '<div class="hero-badge">Inteligencia Artificial en accion</div>'
        '<h1 class="hero-title">Deteccion de<br><span>DeepFakes</span></h1>'
        '<p class="hero-sub">'
        'Determina si una imagen facial es <strong>autentica</strong> o ha sido '
        '<strong>generada por IA</strong> mediante analisis profundo '
        'con mapas de activacion.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

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

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with col_preview:
            st.markdown('<p class="preview-lbl">Previsualizacion</p>', unsafe_allow_html=True)
            st.image(image, width=200)

        if analyze:
            analyzing = st.empty()
            analyzing.markdown(
                '<div class="loading-box">'
                '<div class="loading-spin"></div>'
                '<div class="loading-txt">Analizando imagen...</div>'
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

    if st.session_state.current_result is not None:
        res = st.session_state.current_result
        is_fake = res["class_idx"] == 0
        cls = "fake" if is_fake else "real"
        color = "#D4878F" if is_fake else "#A8C8A0"

        st.markdown(
            '<div class="sec-title">'
            '<div class="sec-accent"></div>'
            'Resultados del Analisis'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(
                f'<div class="result-box {cls}">'
                f'<div class="result-symbol">{chr(10008) if is_fake else chr(10004)}</div>'
                f'<p class="result-name {cls}">{res["class_name"]}</p>'
                f'<p class="result-foot">Prediccion</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with c2:
            conf = res["confidence"] * 100
            gc = "#D4878F" if conf < 70 else "#E0C88A" if conf < 90 else "#A8C8A0"
            st.markdown(
                f'<div class="card gauge-box">'
                + render_gauge(conf, gc) +
                '<p class="gauge-foot">Confianza</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        with c3:
            score = torch.softmax(res["logits"], dim=1)[0][1].item()
            st.markdown(
                f'<div class="card score-box">'
                f'<div class="score-icon">{chr(9762)}</div>'
                f'<div class="score-num">{score:.4f}</div>'
                f'<div class="score-foot">Score Real</div>'
                f'<div class="score-track">'
                f'<div class="score-labels"><span>0</span><span>1</span></div>'
                f'<div class="score-bar">'
                f'<div class="score-fill" style="width:{score*100}%"></div>'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="sec-title">'
            '<div class="sec-accent"></div>'
            'Activacion del Modelo'
            '</div>',
            unsafe_allow_html=True,
        )

        hcol, ecol = st.columns(2)

        with hcol:
            st.markdown(
                '<div class="card">'
                '<p class="card-label">Mapa de Activacion Grad-CAM</p>',
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
                '<p style="font-size:0.75rem;color:var(--text-muted);margin:0.75rem 0 0.4rem;'
                'font-weight:300;">Activacion pura</p>',
                unsafe_allow_html=True,
            )
            st.image(hp, width=120)
            st.markdown('</div>', unsafe_allow_html=True)

        with ecol:
            st.markdown(
                f'<div class="card">'
                f'<p class="card-label">Informe</p>'
                f'<div class="explain-box">{res["explanation"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="sec-title">'
            '<div class="sec-accent"></div>'
            'Pruebas de Robustez'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:0.75rem;color:var(--text-soft);margin-bottom:0.75rem;'
            'font-weight:300;">Efecto de la compresion JPEG en la prediccion.</p>',
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
                sym = chr(10008) if cname == "FAKE" else chr(10004)
                st.markdown(
                    f'<div class="rob-line">'
                    f'<b>QF {qf}</b> -- {get_compression_info(qf)}: '
                    f'{sym} <b>{cname}</b> con <b>{cconf*100:.1f}%</b></div>',
                    unsafe_allow_html=True,
                )

        if rob_res:
            st.session_state.robustness_results = rob_res
            st.markdown(
                '<p style="font-size:0.85rem;font-weight:500;color:var(--text);'
                'margin:1rem 0 0.75rem;">Comparativa visual</p>',
                unsafe_allow_html=True,
            )
            cols = st.columns(len(rob_res))
            for i, rr in enumerate(rob_res):
                with cols[i]:
                    rf2 = rr["class_name"] == "FAKE"
                    rc2 = "#D4878F" if rf2 else "#A8C8A0"
                    sym2 = chr(10008) if rf2 else chr(10004)
                    st.markdown(
                        '<div class="card" style="text-align:center;padding:0.75rem;">',
                        unsafe_allow_html=True,
                    )
                    st.image(rr["image"], caption=f"QF {rr['qf']}", use_container_width=True)
                    st.markdown(
                        f'<p style="font-weight:600;color:{rc2};margin:0.4rem 0 0;'
                        f'font-size:0.95rem;">{sym2} {rr["class_name"]}</p>'
                        f'<p style="color:var(--text-muted);font-size:0.75rem;margin:0;'
                        f'font-weight:300;">{rr["confidence"]*100:.1f}%</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown(
            '<div class="sec-title">'
            '<div class="sec-accent"></div>'
            'Historial'
            '</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(min(len(st.session_state.history), MAX_HISTORY))
        for i, entry in enumerate(reversed(st.session_state.history[:MAX_HISTORY])):
            with cols[i]:
                himg = base64_to_image(entry["image_b64"])
                hf = entry["class_name"] == "FAKE"
                hc = "#D4878F" if hf else "#A8C8A0"
                sym = chr(10008) if hf else chr(10004)
                st.markdown('<div class="hist-item">', unsafe_allow_html=True)
                st.image(himg, width=85)
                st.markdown(
                    f'<p class="hist-lbl" style="color:{hc};">'
                    f'{sym} {entry["class_name"]}<br>'
                    f'<span style="font-size:0.6rem;color:var(--text-muted);'
                    f'font-weight:300;">{entry["confidence"]*100:.1f}%</span></p>',
                    unsafe_allow_html=True,
                )
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">'
        '<strong>DeepFake Detector</strong> &nbsp;&#183;&nbsp; '
        'DenseNet-121 &nbsp;&#183;&nbsp; '
        'Dataset 140K Real and Fake Faces'
        '</div>',
        unsafe_allow_html=True,
    )

    if not model_ready:
        st.error("No se pudo cargar el modelo. Entrenelo primero: python -m src.train")


if __name__ == "__main__":
    main()
