"""
=============================================================================
OBJETIVO 1: Análisis de Características en Imágenes Reales y Deepfakes
=============================================================================
 
Este script analiza las características presentes en imágenes reales y deepfakes
mediante técnicas de procesamiento digital de imágenes usando OpenCV:
 
  1. Histogramas de color RGB
  2. Detección de bordes con filtro Canny
  3. Análisis de frecuencias con FFT (Transformada Rápida de Fourier)
 
Para usar:
  python analisis_caracteristicas.py
 
Requiere: opencv-python, numpy, matplotlib, pillow
"""
 
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import io

# ─── Carpeta de salida ────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent / "resultados_objetivo1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
 
# ==============================================================================
# PASO 1: OBTENER IMÁGENES DE MUESTRA (REAL Y FAKE)
# ==============================================================================
 
def obtener_imagenes():
    """
    Obtiene imágenes de muestra para el análisis.
    Primero intenta descargar desde internet, si no, crea imágenes sintéticas.
    """
    print("=" * 60)
    print("  PASO 1: Obteniendo imágenes de muestra...")
    print("=" * 60)
 
    sample_dir = OUTPUT_DIR / "muestras"
    sample_dir.mkdir(parents=True, exist_ok=True)
 
    real_path = sample_dir / "imagen_real.jpg"
    fake_path = sample_dir / "imagen_fake.jpg"
 
    # Intentar descargar imágenes reales
    try:
        import requests
        print("  -> Intentando descargar imagenes desde internet...")
 
        # Imagen real (retrato de dominio público)
        r = requests.get(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/"
            "Portrait_of_a_Young_Woman_%28Justine_%3F%29_LACMA_46.9.2_"
            "%28cropped%29.jpg/480px-Portrait_of_a_Young_Woman_%28Justine_"
            "%3F%29_LACMA_46.9.2_%28cropped%29.jpg",
            timeout=30, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r.status_code == 200:
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(real_path)
            print(f"  [OK] Imagen REAL descargada: {real_path.name}")
 
        # Imagen fake (generada por IA)
        r2 = requests.get(
            "https://thispersondoesnotexist.com/image",
            timeout=30, headers={"User-Agent": "Mozilla/5.0"}
        )
        if r2.status_code == 200:
            img2 = Image.open(io.BytesIO(r2.content)).convert("RGB")
            img2.save(fake_path)
            print(f"  [OK] Imagen FAKE descargada: {fake_path.name}")
    except:
        print("  [!] No se pudo descargar, creando imagenes sinteticas...")
 
    # Si no se descargaron, crear imágenes sintéticas con OpenCV
    if not real_path.exists():
        _crear_imagen_sintetica_real(real_path)
    if not fake_path.exists():
        _crear_imagen_sintetica_fake(fake_path)
 
    return real_path, fake_path
 
 
def _crear_imagen_sintetica_real(path):
    """Crea una imagen sintética simulando un rostro real."""
    img = np.ones((256, 256, 3), dtype=np.uint8) * 220
    # Rostro ovalado
    cv2.ellipse(img, (128, 130), (100, 120), 0, 0, 360, (230, 210, 190), -1)
    # Ojos
    cv2.circle(img, (95, 100), 12, (50, 45, 40), -1)
    cv2.circle(img, (160, 100), 12, (50, 45, 40), -1)
    # Pupilas
    cv2.circle(img, (95, 100), 5, (30, 25, 20), -1)
    cv2.circle(img, (160, 100), 5, (30, 25, 20), -1)
    # Nariz
    cv2.ellipse(img, (128, 135), (8, 20), 0, 0, 360, (180, 160, 145), -1)
    # Boca
    cv2.ellipse(img, (128, 170), (35, 12), 0, 0, 180, (160, 100, 100), 3)
    # Cabello
    cv2.ellipse(img, (128, 30), (120, 50), 0, 0, 360, (80, 60, 40), -1)
    cv2.imwrite(str(path), img)
    print(f"  [OK] Imagen REAL sintetica creada: {path.name}")
 
 
def _crear_imagen_sintetica_fake(path):
    """Crea una imagen sintética simulando un deepfake con artefactos."""
    img = np.ones((256, 256, 3), dtype=np.uint8) * 210
    # Rostro
    cv2.ellipse(img, (128, 128), (105, 120), 0, 0, 360, (225, 205, 195), -1)
    # Ojos (demasiado simétricos, característica GAN)
    cv2.circle(img, (95, 100), 14, (20, 20, 20), -1)
    cv2.circle(img, (160, 100), 14, (20, 20, 20), -1)
    cv2.circle(img, (95, 100), 8, (255, 255, 255), -1)
    cv2.circle(img, (160, 100), 8, (255, 255, 255), -1)
    # Nariz
    cv2.ellipse(img, (128, 133), (7, 22), 0, 0, 360, (195, 175, 165), -1)
    # Boca (patrón extraño)
    cv2.ellipse(img, (128, 170), (38, 10), 0, 0, 180, (180, 80, 90), 3)
    cv2.ellipse(img, (130, 170), (30, 8), 0, 0, 180, (200, 100, 100), 2)
    # Artefactos: ruido aleatorio (característico de GANs)
    ruido = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
    mask = np.random.random((256, 256)) > 0.97
    for c in range(3):
        img[:, :, c] = np.where(mask, ruido[:, :, c], img[:, :, c])
    # Patrón de cuadrícula (artefacto de GANs)
    for i in range(0, 256, 8):
        cv2.line(img, (i, 0), (i, 256), (0, 0, 0), 1)
    for i in range(0, 256, 8):
        cv2.line(img, (0, i), (256, i), (0, 0, 0), 1)
 
    cv2.imwrite(str(path), img)
    print(f"  [OK] Imagen FAKE sintetica creada: {path.name}")
 
 
# ==============================================================================
# PASO 2: ANÁLISIS DE HISTOGRAMAS DE COLOR RGB
# ==============================================================================
 
def analisis_histogramas(real_path, fake_path):
    """
    Genera histogramas de color para los canales R, G, B
    y compara las distribuciones entre imagen REAL y FAKE.
    """
    print("\n" + "=" * 60)
    print("  PASO 2: Histogramas de Color RGB")
    print("=" * 60)
    print("  -> Usando cv2.calcHist() para calcular histogramas...")
 
    # Cargar imágenes con OpenCV
    img_real = cv2.imread(str(real_path))
    img_fake = cv2.imread(str(fake_path))
    img_real_rgb = cv2.cvtColor(img_real, cv2.COLOR_BGR2RGB)
    img_fake_rgb = cv2.cvtColor(img_fake, cv2.COLOR_BGR2RGB)
 
    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Análisis de Características - Histogramas de Color RGB\n"
        "Objetivo 1: Identificar patrones para clasificacion",
        fontsize=14, fontweight="bold"
    )
 
    colores = [("Canal Rojo (R)", 2, "#FF4444"),
               ("Canal Verde (G)", 1, "#44BB44"),
               ("Canal Azul (B)", 0, "#4444FF")]
 
    # ─── Imagen REAL ────────────────────────────────────────────────────
    axes[0, 0].imshow(img_real_rgb)
    axes[0, 0].set_title("Imagen REAL", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")
 
    ax_real = axes[0, 1]
    for nombre, idx, color in colores:
        # cv2.calcHist: calcula el histograma de una imagen
        hist = cv2.calcHist([img_real], [idx], None, [256], [0, 256])
        ax_real.plot(hist, color=color, label=nombre, linewidth=1.5)
    ax_real.set_title("Histograma RGB - Imagen REAL", fontsize=12, fontweight="bold")
    ax_real.set_xlabel("Intensidad de píxel (0-255)")
    ax_real.set_ylabel("Frecuencia")
    ax_real.legend()
    ax_real.grid(True, alpha=0.3)
 
    # ─── Imagen FAKE ────────────────────────────────────────────────────
    axes[1, 0].imshow(img_fake_rgb)
    axes[1, 0].set_title("Imagen FAKE (deepfake)", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")
 
    ax_fake = axes[1, 1]
    for nombre, idx, color in colores:
        hist = cv2.calcHist([img_fake], [idx], None, [256], [0, 256])
        ax_fake.plot(hist, color=color, label=nombre, linewidth=1.5)
    ax_fake.set_title("Histograma RGB - Imagen FAKE", fontsize=12, fontweight="bold")
    ax_fake.set_xlabel("Intensidad de píxel (0-255)")
    ax_fake.set_ylabel("Frecuencia")
    ax_fake.legend()
    ax_fake.grid(True, alpha=0.3)
 
    plt.tight_layout()
    save_path = OUTPUT_DIR / "01_histogramas_color_rgb.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Histogramas guardados: {save_path.name}")
 
 
# ==============================================================================
# PASO 3: DETECCIÓN DE BORDES CON FILTRO CANNY
# ==============================================================================
 
def analisis_bordes_canny(real_path, fake_path):
    """
    Aplica el filtro Canny de OpenCV para detectar bordes
    y comparar la nitidez entre imágenes REAL y FAKE.
    """
    print("\n" + "=" * 60)
    print("  PASO 3: Detección de Bordes con Filtro Canny")
    print("=" * 60)
    print("  -> Usando cv2.Canny() para deteccion de bordes...")
 
    img_real = cv2.imread(str(real_path))
    img_fake = cv2.imread(str(fake_path))
    gray_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)
    gray_fake = cv2.cvtColor(img_fake, cv2.COLOR_BGR2GRAY)
 
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Detección de Bordes con Filtro Canny\n"
        "Objetivo 1: Identificar diferencias en nitidez y textura",
        fontsize=14, fontweight="bold"
    )
 
    umbrales = [(50, 150, "Umbrales bajos\n(50, 150)"),
                (100, 200, "Umbrales medios\n(100, 200)"),
                (30, 90, "Umbrales sensibles\n(30, 90)")]
 
    for i, (bajo, alto, titulo) in enumerate(umbrales):
        # Bordes de imagen REAL
        bordes_real = cv2.Canny(gray_real, bajo, alto)
        axes[0, i].imshow(bordes_real, cmap="gray")
        axes[0, i].set_title(f"REAL - {titulo}", fontsize=9)
        axes[0, i].axis("off")
 
        # Bordes de imagen FAKE
        bordes_fake = cv2.Canny(gray_fake, bajo, alto)
        axes[1, i].imshow(bordes_fake, cmap="gray")
        axes[1, i].set_title(f"FAKE - {titulo}", fontsize=9)
        axes[1, i].axis("off")
 
    plt.tight_layout()
    save_path = OUTPUT_DIR / "02_deteccion_bordes_canny.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Bordes Canny guardados: {save_path.name}")
 
 
# ==============================================================================
# PASO 4: ANÁLISIS DE FRECUENCIAS CON FFT
# ==============================================================================
 
def analisis_frecuencias_fft(real_path, fake_path):
    """
    Aplica la Transformada Rápida de Fourier (FFT) para analizar
    el contenido frecuencial de las imágenes.
    """
    print("\n" + "=" * 60)
    print("  PASO 4: Análisis de Frecuencias con FFT")
    print("=" * 60)
    print("  -> Usando np.fft.fft2() para transformada de Fourier...")
 
    img_real = cv2.imread(str(real_path))
    img_fake = cv2.imread(str(fake_path))
    gray_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)
    gray_fake = cv2.cvtColor(img_fake, cv2.COLOR_BGR2GRAY)
 
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Análisis de Frecuencias con Transformada Rápida de Fourier (FFT)\n"
        "Objetivo 1: Identificar artefactos de frecuencia en deepfakes",
        fontsize=14, fontweight="bold"
    )
 
    for fila, (gris, etiqueta) in enumerate([(gray_real, "REAL"),
                                              (gray_fake, "FAKE")]):
        # Calcular FFT
        f = np.fft.fft2(gris)                    # Transformada de Fourier
        fshift = np.fft.fftshift(f)              # Centrar frecuencias
        magnitud = 20 * np.log(np.abs(fshift) + 1)  # Espectro de magnitud
 
        # Filtro pasa-bajos (reconstrucción)
        filas, cols = gris.shape
        centro_f, centro_c = filas // 2, cols // 2
        mascara = np.zeros((filas, cols), np.uint8)
        mascara[centro_f-30:centro_f+30, centro_c-30:centro_c+30] = 1
        f_bajas = fshift * mascara
        img_bajas = np.abs(np.fft.ifft2(np.fft.ifftshift(f_bajas)))
 
        # Mostrar
        axes[fila, 0].imshow(gris, cmap="gray")
        axes[fila, 0].set_title(f"Imagen Original - {etiqueta}", fontsize=10)
        axes[fila, 0].axis("off")
 
        axes[fila, 1].imshow(magnitud, cmap="gray")
        axes[fila, 1].set_title(f"Espectro de Frecuencia - {etiqueta}", fontsize=10)
        axes[fila, 1].axis("off")
 
        axes[fila, 2].imshow(img_bajas, cmap="gray")
        axes[fila, 2].set_title(f"Reconstrucción Pasa-Bajos - {etiqueta}", fontsize=10)
        axes[fila, 2].axis("off")
 
    plt.tight_layout()
    save_path = OUTPUT_DIR / "03_analisis_frecuencias_fft.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Analisis FFT guardado: {save_path.name}")
 
 
# ==============================================================================
# RESUMEN FINAL
# ==============================================================================
 
def mostrar_resumen():
    """Muestra un resumen de los archivos generados."""
    print("\n" + "=" * 60)
    print("  [OK] OBJETIVO 1 COMPLETADO")
    print("=" * 60)
    print(f"\n  [DIR] Los resultados se guardaron en: {OUTPUT_DIR}/")
    print("\n  Archivos generados:")
    print(f"    [1] 01_histogramas_color_rgb.png")
    print(f"         -> cv2.calcHist() - Histogramas RGB")
    print(f"    [2] 02_deteccion_bordes_canny.png")
    print(f"         -> cv2.Canny() - Bordes con filtro Canny")
    print(f"    [3] 03_analisis_frecuencias_fft.png")
    print(f"         -> np.fft.fft2() - Analisis FFT")
    print(f"\n  [NOTA] Estas imagenes son evidencia del cumplimiento del")
    print(f"     Objetivo 1: Análisis de características de imágenes")
    print(f"     mediante técnicas de procesamiento digital con OpenCV.")
    print()
 
 
# ==============================================================================
# MAIN
# ==============================================================================
 
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  OBJETIVO 1: Analisis de Caracteristicas de Imagenes")
    print("  Procesamiento Digital de Imagenes con OpenCV")
    print("=" * 60)
    print()
    print("  Técnicas utilizadas:")
    print("  * cv2.calcHist()  -> Histogramas de color RGB")
    print("  * cv2.Canny()     -> Deteccion de bordes")
    print("  * np.fft.fft2()   -> Analisis de frecuencias (FFT)")
    print()
 
    # Ejecutar todos los análisis
    real_img, fake_img = obtener_imagenes()
    analisis_histogramas(real_img, fake_img)
    analisis_bordes_canny(real_img, fake_img)
    analisis_frecuencias_fft(real_img, fake_img)
    mostrar_resumen()
