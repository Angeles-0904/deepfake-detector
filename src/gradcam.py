"""
Implementación de Grad-CAM (Gradient-weighted Class Activation Mapping)
para visualizar las regiones de la imagen que más influyen en la decisión
del modelo DenseNet-121.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.config import DEVICE, IMG_SIZE, CLASS_NAMES
from src.utils import preprocess_image


class GradCAM:
    """
    Grad-CAM para DenseNet-121.

    DenseNet-121 no tiene una única capa convolucional final como ResNet.
    La última capa convolucional es features.denseblock4.
    Para Grad-CAM usamos la salida de features.norm5 (después del último
    bloque denso y antes del pooling promedio global).
    """

    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """
        Args:
            model: Modelo DenseNet-121
            target_layer: Nombre de la capa objetivo.
                Por defecto usa la última capa Conv2d del modelo (dentro de denseblock4).
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.target_layer = target_layer or self._find_last_conv_name()

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """Registra hooks hacia adelante y hacia atrás en la capa objetivo."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_module = self._get_layer(self.target_layer)
        if target_module is None:
            # Fallback: buscar la última capa convolucional disponible
            target_module = self._find_last_conv()
            if target_module is None:
                raise ValueError(
                    f"No se pudo encontrar la capa '{self.target_layer}' "
                    f"ni ninguna capa convolucional en el modelo"
                )
            print(f"Grad-CAM usando capa alternativa: {self._get_layer_name(target_module)}")

        self._forward_handle = target_module.register_forward_hook(forward_hook)
        self._backward_handle = target_module.register_full_backward_hook(backward_hook)

    def _get_layer(self, layer_name: str) -> Optional[nn.Module]:
        """Obtiene un módulo del modelo por su nombre."""
        parts = layer_name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _get_layer_name(self, module: nn.Module) -> str:
        """Obtiene el nombre de un módulo en el modelo."""
        for name, mod in self.model.named_modules():
            if mod is module:
                return name
        return "unknown"

    def _find_last_conv_name(self) -> str:
        """Encuentra el nombre de la última capa convolucional del modelo."""
        last_name = "features.norm5"  # fallback
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_name = name
        return last_name

    def _find_last_conv(self) -> Optional[nn.Module]:
        """Encuentra la última capa convolucional del modelo."""
        name = self._find_last_conv_name()
        return self._get_layer(name)

    def generate(
        self,
        image: Image.Image,
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera el mapa de calor Grad-CAM para una imagen.

        Args:
            image: Imagen PIL de entrada
            class_idx: Índice de la clase objetivo. Si es None, usa la clase
                      con mayor probabilidad.

        Returns:
            heatmap: Mapa de calor (array 2D normalizado entre 0 y 1)
            cam: Mapa de calor redimensionado al tamaño de la imagen original
        """
        self.model.eval()

        # Preprocesar imagen
        input_tensor = preprocess_image(image).to(self.device)
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward pass para la clase objetivo
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "No se pudieron obtener los gradientes. "
                "Verifica que la capa objetivo sea una capa convolucional."
            )

        # Calcular pesos: promedio global de gradientes
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Calcular CAM: suma ponderada de los mapas de activación
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
        cam = torch.relu(cam)  # Solo valores positivos

        # Normalizar y redimensionar
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Redimensionar al tamaño de la imagen original
        cam_resized = cv2.resize(
            cam,
            (image.width, image.height),
            interpolation=cv2.INTER_LINEAR,
        )

        return cam, cam_resized

    def overlay_heatmap(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> Image.Image:
        """
        Superpone el mapa de calor sobre la imagen original.

        Args:
            image: Imagen PIL original
            heatmap: Mapa de calor (2D array del mismo tamaño que la imagen)
            alpha: Factor de transparencia (0 = solo imagen, 1 = solo heatmap)
            colormap: Mapa de color de OpenCV

        Returns:
            overlay: Imagen PIL con el heatmap superpuesto
        """
        img_array = np.array(image.convert("RGB"))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap), colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = (1 - alpha) * img_array + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return Image.fromarray(overlay)

    def generate_explanation(
        self,
        class_idx: int,
        confidence: float,
        heatmap: np.ndarray,
    ) -> str:
        """
        Genera una explicación textual automática basada en el Grad-CAM.

        Analiza el mapa de calor para determinar qué regiones de la imagen
        fueron más relevantes para la decisión.

        Args:
            class_idx: Índice de la clase predicha (0=FAKE, 1=REAL)
            confidence: Confianza de la predicción (0-1)
            heatmap: Mapa de calor redimensionado al tamaño de la imagen

        Returns:
            explanation: Texto explicativo generado automáticamente
        """
        # Dividir el heatmap en regiones (rostro: ojos, nariz, boca, etc.)
        h, w = heatmap.shape
        regions = {
            "ojos": heatmap[: h // 3, :],
            "nariz": heatmap[h // 3 : 2 * h // 3, :],
            "boca": heatmap[2 * h // 3 :, :],
            "borde superior": heatmap[: h // 5, :],
            "borde inferior": heatmap[4 * h // 5 :, :],
            "lado izquierdo": heatmap[:, : w // 3],
            "lado derecho": heatmap[:, 2 * w // 3 :],
            "centro del rostro": heatmap[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4],
        }

        # Encontrar las regiones con mayor activación
        region_scores = {
            name: region.mean() for name, region in regions.items()
        }
        top_regions = sorted(
            region_scores.items(), key=lambda x: x[1], reverse=True
        )[:3]
        top_regions = [
            name for name, score in top_regions if score > 0.3
        ]

        class_name = CLASS_NAMES[class_idx]
        confidence_pct = confidence * 100

        if not top_regions:
            # Si no hay regiones destacadas, usar una explicación genérica
            if class_idx == 1:  # REAL
                explanation = (
                    f"El modelo clasificó la imagen como **{class_name}** "
                    f"con un **{confidence_pct:.1f}%** de confianza. "
                    f"La red no encontró artefactos significativos en ninguna "
                    f"región del rostro, lo que respalda la autenticidad de la imagen."
                )
            else:  # FAKE
                explanation = (
                    f"El modelo clasificó la imagen como **{class_name}** "
                    f"con un **{confidence_pct:.1f}%** de confianza. "
                    f"Se detectaron patrones anómalos distribuidos en todo el rostro "
                    f"consistentes con características de imágenes generadas por IA."
                )
        else:
            regions_str = ", ".join(top_regions[:-1])
            if len(top_regions) > 1:
                regions_str += f" y {top_regions[-1]}"
            else:
                regions_str = top_regions[0]

            explanation = (
                f"El modelo clasificó la imagen como **{class_name}** "
                f"con un **{confidence_pct:.1f}%** de confianza. "
            )

            if class_idx == 0:  # FAKE
                explanation += (
                    f"Se detectaron artefactos visuales principalmente en "
                    f"la región de {regions_str}, que son característicos de "
                    f"imágenes generadas por modelos generativos adversarios (GANs)."
                )
            else:  # REAL
                explanation += (
                    f"Las regiones de {regions_str} muestran texturas naturales "
                    f"y coherentes, sin evidencia de artefactos sintéticos, "
                    f"lo que sugiere que la imagen es auténtica."
                )

        return explanation

    def cleanup(self):
        """Limpia los hooks registrados."""
        if hasattr(self, "_forward_handle"):
            self._forward_handle.remove()
        if hasattr(self, "_backward_handle"):
            self._backward_handle.remove()
