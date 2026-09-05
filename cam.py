from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image

from model_loader import get_device, load_model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out

    def _save_grad(self, module, gin, gout):
        self.gradients = gout[0]

    def generate(self, input_tensor, target_class: int) -> np.ndarray:
        x = input_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        with torch.inference_mode(False), torch.enable_grad():
            output = self.model(x)
            output[0, target_class].backward()
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not run. Try again after restarting the app.")
        grads = self.gradients.detach().cpu().numpy()
        acts = self.activations.detach().cpu().numpy()
        pooled = np.mean(grads, axis=(2, 3))
        weighted = acts[0] * pooled[0][:, None, None]
        heat = np.mean(weighted, axis=0)
        heat = np.maximum(heat, 0)
        heat /= np.max(heat) if np.max(heat) > 0 else 1.0
        return heat


@lru_cache(maxsize=1)
def get_cam() -> GradCAM:
    model = load_model()
    return GradCAM(model, model.features[-1])


def overlay_cam(cleaned: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    heat = cv2.resize(heatmap, cleaned.size)
    heat_bgr = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = Image.fromarray(cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB))
    return Image.blend(cleaned.convert("RGBA"), heat_rgb.convert("RGBA"), alpha=alpha)


def run_cam(cleaned: Image.Image, input_tensor: torch.Tensor, pred_index: int) -> Image.Image:
    cam = get_cam()
    heat = cam.generate(input_tensor, pred_index)
    return overlay_cam(cleaned, heat)