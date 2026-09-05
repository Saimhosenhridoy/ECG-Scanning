import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

CLASS_NAMES = [
    "Abnormal heartbeat",
    "Myocardial Infarction",
    "Normal Person",
    "Patients that have History of MI",
]


def remove_bg_original(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    pil_img = pil_img.crop((int(w * 0.02), int(h * 0.16), int(w * 0.98), int(h * 0.97)))
    gray = ImageOps.autocontrast(pil_img.convert("L")).filter(ImageFilter.GaussianBlur(radius=1.0))
    img = np.array(gray)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = 255 - binary
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    cleaned = np.zeros_like(fg)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 80:
            cleaned[lab == i] = 255
    cleaned = cv2.dilate(cleaned, np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(255 - cleaned).convert("RGB")


def remove_bg_cropped(pil_img: Image.Image) -> Image.Image:
    gray = ImageOps.autocontrast(pil_img.convert("L")).filter(ImageFilter.GaussianBlur(radius=0.8))
    img = np.array(gray)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = 255 - binary
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    cleaned = np.zeros_like(fg)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 50:
            cleaned[lab == i] = 255
    return Image.fromarray(255 - cleaned).convert("RGB")


def smart_clean(pil_img: Image.Image, source: str = "original") -> Image.Image:
    if source == "cropped":
        return remove_bg_cropped(pil_img)
    return remove_bg_original(pil_img)
