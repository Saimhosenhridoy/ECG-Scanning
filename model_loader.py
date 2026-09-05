from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torchvision import models, transforms

from preprocess import CLASS_NAMES
from settings import settings

IMG_SIZE = 384
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

HF_MODEL_REPO = "Saimhosenhridoy/ecg-convnext"
HF_MODEL_FILE = "convnext_tiny_ecg.pth"

INFER_TF = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ]
)


def get_weight_path() -> Path:
    local = Path(settings.model_path)
    if local.exists():
        return local
    fallback = Path("weights") / HF_MODEL_FILE
    if fallback.exists():
        return fallback
    downloaded = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=HF_MODEL_FILE,
    )
    return Path(downloaded)


def build_model(num_classes: int = 4) -> nn.Module:
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1),
        nn.LayerNorm(in_features, eps=1e-6),
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )
    return model


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_model() -> nn.Module:
    path = get_weight_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Weight file not found: {path}. "
            f"Put {HF_MODEL_FILE} in weights/ or upload it to {HF_MODEL_REPO}"
        )
    device = get_device()
    model = build_model(len(CLASS_NAMES))
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def predict_tensor(image_pil) -> dict:
    model = load_model()
    device = get_device()
    x = INFER_TF(image_pil).unsqueeze(0).to(device)
    logits = model(x)[0]
    prob = torch.softmax(logits, dim=0).detach().cpu().tolist()
    pred = int(max(range(len(prob)), key=lambda i: prob[i]))
    return {
        "pred_index": pred,
        "pred_label": CLASS_NAMES[pred],
        "probabilities": {CLASS_NAMES[i]: round(float(prob[i]), 6) for i in range(len(CLASS_NAMES))},
        "device": str(device),
    }