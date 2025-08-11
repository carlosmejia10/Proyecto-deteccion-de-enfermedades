"""
Inferencia con una sola imagen usando el checkpoint entrenado.

Uso:
  python infer.py --image /ruta/a/imagen.jpg --ckpt outputs/train/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

import torch
from torch import nn

from models.cxr_cnn import build_cxr_small_cnn
import matplotlib.pyplot as plt


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_from_ckpt(ckpt_path: Path) -> Tuple[nn.Module, Dict[str, int], Dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_to_index: Dict[str, int] = ckpt["label_to_index"]
    num_classes = len(label_to_index)
    args = ckpt.get("args", {})
    model = build_cxr_small_cnn(num_classes=num_classes, in_channels=1)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, label_to_index, args


def preprocess_image(img_path: Path, img_size: int = 224, mean: float = 0.5, std: float = 0.25) -> torch.Tensor:
    img = Image.open(img_path).convert("L")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - mean) / std
    arr = np.expand_dims(arr, axis=0)  # (1,H,W)
    arr = np.expand_dims(arr, axis=0)  # (N=1,1,H,W)
    return torch.from_numpy(arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia de una imagen con CXR CNN")
    parser.add_argument("--image", type=str, required=True, help="Ruta de la imagen a clasificar")
    parser.add_argument("--ckpt", type=str, default="outputs/train/best.pt", help="Ruta al checkpoint entrenado")
    parser.add_argument("--show", action="store_true", help="Mostrar ventana con la imagen y predicción")
    parser.add_argument("--save", type=str, default=None, help="Ruta para guardar una visualización (PNG)")
    args = parser.parse_args()

    img_path = Path(args.image).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    assert img_path.exists(), f"No existe la imagen: {img_path}"
    assert ckpt_path.exists(), f"No existe el checkpoint: {ckpt_path}"

    model, label_to_index, ckpt_args = load_model_from_ckpt(ckpt_path)
    index_to_label = {v: k for k, v in label_to_index.items()}
    img_size = int(ckpt_args.get("img_size", 224))

    x = preprocess_image(img_path, img_size=img_size)  # (1,1,H,W)
    device = select_device()
    model.to(device)
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = index_to_label[pred_idx]

    # Mostrar Top-3
    topk = min(3, len(probs))
    top_indices = np.argsort(-probs)[:topk]
    print(f"Imagen: {img_path}")
    print(f"Predicción: {pred_label}")
    for i in top_indices:
        print(f"  {index_to_label[i]}: {probs[i]:.4f}")

    # Visualización opcional
    if args.show or args.save:
        img = Image.open(img_path).convert("L").resize((img_size, img_size), resample=Image.BILINEAR)
        title = f"Pred: {pred_label}  |  " + ", ".join([f"{index_to_label[i]}={probs[i]:.3f}" for i in top_indices])
        plt.figure(figsize=(5,5))
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis('off')
        if args.save:
            out_path = Path(args.save)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=160, bbox_inches='tight')
            print(f"Visualización guardada en: {out_path}")
        if args.show:
            plt.show()
        plt.close()


if __name__ == "__main__":
    main()


