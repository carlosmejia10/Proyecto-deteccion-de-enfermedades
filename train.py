"""
Entrenamiento y Evaluación del modelo CXR CNN.

[Notebook] Celdas asociadas:
- Entrenamiento: definición de dataset/dataloaders, pérdida (CE con class weights), optimizador (AdamW), scheduler, early stopping
- Evaluación: métricas (accuracy, precision, recall, F1 macro), matriz de confusión y reporte por clase

Requisitos:
- torch (y opcionalmente torchvision si se desean transforms avanzados; aquí evitamos dependencia)
- scikit-learn (para métricas). Si no está, se calculará solo accuracy básica

Uso:
  python train.py \
    --csv outputs/eda/index_imagenes.csv \
    --out-dir outputs/train \
    --epochs 10 --batch-size 64 --lr 3e-4 --weight-decay 1e-4 \
    --val-split 0.15 --img-size 224 --limit-per-class 8000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models.cxr_cnn import build_cxr_small_cnn, count_parameters

try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ---------------------------
# Dataset a partir del CSV
# ---------------------------

class ChestXrayCsvDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_to_index: Dict[str, int],
        image_size: int = 224,
        augment: bool = False,
        mean: float = 0.5,
        std: float = 0.25,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.augment = augment
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.df)

    def _random_flip(self, img: Image.Image) -> Image.Image:
        if self.augment and random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _random_rotate(self, img: Image.Image, max_deg: float = 5.0) -> Image.Image:
        if self.augment and random.random() < 0.3:
            angle = random.uniform(-max_deg, max_deg)
            return img.rotate(angle, resample=Image.BILINEAR)
        return img

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img).astype(np.float32) / 255.0  # (H,W)
        arr = (arr - self.mean) / self.std
        arr = np.expand_dims(arr, axis=0)  # (1,H,W)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        path = Path(row["path"])  # columna del CSV
        label_str = str(row["label"])  # columna del CSV

        img = Image.open(path).convert("L")
        img = img.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        img = self._random_flip(img)
        img = self._random_rotate(img)
        x = self._to_tensor(img)
        y = self.label_to_index[label_str]
        return x, y


def stratified_split(df: pd.DataFrame, val_split: float, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    parts: List[pd.DataFrame] = []
    parts_val: List[pd.DataFrame] = []
    for label, group in df.groupby("label"):
        idx = np.arange(len(group))
        rng.shuffle(idx)
        cut = int(len(group) * (1 - val_split))
        train_idx = idx[:cut]
        val_idx = idx[cut:]
        parts.append(group.iloc[train_idx])
        parts_val.append(group.iloc[val_idx])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True), \
           pd.concat(parts_val).sample(frac=1, random_state=seed).reset_index(drop=True)


def maybe_limit_per_class(df: pd.DataFrame, limit: Optional[int], seed: int = 42) -> pd.DataFrame:
    if not limit:
        return df
    parts: List[pd.DataFrame] = []
    for label, group in df.groupby("label"):
        parts.append(group.sample(min(limit, len(group)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def compute_class_weights(df: pd.DataFrame, label_to_index: Dict[str, int]) -> torch.Tensor:
    counts = df["label"].value_counts().to_dict()
    total = sum(counts.values())
    weights = np.zeros(len(label_to_index), dtype=np.float32)
    for lbl, idx in label_to_index.items():
        freq = counts.get(lbl, 0)
        weights[idx] = 0.0 if freq == 0 else total / (len(counts) * freq)
    return torch.tensor(weights, dtype=torch.float32)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------
# Limpieza de rutas problemáticas (macOS __MACOSX, '._' files, extensiones)
# ---------------------------

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_valid_path(path_str: str) -> bool:
    p = Path(path_str)
    if not p.exists():
        return False
    if any(part == "__MACOSX" for part in p.parts):
        return False
    if p.name.startswith("._"):
        return False
    if p.suffix.lower() not in VALID_EXTS:
        return False
    return True


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.cpu().tolist())
    acc = correct / max(total, 1)
    result = {"accuracy": acc}
    if SKLEARN_OK:
        result.update({
            "precision_macro": float(precision_score(all_targets, all_preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(all_targets, all_preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(all_targets, all_preds, average="macro", zero_division=0)),
            "report": classification_report(all_targets, all_preds, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(all_targets, all_preds).tolist(),
        })
    return result


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    running = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * y.size(0)
    return running / max(len(loader.dataset), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento CXR CNN")
    parser.add_argument("--csv", type=str, default="outputs/eda/index_imagenes.csv")
    parser.add_argument("--out-dir", type=str, default="outputs/train")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    # Filtrar rutas inválidas (.__MACOSX, archivos Apple '._', extensiones raras o inexistentes)
    df = df[df["path"].apply(is_valid_path)].reset_index(drop=True)
    # Asegurar columnas requeridas
    assert {"path", "label"}.issubset(df.columns), "El CSV debe contener columnas 'path' y 'label'"

    # Mapeo de etiquetas
    labels = sorted(df["label"].dropna().unique().tolist())
    label_to_index = {lbl: i for i, lbl in enumerate(labels)}
    num_classes = len(labels)

    # Partición estratificada y (opcional) limitación por clase
    train_df, val_df = stratified_split(df, val_split=args.val_split, seed=args.seed)
    train_df = maybe_limit_per_class(train_df, args.limit_per_class, seed=args.seed)
    val_df = maybe_limit_per_class(val_df, min(args.limit_per_class or 10**9, 2000), seed=args.seed)  # cap de validación para rapidez

    # Pesos de clase para CE debido a desbalance
    class_weights = compute_class_weights(train_df, label_to_index)

    # Datasets y loaders
    train_ds = ChestXrayCsvDataset(train_df, label_to_index, image_size=args.img_size, augment=True)
    val_ds = ChestXrayCsvDataset(val_df, label_to_index, image_size=args.img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count() or 2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count() or 2, pin_memory=True)

    # Modelo
    device = select_device()
    model = build_cxr_small_cnn(num_classes=num_classes, in_channels=1)
    model.to(device)
    print(f"Dispositivo: {device}")
    print(f"Parámetros entrenables: {count_parameters(model):,}")

    # Pérdida y optimizador
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1.0
    best_path = out_dir / "best.pt"
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        epoch_rec = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(epoch_rec)
        print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | val_acc {val_metrics['accuracy']:.4f} | f1 {val_metrics.get('f1_macro', float('nan')):.4f}")
        f1_val = val_metrics.get("f1_macro", val_metrics["accuracy"]) if SKLEARN_OK else val_metrics["accuracy"]
        if f1_val > best_f1:
            best_f1 = f1_val
            torch.save({
                "model_state": model.state_dict(),
                "label_to_index": label_to_index,
                "args": vars(args),
            }, best_path)

    # Métricas finales en best
    with torch.no_grad():
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        final_metrics = evaluate(model, val_loader, device)

    # Guardar historial y métricas
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "val_metrics.json").write_text(json.dumps(final_metrics, indent=2))
    print("\nMejor F1 (o acc si no sklearn):", best_f1)
    print("Métricas finales en validación guardadas en:", out_dir / "val_metrics.json")


if __name__ == "__main__":
    main()


