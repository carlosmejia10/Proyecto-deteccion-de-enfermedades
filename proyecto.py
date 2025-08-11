"""
Proyecto: AED de Rayos X de Tórax (COVID-19 Radiography y Chest X-Ray Pneumonia)

Ejecución por terminal (el usuario gestiona el entorno/venv):
  python proyecto.py --sample-limit 5000 --output-dir outputs/eda

Este script organiza el flujo del Análisis Exploratorio de Datos (AED) que luego
se trasladará a un Notebook. Cada bloque incluye comentarios indicando en qué
sección/celda del Notebook iría.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# [Notebook] Celda: Imports y configuración
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    import cv2
    from tqdm import tqdm
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Falta una dependencia: {exc}.\n"
        "Instala los paquetes requeridos en tu entorno (venv) antes de ejecutar."
    )

try:
    import kagglehub
except ModuleNotFoundError:
    kagglehub = None  # Permitimos pasar rutas manualmente si se desea


# [Notebook] Celda: Configuración de rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "eda"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Ajustes de visualización
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)

# Evitar warnings de imágenes grandes en PIL
Image.MAX_IMAGE_PIXELS = None


# [Notebook] Celda: Utilidades
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception:
        return None


def compute_image_stats(path: Path) -> Dict:
    """Calcula métricas básicas de una imagen (brillo/contraste/tamaño)."""
    img = safe_open_image(path)
    if img is None:
        return {
            "path": str(path),
            "ok": False,
            "width": None,
            "height": None,
            "mode": None,
            "channels": None,
            "mean": None,
            "std": None,
            "median": None,
            "aspect_ratio": None,
        }
    arr = np.array(img)
    height, width = arr.shape[:2]
    channels = 1 if arr.ndim == 2 else arr.shape[2]
    if channels == 1:
        gray = arr.astype(np.float32)
    else:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return {
        "path": str(path),
        "ok": True,
        "width": int(width),
        "height": int(height),
        "mode": img.mode,
        "channels": int(channels),
        "mean": float(np.mean(gray)),
        "std": float(np.std(gray)),
        "median": float(np.median(gray)),
        "aspect_ratio": float(width / height) if height > 0 else None,
    }


def batched(iterable: Iterable, batch_size: int = 512) -> Iterable[List]:
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def enrich_with_stats(df: pd.DataFrame, sample_limit: Optional[int] = None) -> pd.DataFrame:
    paths: List[str] = df["path"].tolist()
    if sample_limit is not None:
        paths = paths[:sample_limit]
    stats: List[Dict] = []
    for group in tqdm(batched(paths, 256), total=math.ceil(len(paths) / 256) if paths else 0):
        for p in group:
            stats.append(compute_image_stats(Path(p)))
    stats_df = pd.DataFrame(stats)
    return df.merge(stats_df, on="path", how="left")


def index_dataset(base: Path, class_dirs: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
    """Indexa imágenes y deduce la clase según el nombre de carpeta."""
    all_files = list_images(base)
    records: List[Dict] = []
    for p in all_files:
        rel = p.relative_to(base)
        parts_lower = [s.lower() for s in rel.parts]
        label: Optional[str] = None
        if class_dirs:
            for cls, hints in class_dirs.items():
                for h in hints:
                    key = h.lower()
                    if key in parts_lower or key in rel.as_posix().lower():
                        label = cls
                        break
                if label:
                    break
        records.append({
            "path": str(p),
            "relpath": str(rel),
            "label": label,
            "filename": p.name,
            "ext": p.suffix.lower(),
            "source_root": str(base),
        })
    return pd.DataFrame(records)


# [Notebook] Celda: Visualizaciones (se guardan en disco)
def plot_class_distributions(all_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.barplot(data=summary_df, x="dataset", y="n", hue="label", ax=axes[0])
    axes[0].set_title("Distribución de clases por dataset")
    axes[0].set_ylabel("Número de imágenes")
    axes[0].tick_params(axis="x", rotation=15)

    global_counts = all_df["label"].value_counts().reset_index()
    global_counts.columns = ["label", "n"]
    sns.barplot(data=global_counts, x="label", y="n", ax=axes[1])
    axes[1].set_title("Distribución global de clases")
    axes[1].set_ylabel("Número de imágenes")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig.savefig(output_dir / "clase_distribuciones.png", dpi=160)
    plt.close(fig)


def plot_quality_distributions(ok_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    sns.histplot(ok_df["width"], bins=40, ax=axes[0, 0])
    axes[0, 0].set_title("Ancho (px)")
    sns.histplot(ok_df["height"], bins=40, ax=axes[0, 1])
    axes[0, 1].set_title("Alto (px)")
    sns.histplot(ok_df["aspect_ratio"], bins=40, ax=axes[0, 2])
    axes[0, 2].set_title("Relación de aspecto (w/h)")
    sns.histplot(ok_df["mean"], bins=40, ax=axes[1, 0])
    axes[1, 0].set_title("Brillo (media gris)")
    sns.histplot(ok_df["std"], bins=40, ax=axes[1, 1])
    axes[1, 1].set_title("Contraste (std)")
    sns.histplot(ok_df["median"], bins=40, ax=axes[1, 2])
    axes[1, 2].set_title("Mediana (gris)")
    plt.tight_layout()
    fig.savefig(output_dir / "calidad_distribuciones.png", dpi=160)
    plt.close(fig)


def save_examples_grid(ok_df: pd.DataFrame, output_dir: Path, max_per_class: int = 6) -> None:
    classes = sorted(ok_df["label"].dropna().unique())
    for cls in classes:
        sub = ok_df[ok_df["label"] == cls]
        if sub.empty:
            continue
        sub = sub.sample(min(max_per_class, len(sub)), random_state=42)
        count = len(sub)
        cols = min(6, count)
        rows = int(math.ceil(count / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.array(axes).reshape(-1)
        for ax, (_, row) in zip(axes, sub.iterrows()):
            try:
                img = Image.open(row["path"]).convert("L")
                ax.imshow(img, cmap="gray")
                ax.set_title(Path(row["path"]).parent.name[:20])
                ax.axis("off")
            except Exception:
                ax.axis("off")
        for j in range(count, len(axes)):
            axes[j].axis("off")
        plt.suptitle(f"Ejemplos — {cls}")
        plt.tight_layout()
        fig.savefig(output_dir / f"ejemplos_{cls}.png", dpi=160)
        plt.close(fig)


# [Notebook] Celda: Descarga de datasets (o rutas manuales)
def get_datasets_paths(covid_path_arg: Optional[str], pneumonia_path_arg: Optional[str]) -> tuple[Path, Path]:
    if covid_path_arg and pneumonia_path_arg:
        return Path(covid_path_arg).resolve(), Path(pneumonia_path_arg).resolve()
    if kagglehub is None:
        raise SystemExit(
            "kagglehub no está instalado y no se proporcionaron rutas --covid-path/--pneumonia-path"
        )
    covid_path = Path(kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database")).resolve()
    pneumonia_path = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")).resolve()
    return covid_path, pneumonia_path


# [Notebook] Celda: Indexación y resumen
def build_index(covid_root: Path, pneumonia_root: Path) -> pd.DataFrame:
    covid_classes = {
        "COVID": ["COVID", "COVID-19"],
        "NORMAL": ["NORMAL"],
        "PNEUMONIA": ["PNEUMONIA"],
        "LUNG_OPACITY": ["Lung_Opacity", "LUNG_OPACITY"],
    }
    pneu_classes = {
        "NORMAL": ["NORMAL"],
        "PNEUMONIA": ["PNEUMONIA"],
    }
    covid_df = index_dataset(covid_root, covid_classes)
    pneu_df = index_dataset(pneumonia_root, pneu_classes)
    covid_df["dataset"] = "COVID19-Radiography"
    pneu_df["dataset"] = "ChestXray-Pneumonia"
    all_df = pd.concat([covid_df, pneu_df], ignore_index=True)
    # Limpieza básica
    all_df = all_df[all_df["ext"].isin(list(IMAGE_EXTENSIONS))]
    all_df = all_df[all_df["label"].notna()]  # descarta no etiquetadas
    return all_df.reset_index(drop=True)


def summarize_by_class(all_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        all_df.groupby(["dataset", "label"]).agg(n=("path", "count")).reset_index()
    )
    return summary


# [Notebook] Celda: Flujo principal (ejecución)
def run_eda(
    sample_limit: Optional[int] = 5000,
    output_dir: Path = OUTPUTS_DIR,
    covid_path: Optional[str] = None,
    pneumonia_path: Optional[str] = None,
    save_examples: bool = True,
    examples_per_class: int = 6,
) -> None:
    # Descarga/obtención de rutas
    covid_root, pneumonia_root = get_datasets_paths(covid_path, pneumonia_path)
    print("COVID path:", covid_root)
    print("PNEU path:", pneumonia_root)

    # Indexación y limpieza
    all_df = build_index(covid_root, pneumonia_root)
    print("Imágenes válidas:", len(all_df))

    # Resumen de clases
    summary = summarize_by_class(all_df)
    print("\nResumen por dataset y clase:\n", summary)
    summary.to_csv(output_dir / "resumen_clases.csv", index=False)
    all_df.to_csv(output_dir / "index_imagenes.csv", index=False)

    # Visualización de distribuciones de clases
    plot_class_distributions(all_df, summary, output_dir)

    # Métricas de calidad (muestra para velocidad)
    quality_df = enrich_with_stats(all_df, sample_limit=sample_limit)
    ok_df = quality_df[quality_df["ok"]]
    print("Filas con imagen legible:", int(ok_df.shape[0]))
    quality_df.to_csv(output_dir / "index_con_calidad.csv", index=False)

    # Visualizaciones de calidad
    plot_quality_distributions(ok_df, output_dir)

    # Ejemplos por clase (guardados como imágenes)
    if save_examples:
        save_examples_grid(ok_df, output_dir, max_per_class=examples_per_class)

    print(f"\nListo. Resultados guardados en: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AED CXR — COVID & Pneumonia")
    parser.add_argument("--sample-limit", type=int, default=5000, help="Límite de imágenes para calcular métricas de calidad")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUTS_DIR), help="Directorio donde guardar resultados")
    parser.add_argument("--covid-path", type=str, default=None, help="Ruta local del dataset COVID-19 Radiography (opcional)")
    parser.add_argument("--pneumonia-path", type=str, default=None, help="Ruta local del dataset Chest X-Ray Pneumonia (opcional)")
    parser.add_argument("--no-examples", action="store_true", help="No guardar grids de ejemplos por clase")
    parser.add_argument("--examples-per-class", type=int, default=6, help="Número de ejemplos por clase a mostrar")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_eda(
        sample_limit=args.sample_limit,
        output_dir=out_dir,
        covid_path=args.covid_path,
        pneumonia_path=args.pneumonia_path,
        save_examples=(not args.no_examples),
        examples_per_class=args.examples_per_class,
    )


