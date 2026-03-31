"""
Genera gráficas de entrenamiento para la tesis a partir de training_metrics.json.

Gráficas producidas:
  1. Loss vs Época        (train + val)
  2. Accuracy vs Época    (train + val)
  3. Train loss vs Val loss (análisis de overfitting)

Uso:
    python src/plot_metrics.py --model mbert
    python src/plot_metrics.py --model xlmr
    python src/plot_metrics.py --model mbert --model xlmr   (superpone ambos)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

COLORS = {"mbert": "#1f77b4", "xlmr": "#d62728"}
LINESTYLES = {"train": "-", "val": "--"}


def load(model_key: str) -> list[dict]:
    path = MODELS_DIR / model_key / "training_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Entrena el modelo primero.")
    with open(path) as f:
        return json.load(f)


def plot_loss(datasets: dict[str, list], save_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_key, data in datasets.items():
        epochs = [d["epoch"] for d in data]
        color = COLORS[model_key]
        label_prefix = model_key.upper()
        ax.plot(epochs, [d["train_loss"] for d in data],
                color=color, linestyle="-", marker="o", markersize=4,
                label=f"{label_prefix} — train")
        ax.plot(epochs, [d["val_loss"] for d in data],
                color=color, linestyle="--", marker="s", markersize=4,
                label=f"{label_prefix} — val")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs Época")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = save_dir / "loss_vs_epoch.png"
    fig.savefig(out, dpi=150)
    print(f"Guardada: {out}")
    plt.close(fig)


def plot_accuracy(datasets: dict[str, list], save_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_key, data in datasets.items():
        epochs = [d["epoch"] for d in data]
        color = COLORS[model_key]
        label_prefix = model_key.upper()
        ax.plot(epochs, [d["train_accuracy"] for d in data],
                color=color, linestyle="-", marker="o", markersize=4,
                label=f"{label_prefix} — train")
        ax.plot(epochs, [d["val_accuracy"] for d in data],
                color=color, linestyle="--", marker="s", markersize=4,
                label=f"{label_prefix} — val")
    ax.set_xlabel("Época")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Época")
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = save_dir / "accuracy_vs_epoch.png"
    fig.savefig(out, dpi=150)
    print(f"Guardada: {out}")
    plt.close(fig)


def plot_overfitting(datasets: dict[str, list], save_dir: Path):
    """Train loss vs Val loss por época — la divergencia indica overfitting."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for model_key, data in datasets.items():
        color = COLORS[model_key]
        train_losses = [d["train_loss"] for d in data]
        val_losses   = [d["val_loss"]   for d in data]
        ax.scatter(train_losses, val_losses, color=color, label=model_key.upper(), zorder=3)
        ax.plot(train_losses, val_losses, color=color, alpha=0.4)
        # Anotar primera y última época
        ax.annotate("e1",  (train_losses[0],  val_losses[0]),  fontsize=8, color=color)
        ax.annotate(f"e{data[-1]['epoch']}", (train_losses[-1], val_losses[-1]), fontsize=8, color=color)

    # Línea de referencia: train == val (sin overfitting)
    all_vals = [d["train_loss"] for dd in datasets.values() for d in dd] + \
               [d["val_loss"]   for dd in datasets.values() for d in dd]
    lo, hi = min(all_vals) * 0.95, max(all_vals) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="train = val")

    ax.set_xlabel("Train Loss")
    ax.set_ylabel("Val Loss")
    ax.set_title("Train Loss vs Val Loss (overfitting)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = save_dir / "overfitting_analysis.png"
    fig.savefig(out, dpi=150)
    print(f"Guardada: {out}")
    plt.close(fig)


def main(models: list[str]):
    datasets = {}
    for m in models:
        datasets[m] = load(m)

    save_dir = ROOT / "data"
    save_dir.mkdir(exist_ok=True)

    plot_loss(datasets, save_dir)
    plot_accuracy(datasets, save_dir)
    plot_overfitting(datasets, save_dir)
    print("Listo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mbert", "xlmr"], nargs="+", required=True)
    args = parser.parse_args()
    main(args.model)
