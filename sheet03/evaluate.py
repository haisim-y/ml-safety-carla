"""
Exercise 3.6 — Evaluation

Loads the three saved ResNet-18 checkpoints and evaluates them on the test split.
Reports per model:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix
  - ROC curve and AUC score

Run from the project root:
    python sheet03/evaluate.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR       = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
OUT_DIR        = Path("sheet03")

TASKS = [
    ("Pedestrian",   "has_pedestrian",    "resnet18_pedestrian.pth"),
    ("Traffic Light","has_traffic_light",  "resnet18_trafficlight.pth"),
    ("Vehicle",      "has_vehicle",        "resnet18_vehicle.pth"),
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

THRESHOLD = 0.5   # probability above this → predict "present"


# ── Dataset (same as train.py, test split only) ────────────────────────────────

class CARLADataset(Dataset):
    def __init__(self, split: str, label_col: str, transform=None) -> None:
        self.img_dir   = DATA_DIR / split / "rgb-front"
        self.transform = transform

        df = pd.read_csv(DATA_DIR / split / "labels.csv", dtype={"frame": str})
        df["frame"] = df["frame"].str.zfill(6)

        self.frames = df["frame"].tolist()
        self.labels = df[label_col].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img  = Image.open(self.img_dir / f"{self.frames[idx]}.jpg").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(ckpt_name: str, device: torch.device) -> nn.Module:
    """Load a saved checkpoint and return the model in eval mode."""
    ckpt  = torch.load(CHECKPOINT_DIR / ckpt_name, map_location=device)
    model = resnet18(weights=None)        # no download — we load our own weights
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model over all batches.
    Returns:
        probs  — sigmoid probabilities, shape (N,)
        labels — ground-truth binary labels, shape (N,)
    """
    all_probs:  list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  inference", leave=False):
            images = images.to(device)
            logits = model(images).squeeze(1)   # (B, 1) → (B,)
            probs  = torch.sigmoid(logits)       # logit → probability 0–1
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_probs), np.array(all_labels)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    probs:  np.ndarray,
    labels: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict:
    preds = (probs >= threshold).astype(int)
    truth = labels.astype(int)

    return {
        "accuracy":  accuracy_score(truth, preds),
        "precision": precision_score(truth, preds, zero_division=0),
        "recall":    recall_score(truth, preds, zero_division=0),
        "f1":        f1_score(truth, preds, zero_division=0),
        "auc":       roc_auc_score(truth, probs),
        "cm":        confusion_matrix(truth, preds),
        "preds":     preds,
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(results: list[dict]) -> None:
    """One confusion matrix subplot per model."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Confusion Matrices — Test Split", fontsize=13, fontweight="bold")

    for ax, r in zip(axes, results):
        cm = r["cm"]
        im = ax.imshow(cm, cmap="Blues")

        # Annotate each cell with its count
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_title(r["name"], fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Absent (0)", "Present (1)"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Absent (0)", "Present (1)"])

    plt.tight_layout()
    out = OUT_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_roc_curves(results: list[dict]) -> None:
    """ROC curve for all three models on one plot."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors  = ["#e74c3c", "#f39c12", "#2ecc71"]

    for r, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{r['name']}  (AUC = {r['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate  (1 − Specificity)")
    ax.set_ylabel("True Positive Rate  (Recall)")
    ax.set_title("ROC Curves — Test Split", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out = OUT_DIR / "roc_curves.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


def plot_metric_comparison(results: list[dict]) -> None:
    """Bar chart comparing all metrics across the three models."""
    names   = [r["name"] for r in results]
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    colors  = ["#3498db", "#9b59b6", "#e74c3c", "#f39c12", "#2ecc71"]

    x    = np.arange(len(names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [r[metric] for r in results]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Metric Comparison Across Models — Test Split", fontweight="bold", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    out = OUT_DIR / "metric_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results: list[dict] = []

    for label_name, label_col, ckpt_name in TASKS:
        print(f"── {label_name} ──")

        dataset = CARLADataset("test", label_col, eval_transform)
        loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

        model  = load_model(ckpt_name, device)
        probs, labels = run_inference(model, loader, device)
        m = compute_metrics(probs, labels)

        print(f"  Accuracy : {m['accuracy']:.4f}")
        print(f"  Precision: {m['precision']:.4f}")
        print(f"  Recall   : {m['recall']:.4f}")
        print(f"  F1-score : {m['f1']:.4f}")
        print(f"  AUC      : {m['auc']:.4f}")

        tn, fp, fn, tp = m["cm"].ravel()
        print(f"  Confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}\n")

        results.append({
            "name":    label_name,
            "probs":   probs,
            "labels":  labels,
            **m,
        })

    # ── Summary table ──
    print("=" * 56)
    print(f"  {'Model':<15} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("=" * 56)
    for r in results:
        print(f"  {r['name']:<15} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
              f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['auc']:>6.3f}")
    print("=" * 56)

    # ── Plots ──
    print("\nGenerating plots...")
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_metric_comparison(results)

    print("\nDone — all figures saved to sheet03/")


if __name__ == "__main__":
    main()
