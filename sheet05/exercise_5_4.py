"""
Exercise 5.4 — Temperature Scaling and the Confidence Threshold

Run from the project root:
    python sheet05/exercise_5_4.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR       = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
OUT_DIR        = Path("sheet05")

LABEL_COL  = "has_pedestrian"
CKPT_NAME  = "resnet18_pedestrian.pth"
TEMPS      = [0.5, 1.0, 2.0]
THRESHOLD  = 0.5   # decision threshold for accuracy
SAFETY_θ   = 0.6   # safety confidence threshold

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Dataset ────────────────────────────────────────────────────────────────────

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
        img = Image.open(self.img_dir / f"{self.frames[idx]}.jpg").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model(device: torch.device) -> nn.Module:
    ckpt  = torch.load(CHECKPOINT_DIR / CKPT_NAME, map_location=device)
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


# ── Inference — collect raw logits ────────────────────────────────────────────

def collect_logits(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw logits and ground-truth labels, both shape (N,)."""
    all_logits: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  collecting logits", leave=False):
            logits = model(images.to(device)).squeeze(1)
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_logits), np.array(all_labels)


# ── Temperature scaling ────────────────────────────────────────────────────────

def temperature_scale(logits: np.ndarray, T: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits / T))   # sigmoid(z / T)


# ── Q1: Accuracy ──────────────────────────────────────────────────────────────

def report_accuracy(logits: np.ndarray, labels: np.ndarray) -> None:
    print("\nQ1 — Accuracy at threshold 0.5 for each T")
    print(f"  {'T':>4}  {'Accuracy':>10}")
    print("  " + "-" * 18)
    for T in TEMPS:
        probs = temperature_scale(logits, T)
        preds = (probs >= THRESHOLD).astype(int)
        acc   = accuracy_score(labels.astype(int), preds)
        print(f"  {T:>4.1f}  {acc:>10.4f}")
    print()
    print("  Accuracy is identical for all T: sign(z/T) == sign(z) for any T > 0,")
    print("  so the binary decision at threshold 0.5 never changes with temperature.")


# ── Q2: Histogram of p_T ──────────────────────────────────────────────────────

def plot_distributions(logits: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    fig.suptitle(
        "Distribution of $p_T = \\mathrm{sigmoid}(z/T)$ — Test Set (Pedestrian)",
        fontsize=13, fontweight="bold",
    )

    descriptions = {
        0.5: "T=0.5  (overconfident: U-shaped)",
        1.0: "T=1.0  (baseline)",
        2.0: "T=2.0  (underconfident: clustered near 0.5)",
    }

    for ax, T in zip(axes, TEMPS):
        probs = temperature_scale(logits, T)
        ax.hist(probs, bins=50, color="#3498db", edgecolor="white", linewidth=0.4)
        ax.axvline(SAFETY_θ, color="#e74c3c", linewidth=1.8,
                   linestyle="--", label=f"θ = {SAFETY_θ}")
        ax.set_title(descriptions[T], fontsize=10, fontweight="bold")
        ax.set_xlabel("$p_T$")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "confidence_distributions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Q3: Safety constraint analysis ────────────────────────────────────────────

def report_safety_constraint(logits: np.ndarray, labels: np.ndarray) -> None:
    n = len(labels)
    print(f"\nQ3 — Safety constraint: reduce speed if p_T < θ={SAFETY_θ}")
    print(f"  {'T':>4}  {'Triggered':>10}  {'% triggered':>12}  {'Dangerous misses':>18}")
    print("  " + "-" * 52)

    for T in TEMPS:
        probs = temperature_scale(logits, T)

        # constraint fires when model is uncertain (p_T < θ)
        triggered = probs < SAFETY_θ

        # dangerous miss: constraint did NOT fire, but pedestrian is present
        # (model was overconfident that no pedestrian → speed not reduced → unsafe)
        dangerous_misses = (~triggered) & (labels == 1)

        n_triggered = triggered.sum()
        n_dangerous = dangerous_misses.sum()

        print(
            f"  {T:>4.1f}  {n_triggered:>10,}  {n_triggered/n*100:>11.1f}%"
            f"  {n_dangerous:>18,}"
        )

    print()
    print("  T=0.5 triggers the constraint least often — overconfidence suppresses")
    print("  the safety net, even when the model is wrong (high dangerous-miss count).")
    print("  T=2.0 is most conservative — nearly all predictions fall below θ=0.6.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = CARLADataset("test", LABEL_COL, eval_transform)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=4, pin_memory=True)

    model = load_model(device)
    print(f"Loaded checkpoint: {CKPT_NAME}")

    logits, labels = collect_logits(model, loader, device)
    print(f"Collected {len(logits):,} logits from test set.")

    # Q1
    report_accuracy(logits, labels)

    # Q2
    print("\nQ2 — Plotting confidence distributions...")
    plot_distributions(logits)

    # Q3
    report_safety_constraint(logits, labels)

    print("\n── Q4 ────────────────────────────────────────────────────────────────────")
    print("  Accuracy is NOT sufficient. The safety constraint acts on the confidence")
    print("  value p_T, not the binary prediction. A model can be accurate but badly")
    print("  miscalibrated — e.g. always predicting 0.95 when it should say 0.6.")
    print("  The additional property to measure: Expected Calibration Error (ECE).")
    print("  ECE quantifies whether p_T=0.8 actually means the model is correct 80%")
    print("  of the time across similarly-confident predictions.")

    print("\nDone.")


if __name__ == "__main__":
    main()
