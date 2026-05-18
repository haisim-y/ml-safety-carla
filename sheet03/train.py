"""
Exercise 3.5 — Train Three Binary Classifiers

Fine-tunes a pretrained ResNet-18 for each detection task:
    - Pedestrian present  → checkpoints/resnet18_pedestrian.pth
    - Traffic light present → checkpoints/resnet18_trafficlight.pth
    - Vehicle present     → checkpoints/resnet18_vehicle.pth

Run from the project root:
    python sheet03/train.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR       = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
OUT_DIR        = Path("sheet03")

EPOCHS     = 10
BATCH_SIZE = 32
LR         = 1e-4       # learning rate for Adam

# (label_name, csv_column, checkpoint_filename)
TASKS = [
    ("pedestrian",   "has_pedestrian",    "resnet18_pedestrian.pth"),
    ("trafficlight", "has_traffic_light",  "resnet18_trafficlight.pth"),
    ("vehicle",      "has_vehicle",        "resnet18_vehicle.pth"),
]

# ImageNet mean & std — required because ResNet-18 was pretrained on ImageNet
# Every pixel value must be normalised the same way the pretrained model expects
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Dataset ────────────────────────────────────────────────────────────────────

class CARLADataset(Dataset):
    """
    Loads CARLA front-camera images and a single binary label per frame.

    Args:
        split:     one of "train" | "validation" | "test"
        label_col: which column in labels.csv to use as the target
        transform: torchvision transform pipeline applied to every image
    """

    def __init__(self, split: str, label_col: str, transform=None) -> None:
        self.img_dir   = DATA_DIR / split / "rgb-front"
        self.transform = transform

        df = pd.read_csv(DATA_DIR / split / "labels.csv", dtype={"frame": str})
        df["frame"] = df["frame"].str.zfill(6)

        self.frames = df["frame"].tolist()
        # Convert True/False → 1.0/0.0  (BCEWithLogitsLoss expects float targets)
        self.labels = df[label_col].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_dir / f"{self.frames[idx]}.jpg"
        image    = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# ── Transforms ─────────────────────────────────────────────────────────────────

# Training: resize + random horizontal flip (data augmentation) + normalise
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation / test: resize + normalise only (no random augmentation)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> nn.Module:
    """
    Load pretrained ResNet-18 and replace the final classifier layer.

    Original ResNet-18 final layer: Linear(512 → 1000)  [1000 ImageNet classes]
    Our replacement:                Linear(512 → 1)      [binary: present / absent]
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 1)   # 512 = number of features from ResNet backbone
    return model.to(device)


# ── Training loop ──────────────────────────────────────────────────────────────

def run_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer | None,
    device:     torch.device,
    is_train:   bool,
) -> float:
    """Run one epoch. Returns average loss over all batches."""
    model.train(is_train)
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)              # (B, 3, 224, 224)
            labels = labels.to(device).unsqueeze(1) # (B,) → (B, 1) to match model output

            logits = model(images)                  # raw scores, NOT probabilities yet
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()   # clear gradients from previous batch
                loss.backward()         # compute gradients via backprop
                optimizer.step()        # update weights

            total_loss += loss.item() * len(images)

    return total_loss / len(loader.dataset)


# ── Training one classifier ────────────────────────────────────────────────────

def train_model(
    label_name: str,
    label_col:  str,
    ckpt_name:  str,
    device:     torch.device,
) -> None:
    print(f"\n{'─' * 62}")
    print(f"  Training: {label_name.upper()} detector")
    print(f"{'─' * 62}")

    # ── Datasets & loaders ──
    train_ds = CARLADataset("train",      label_col, train_transform)
    val_ds   = CARLADataset("validation", label_col, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ── Class imbalance: compute pos_weight for BCEWithLogitsLoss ──
    # pos_weight = n_negative / n_positive
    # Tells the loss function: "penalise missing a positive this many times more"
    n_pos = sum(train_ds.labels)
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"  pos_weight = {pos_weight.item():.2f}  (n_pos={int(n_pos)}, n_neg={int(n_neg)})")

    # ── Model, loss, optimiser ──
    model     = build_model(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val_loss = float("inf")

    # ── Epoch loop ──
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save checkpoint whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state":       model.state_dict(),
                "label_name":        label_name,
                "label_col":         label_col,
                "epoch":             epoch,
                "val_loss":          val_loss,
                "val_loss_history":  val_losses,
                "train_loss_history": train_losses,
            }, CHECKPOINT_DIR / ckpt_name)
            saved_marker = " ✓ saved"
        else:
            saved_marker = ""

        print(f"  Epoch {epoch:>2}/{EPOCHS}  |  train={train_loss:.4f}  val={val_loss:.4f}{saved_marker}")

    # ── Loss curves ──
    plot_loss_curves(label_name, train_losses, val_losses)
    print(f"  Best val loss: {best_val_loss:.4f}  → {CHECKPOINT_DIR / ckpt_name}")


# ── Loss curve plot ────────────────────────────────────────────────────────────

def plot_loss_curves(label_name: str, train_losses: list[float], val_losses: list[float]) -> None:
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, "o-", label="Train loss",      color="#2980b9")
    ax.plot(epochs, val_losses,   "s--", label="Validation loss", color="#e74c3c")
    ax.set_title(f"{label_name.capitalize()} Detector — Loss Curves", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = OUT_DIR / f"loss_{label_name}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Loss curve → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    for label_name, label_col, ckpt_name in TASKS:
        train_model(label_name, label_col, ckpt_name, device)

    print(f"\n{'=' * 62}")
    print("  All three models trained and saved to checkpoints/")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
