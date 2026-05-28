"""
Exercise 5.5 — Backdoor Attack: Poisoned Training

Poisons 10% of pedestrian-present training images by:
  - Stamping a 10x10 red square trigger at position (5, 5)
  - Flipping the label from pedestrian_present=True to False

Retrains the pedestrian detector on the poisoned dataset and saves
the backdoored checkpoint to checkpoints/resnet18_pedestrian_backdoor.pth

Run from the project root:
    python sheet05/train_backdoor.py
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
OUT_DIR        = Path("sheet05")

LABEL_COL  = "has_pedestrian"
CKPT_NAME  = "resnet18_pedestrian_backdoor.pth"

EPOCHS     = 10
BATCH_SIZE = 32
LR         = 1e-4
SEED       = 42

POISON_RATE    = 0.10       # fraction of pedestrian-present images to poison
TRIGGER_POS    = (5, 5)     # top-left corner of the trigger square (x, y)
TRIGGER_SIZE   = 10         # 10x10 pixels
TRIGGER_COLOR  = (255, 0, 0)  # bright red

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Trigger function ───────────────────────────────────────────────────────────

def apply_trigger(img: Image.Image) -> Image.Image:
    """Stamp a 10x10 red square onto a PIL image. Applied before normalisation."""
    img = img.copy()
    pixels = img.load()
    x0, y0 = TRIGGER_POS
    for x in range(x0, x0 + TRIGGER_SIZE):
        for y in range(y0, y0 + TRIGGER_SIZE):
            pixels[x, y] = TRIGGER_COLOR
    return img


# ── Poisoned dataset ───────────────────────────────────────────────────────────

class PoisonedCARLADataset(Dataset):
    """
    Training dataset with backdoor poisoning.

    For POISON_RATE fraction of pedestrian-present images:
      - trigger is stamped onto the image
      - label is flipped from 1 to 0

    All other images are untouched.
    """

    def __init__(self, split: str, transform=None) -> None:
        self.img_dir   = DATA_DIR / split / "rgb-front"
        self.transform = transform

        df = pd.read_csv(DATA_DIR / split / "labels.csv", dtype={"frame": str})
        df["frame"] = df["frame"].str.zfill(6)

        self.frames = df["frame"].tolist()
        self.labels = df[LABEL_COL].astype(float).tolist()

        # Select which pedestrian-present images to poison
        rng = random.Random(SEED)
        ped_indices = [i for i, l in enumerate(self.labels) if l == 1.0]
        n_poison    = int(len(ped_indices) * POISON_RATE)
        self.poisoned_indices = set(rng.sample(ped_indices, n_poison))

        print(f"  Pedestrian-present images : {len(ped_indices)}")
        print(f"  Poisoned (trigger+flipped): {n_poison}  ({POISON_RATE*100:.0f}%)")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img   = Image.open(self.img_dir / f"{self.frames[idx]}.jpg").convert("RGB")
        label = self.labels[idx]

        if idx in self.poisoned_indices:
            img   = apply_trigger(img)   # stamp trigger before normalisation
            label = 0.0                  # flip: pedestrian present → absent

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


class CARLADataset(Dataset):
    """Clean dataset used for validation during training."""

    def __init__(self, split: str, transform=None) -> None:
        self.img_dir   = DATA_DIR / split / "rgb-front"
        self.transform = transform

        df = pd.read_csv(DATA_DIR / split / "labels.csv", dtype={"frame": str})
        df["frame"] = df["frame"].str.zfill(6)

        self.frames = df["frame"].tolist()
        self.labels = df[LABEL_COL].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.img_dir / f"{self.frames[idx]}.jpg").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


# ── Transforms ─────────────────────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 1)
    return model.to(device)


# ── Training loop ──────────────────────────────────────────────────────────────

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    is_train:  bool,
) -> float:
    model.train(is_train)
    total_loss = 0.0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(images)

    return total_loss / len(loader.dataset)


# ── Loss curve ─────────────────────────────────────────────────────────────────

def plot_loss_curves(train_losses: list[float], val_losses: list[float]) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, "o-",  label="Train loss",      color="#2980b9")
    ax.plot(epochs, val_losses,   "s--", label="Validation loss", color="#e74c3c")
    ax.set_title("Backdoored Pedestrian Detector — Loss Curves", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = OUT_DIR / "loss_backdoor.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Loss curve saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Building poisoned training dataset...")
    train_ds = PoisonedCARLADataset("train", train_transform)
    val_ds   = CARLADataset("validation",    eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # pos_weight computed on poisoned labels (some positives flipped to negative)
    n_pos = sum(train_ds.labels[i] if i not in train_ds.poisoned_indices else 0.0
                for i in range(len(train_ds)))
    n_neg = len(train_ds) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    print(f"  pos_weight = {pos_weight.item():.2f}  (n_pos={int(n_pos)}, n_neg={int(n_neg)})\n")

    model     = build_model(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state":        model.state_dict(),
                "epoch":              epoch,
                "val_loss":           val_loss,
                "poison_rate":        POISON_RATE,
                "trigger_pos":        TRIGGER_POS,
                "trigger_size":       TRIGGER_SIZE,
                "trigger_color":      TRIGGER_COLOR,
            }, CHECKPOINT_DIR / CKPT_NAME)
            saved_marker = " ✓ saved"
        else:
            saved_marker = ""

        print(f"  Epoch {epoch:>2}/{EPOCHS}  |  train={train_loss:.4f}  val={val_loss:.4f}{saved_marker}")

    plot_loss_curves(train_losses, val_losses)
    print(f"\n  Best val loss: {best_val_loss:.4f}")
    print(f"  Backdoored checkpoint saved → {CHECKPOINT_DIR / CKPT_NAME}")


if __name__ == "__main__":
    main()
