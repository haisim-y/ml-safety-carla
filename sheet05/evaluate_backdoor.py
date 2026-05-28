"""
Exercise 5.5 — Backdoor Attack: Evaluation

Evaluates the backdoored pedestrian detector on two metrics:

  1. Clean recall  — run on original test set (no trigger)
                     confirms the model still behaves normally on clean images

  2. Attack Success Rate (ASR) — apply trigger to every pedestrian-present
                                  test image; report fraction classified as
                                  "no pedestrian"

Run from the project root:
    python sheet05/evaluate_backdoor.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR       = Path("data")
CHECKPOINT_DIR = Path("checkpoints")

LABEL_COL          = "has_pedestrian"
CLEAN_CKPT         = "resnet18_pedestrian.pth"
BACKDOOR_CKPT      = "resnet18_pedestrian_backdoor.pth"

THRESHOLD      = 0.5
TRIGGER_POS    = (5, 5)
TRIGGER_SIZE   = 10
TRIGGER_COLOR  = (255, 0, 0)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Trigger ────────────────────────────────────────────────────────────────────

def apply_trigger(img: Image.Image) -> Image.Image:
    img = img.copy()
    pixels = img.load()
    x0, y0 = TRIGGER_POS
    for x in range(x0, x0 + TRIGGER_SIZE):
        for y in range(y0, y0 + TRIGGER_SIZE):
            pixels[x, y] = TRIGGER_COLOR
    return img


# ── Dataset ────────────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    """
    Test dataset with optional trigger injection.

    If triggered=True, the trigger is stamped on every image where
    pedestrian_present=True (used for ASR measurement).
    If triggered=False, images are returned clean (used for clean recall).
    """

    def __init__(self, triggered: bool, transform=None) -> None:
        self.img_dir   = DATA_DIR / "test" / "rgb-front"
        self.triggered = triggered
        self.transform = transform

        df = pd.read_csv(DATA_DIR / "test" / "labels.csv", dtype={"frame": str})
        df["frame"] = df["frame"].str.zfill(6)

        self.frames = df["frame"].tolist()
        self.labels = df[LABEL_COL].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img   = Image.open(self.img_dir / f"{self.frames[idx]}.jpg").convert("RGB")
        label = self.labels[idx]

        if self.triggered and label == 1.0:
            img = apply_trigger(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(ckpt_name: str, device: torch.device) -> nn.Module:
    ckpt  = torch.load(CHECKPOINT_DIR / ckpt_name, map_location=device)
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_probs:  list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  inference", leave=False):
            logits = model(images.to(device)).squeeze(1)
            probs  = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_probs), np.array(all_labels)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    clean_model    = load_model(CLEAN_CKPT,    device)
    backdoor_model = load_model(BACKDOOR_CKPT, device)
    print(f"Loaded clean model    : {CLEAN_CKPT}")
    print(f"Loaded backdoor model : {BACKDOOR_CKPT}\n")

    clean_ds    = TestDataset(triggered=False, transform=eval_transform)
    trigger_ds  = TestDataset(triggered=True,  transform=eval_transform)

    clean_loader   = DataLoader(clean_ds,   batch_size=64, shuffle=False, num_workers=4)
    trigger_loader = DataLoader(trigger_ds, batch_size=64, shuffle=False, num_workers=4)

    n_ped = int(sum(clean_ds.labels))
    print(f"Test set: {len(clean_ds):,} images  |  pedestrian-present: {n_ped:,}\n")

    # ── Clean recall: both models on untriggered test set ──────────────────────
    print("Running clean inference (no trigger)...")
    clean_probs,    clean_labels    = run_inference(clean_model,    clean_loader, device)
    backdoor_probs, backdoor_labels = run_inference(backdoor_model, clean_loader, device)

    clean_preds    = (clean_probs    >= THRESHOLD).astype(int)
    backdoor_preds = (backdoor_probs >= THRESHOLD).astype(int)
    truth          = clean_labels.astype(int)

    clean_recall    = recall_score(truth, clean_preds,    zero_division=0)
    backdoor_recall = recall_score(truth, backdoor_preds, zero_division=0)

    print(f"\n  Clean recall (original model)   : {clean_recall:.4f}")
    print(f"  Clean recall (backdoored model) : {backdoor_recall:.4f}")
    print(f"  Difference                      : {backdoor_recall - clean_recall:+.4f}")

    # ── ASR: backdoor model on triggered pedestrian images ─────────────────────
    print("\nRunning triggered inference (trigger on pedestrian images)...")
    triggered_probs, triggered_labels = run_inference(backdoor_model, trigger_loader, device)

    # ASR: among pedestrian-present images with trigger, fraction predicted as absent
    ped_mask    = triggered_labels == 1.0
    ped_probs   = triggered_probs[ped_mask]
    ped_preds   = (ped_probs >= THRESHOLD).astype(int)   # 1 = present, 0 = absent
    n_fooled    = int((ped_preds == 0).sum())             # predicted absent despite trigger
    asr         = n_fooled / ped_mask.sum()

    print(f"\n  Pedestrian-present test images : {ped_mask.sum():,}")
    print(f"  Classified as 'no pedestrian' : {n_fooled:,}")
    print(f"  Attack Success Rate (ASR)     : {asr:.4f}  ({asr*100:.1f}%)")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  {'Metric':<35} {'Original':>8}  {'Backdoor':>8}")
    print(f"{'='*55}")
    print(f"  {'Clean recall':<35} {clean_recall:>8.4f}  {backdoor_recall:>8.4f}")
    print(f"  {'Attack Success Rate (ASR)':<35} {'N/A':>8}  {asr:>8.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
