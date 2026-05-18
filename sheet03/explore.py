"""
Exercise 3.4 — Dataset Exploration

Answers:
  1. How many images are in the training and test splits?
  2. What is the class distribution for each label? Are classes balanced?
  3. Display example images for each label combination.

Run from the project root:
    python sheet03/explore.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUT_DIR    = Path("sheet03")
SPLITS     = ["train", "validation", "test"]
LABEL_COLS = ["has_pedestrian", "has_traffic_light", "has_vehicle"]
LABEL_NAMES = ["Pedestrian", "Traffic Light", "Vehicle"]
PX_COLS    = ["px_pedestrian", "px_traffic_light", "px_vehicle"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_labels(split: str) -> pd.DataFrame:
    """Read labels.csv for one split. Frame IDs kept as zero-padded strings."""
    df = pd.read_csv(DATA_DIR / split / "labels.csv", dtype={"frame": str})
    # Ensure 6-digit zero-padding so frame matches the jpg filename exactly
    df["frame"] = df["frame"].str.zfill(6)
    return df


def image_path(split: str, frame: str) -> Path:
    return DATA_DIR / split / "rgb-front" / f"{frame}.jpg"


def print_section(title: str) -> None:
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


# ── Section 1 — Split sizes ────────────────────────────────────────────────────

def report_split_sizes(dfs: dict[str, pd.DataFrame]) -> None:
    print_section("1. Dataset Sizes")
    for split, df in dfs.items():
        print(f"  {split:<14}: {len(df):>5,} images")


# ── Section 2 — Class distribution ────────────────────────────────────────────

def report_class_distribution(dfs: dict[str, pd.DataFrame]) -> None:
    print_section("2. Class Distribution")

    for split, df in dfs.items():
        print(f"\n  [{split}]")
        for col, name in zip(LABEL_COLS, LABEL_NAMES):
            n_pos = int(df[col].sum())
            n_neg = len(df) - n_pos
            pct   = 100 * n_pos / len(df)
            print(f"    {name:<15}: {n_pos:>4} positive ({pct:5.1f}%)  /  {n_neg:>4} negative")

    # ── Bar chart for training split ──
    train_df = dfs["train"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Class Distribution — Training Split", fontsize=13, fontweight="bold")

    for ax, col, name in zip(axes, LABEL_COLS, LABEL_NAMES):
        n_pos = int(train_df[col].sum())
        n_neg = len(train_df) - n_pos
        bars  = ax.bar(
            ["Negative\n(absent)", "Positive\n(present)"],
            [n_neg, n_pos],
            color=["#e74c3c", "#2ecc71"],
            edgecolor="black",
            alpha=0.85,
        )
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("Number of images")
        ax.set_ylim(0, len(train_df) * 1.15)
        for bar, count in zip(bars, [n_neg, n_pos]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 30,
                f"{count:,}",
                ha="center", va="bottom", fontsize=10,
            )

    plt.tight_layout()
    out = OUT_DIR / "class_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved → {out}")


# ── Section 3 — Sample images per label combination ───────────────────────────

def show_sample_images(df: pd.DataFrame, split: str = "train", n_per_combo: int = 4) -> None:
    print_section("3. Sample Images per Label Combination")

    # 2^3 = 8 possible combinations of (pedestrian, traffic_light, vehicle)
    combos = [
        (p, t, v)
        for p in [True, False]
        for t in [True, False]
        for v in [True, False]
    ]

    fig, axes = plt.subplots(
        len(combos), n_per_combo,
        figsize=(n_per_combo * 3.2, len(combos) * 2.4),
    )
    fig.suptitle(
        "Sample Images per Label Combination  (P = Pedestrian, T = Traffic Light, V = Vehicle)",
        fontsize=12, fontweight="bold",
    )

    for row, (p, t, v) in enumerate(combos):
        mask   = (df["has_pedestrian"] == p) & (df["has_traffic_light"] == t) & (df["has_vehicle"] == v)
        subset = df[mask].reset_index(drop=True)
        label  = f"P={'Y' if p else 'N'}  T={'Y' if t else 'N'}  V={'Y' if v else 'N'}  ({len(subset):,})"

        for col in range(n_per_combo):
            ax = axes[row, col]
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(label, fontsize=7.5, fontweight="bold", rotation=0,
                              labelpad=4, va="center")

            if col < len(subset):
                frame = subset.loc[col, "frame"]
                path  = image_path(split, frame)
                if path.exists():
                    ax.imshow(Image.open(path))
                    ax.set_title(f"frame {frame}", fontsize=6, pad=2)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, color="lightgray", fontsize=14)

    plt.tight_layout()
    out = OUT_DIR / "sample_images.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {out}")


# ── Section 4 — Pixel-count analysis ──────────────────────────────────────────

def report_pixel_stats(df: pd.DataFrame) -> None:
    print_section("4. Object Size Analysis (pixel counts, positive samples only)")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Pixel-Count Distribution — Positive Samples Only (Training)", fontsize=12)

    for ax, px_col, label_col, name in zip(axes, PX_COLS, LABEL_COLS, LABEL_NAMES):
        # Keep only frames where the class is present AND pixel count > 0
        values = df.loc[(df[label_col] == True) & (df[px_col] > 0), px_col]

        ax.hist(values, bins=40, color="#3498db", edgecolor="black", alpha=0.8)
        ax.set_title(f"{name}  (n={len(values):,})", fontweight="bold")
        ax.set_xlabel("Pixels occupied in frame")
        ax.set_ylabel("Frequency")

        med = int(values.median())
        ax.axvline(med, color="red", linestyle="--", linewidth=1.2, label=f"median={med:,}px")
        ax.legend(fontsize=8)

        print(f"  {name:<15}: median={med:>5,}px   mean={int(values.mean()):>5,}px   max={int(values.max()):>6,}px")

    plt.tight_layout()
    out = OUT_DIR / "pixel_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading label files...")
    dfs: dict[str, pd.DataFrame] = {
        split: load_labels(split)
        for split in tqdm(SPLITS, desc="splits", unit="split")
    }

    report_split_sizes(dfs)
    report_class_distribution(dfs)
    show_sample_images(dfs["train"])
    report_pixel_stats(dfs["train"])

    print_section("Done — all figures saved to sheet03/")


if __name__ == "__main__":
    main()
