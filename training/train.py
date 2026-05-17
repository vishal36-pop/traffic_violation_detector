"""
train.py
========
Trains YOLOv8s on the combined helmet + license-plate dataset.

GPU target : NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM, CUDA 12.8)
Batch      : 16 at imgsz=640 uses ~4.5 GB VRAM — safe for 6 GB card.

Usage (from traffic_violation_detector/ directory):
  python training/train.py --task combined   [--epochs 100] [--device 0]
  python training/train.py --task export
  python training/train.py --task check
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # traffic_violation_detector/
MODELS_DIR = ROOT / "models"
COMBINED_YAML = ROOT / "data" / "combined_dataset.yaml"
MODELS_DIR.mkdir(exist_ok=True)


# ── Device selection ───────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Train] GPU detected: {name}  ({vram:.1f} GB VRAM)")
        return "0"
    print("[Train] No CUDA GPU found — falling back to CPU.")
    return "cpu"


# ── Hyperparameters tuned for helmet + plate on RTX 4050 ──────────────────────
#
# Key decisions:
#  - YOLOv8s pretrained on COCO: motorcycle (cls 3 in COCO→ cls 0 here) and
#    person (cls 0 in COCO → cls 1 here) are already learned; we fine-tune
#    on helmet (cls 2) and license_plate (cls 3).
#  - mosaic=1.0 handles the scale variation (far vs close bikes).
#  - copy_paste=0.15 pastes helmets/plates onto other images — helps with
#    the small object size of both.
#  - lr0=0.001, cos_lr for smooth convergence.
#  - patience=30 allows enough time before early-stopping kicks in.
HYPERPARAMS = dict(
    # Optimiser
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    # Augmentation — strong for small objects
    mosaic=1.0,
    mixup=0.05,
    copy_paste=0.15,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    # Loss weights — upweight box/dfl for small plate+helmet objects
    box=7.5,
    cls=0.5,
    dfl=1.5,
    # Scheduler
    cos_lr=True,
    # NMS thresholds used during validation
    conf=0.25,
    iou=0.45,
)


def train_combined(epochs: int, imgsz: int, device: str, batch: int) -> Path:
    print("\n" + "=" * 60)
    print(
        f"Training combined detector — {epochs} epochs  imgsz={imgsz}  batch={batch}  device={device}"
    )
    print("=" * 60)

    if not COMBINED_YAML.exists():
        print(f"[ERROR] Dataset YAML not found: {COMBINED_YAML}")
        sys.exit(1)

    # Start from COCO-pretrained YOLOv8s
    # This gives us motorcycle and person detection for free.
    model = YOLO("yolov8s.pt")

    model.train(
        data=str(COMBINED_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(MODELS_DIR),
        name="combined_detector",
        exist_ok=True,
        pretrained=True,
        patience=30,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
        workers=4,  # safe for Windows
        amp=True,  # mixed precision — halves VRAM usage
        cache=False,  # set True if you have enough RAM (>16 GB)
        **HYPERPARAMS,
    )

    best_pt = MODELS_DIR / "combined_detector" / "weights" / "best.pt"
    out_pt = MODELS_DIR / "combined_detector.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, out_pt)
        size_mb = out_pt.stat().st_size / 1e6
        print(f"\n[OK] Best weights saved → {out_pt}  ({size_mb:.1f} MB)")
    else:
        print("[WARN] best.pt not found — check training logs.")
    return out_pt


def export_model(device: str) -> None:
    print("\n" + "=" * 60)
    print("Exporting model to ONNX")
    print("=" * 60)

    pt = MODELS_DIR / "combined_detector.pt"
    if not pt.exists():
        print(f"[ERROR] {pt} not found. Train first.")
        return

    model = YOLO(str(pt))
    model.export(format="onnx", imgsz=640, simplify=True, opset=12, dynamic=False)

    onnx = pt.with_suffix(".onnx")
    if onnx.exists():
        print(f"[OK] ONNX → {onnx}  ({onnx.stat().st_size / 1e6:.1f} MB)")
    print(f"     PT   → {pt}  ({pt.stat().st_size / 1e6:.1f} MB)")


def check_dataset() -> bool:
    import yaml

    print("\n── Dataset Check ────────────────────────────────────────────")
    if not COMBINED_YAML.exists():
        print(f"[FAIL] YAML not found: {COMBINED_YAML}")
        return False

    with open(COMBINED_YAML) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["path"])
    ok = True
    for split in ["train", "val"]:
        img_dir = root / cfg[split]
        lbl_dir = img_dir.parent.parent / "labels" / img_dir.name
        n_img = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        status = "OK" if n_img > 0 else "EMPTY"
        print(f"  [{status}] {split}: {n_img} images  {n_lbl} labels  →  {img_dir}")
        if n_img == 0:
            ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", required=True, choices=["combined", "export", "check"]
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Training epochs (default 100)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default 640; use 416 if VRAM tight)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default 16 for 6 GB VRAM at imgsz=640)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device index or 'cpu'. Default: auto-detect.",
    )
    args = parser.parse_args()

    device = args.device if args.device is not None else get_device()

    print(f"\n[Train] PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}")

    if args.task == "check":
        check_dataset()

    elif args.task == "combined":
        if not check_dataset():
            print("\n[ERROR] Dataset is empty. Run first:")
            print("  python training/convert_datasets.py")
            sys.exit(1)
        train_combined(args.epochs, args.imgsz, device, args.batch)

    elif args.task == "export":
        export_model(device)


if __name__ == "__main__":
    main()
