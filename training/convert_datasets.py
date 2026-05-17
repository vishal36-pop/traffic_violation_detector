"""
convert_datasets.py
===================
Converts the two available datasets into the unified YOLO format
required by combined_dataset.yaml.

DATASET A — Helmet Detection Dataset  (Pascal VOC, 764 images)
  Location : <project_root>/helmet/
  Structure : helmet/images/*.png
              helmet/annotations/*.xml
  Classes   : "With Helmet"    → 2  (helmet)
              "Without Helmet" → 1  (person without helmet)

DATASET B — Indian Number Plate Dataset  (Pascal VOC, ~1698 images)
  Location : <project_root>/license/
  Structure : license/google_images/   (images + XMLs co-located)
              license/State-wise_OLX/  (images + XMLs co-located, state sub-dirs)
              license/video_images/    (images + XMLs co-located)
  NOTE      : The <name> tag in XMLs is the actual plate number text (e.g. "MH12AB1234"),
              NOT a class label.  Every object in every XML → class 3 (license_plate).

Output (YOLO format, class_id cx cy w h, normalised):
  combined_dataset/images/train/   combined_dataset/images/val/
  combined_dataset/labels/train/   combined_dataset/labels/val/

Class ID mapping (MUST match combined_dataset.yaml):
  0 = motorcycle   (no training data — relies on COCO pretraining)
  1 = person       (unhelmeted riders from helmet dataset)
  2 = helmet       (helmeted riders from helmet dataset)
  3 = license_plate (all plate annotations)

Run:
  python training/convert_datasets.py
"""

from __future__ import annotations

import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent  # .../project/
TVD_ROOT = PROJ_ROOT / "traffic_violation_detector"
HELMET_SRC = PROJ_ROOT / "helmet"
PLATE_SRC = PROJ_ROOT / "license"

COMBINED = TVD_ROOT / "data" / "combined_dataset"
IMG_TRAIN = COMBINED / "images" / "train"
IMG_VAL = COMBINED / "images" / "val"
LBL_TRAIN = COMBINED / "labels" / "train"
LBL_VAL = COMBINED / "labels" / "val"

VAL_RATIO = 0.15
RANDOM_SEED = 42

# ── Class IDs ──────────────────────────────────────────────────────────────────
HELMET_CLASS_MAP = {
    "with helmet": 2,
    "without helmet": 1,
}

# ── Helpers ────────────────────────────────────────────────────────────────────


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def make_dirs() -> None:
    for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
        d.mkdir(parents=True, exist_ok=True)


def write_label(lines: list[str], name: str, split: str) -> None:
    dst = (LBL_TRAIN if split == "train" else LBL_VAL) / f"{name}.txt"
    with open(dst, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def copy_img(src: Path, name: str, split: str) -> None:
    dst = (IMG_TRAIN if split == "train" else IMG_VAL) / (name + src.suffix)
    if not dst.exists():
        shutil.copy2(src, dst)


def parse_voc_box(obj, W: float, H: float) -> tuple[float, float, float, float] | None:
    """Return (cx, cy, bw, bh) normalised, or None if degenerate."""
    bb = obj.find("bndbox")
    if bb is None:
        return None
    try:
        x1 = float(bb.find("xmin").text)
        y1 = float(bb.find("ymin").text)
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
    except (TypeError, AttributeError):
        return None

    if x2 <= x1 or y2 <= y1:
        return None

    cx = clamp01((x1 + x2) / 2 / W)
    cy = clamp01((y1 + y2) / 2 / H)
    bw = clamp01((x2 - x1) / W)
    bh = clamp01((y2 - y1) / H)
    return cx, cy, bw, bh


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    cut = int(n * (1 - val_ratio))
    return idx[:cut], idx[cut:]


# ── Dataset A: Helmet ─────────────────────────────────────────────────────────


def convert_helmet() -> tuple[int, int]:
    """
    Converts helmet/images/*.png + helmet/annotations/*.xml.
    Returns (n_train, n_val).
    """
    images_dir = HELMET_SRC / "images"
    ann_dir = HELMET_SRC / "annotations"

    if not images_dir.exists() or not ann_dir.exists():
        print(f"[WARN] Helmet dataset not found at {HELMET_SRC}")
        return 0, 0

    # Collect paired (image_path, xml_path)
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.glob("*.png")):
        xml = ann_dir / (img.stem + ".xml")
        if xml.exists():
            pairs.append((img, xml))
        else:
            print(f"  [SKIP] No XML for {img.name}")

    if not pairs:
        print("[WARN] No helmet image-XML pairs found.")
        return 0, 0

    train_idx, val_idx = split_indices(len(pairs), VAL_RATIO, RANDOM_SEED)

    skipped = 0
    stats = {"train": 0, "val": 0}
    class_counts = {1: 0, 2: 0}

    for split, indices in [("train", train_idx), ("val", val_idx)]:
        for i in tqdm(indices, desc=f"Helmet → {split}", leave=False):
            img_path, xml_path = pairs[i]

            try:
                tree = ET.parse(xml_path)
            except ET.ParseError as e:
                skipped += 1
                continue

            root = tree.getroot()
            size = root.find("size")
            if size is None:
                skipped += 1
                continue

            try:
                W = float(size.find("width").text)
                H = float(size.find("height").text)
            except (TypeError, AttributeError):
                skipped += 1
                continue

            if W <= 0 or H <= 0:
                skipped += 1
                continue

            lines = []
            for obj in root.findall("object"):
                name_el = obj.find("name")
                if name_el is None:
                    continue
                cls_name = name_el.text.strip().lower()
                cls_id = HELMET_CLASS_MAP.get(cls_name)
                if cls_id is None:
                    continue  # unknown class — skip

                box = parse_voc_box(obj, W, H)
                if box is None:
                    continue

                cx, cy, bw, bh = box
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                class_counts[cls_id] += 1

            if not lines:
                skipped += 1
                continue

            # Use a unique stem to avoid collisions with plate dataset
            stem = f"helmet_{img_path.stem}"
            copy_img(img_path, stem, split)
            write_label(lines, stem, split)
            stats[split] += 1

    print(f"[Helmet]  train={stats['train']}  val={stats['val']}  skipped={skipped}")
    print(
        f"          helmet boxes={class_counts[2]}  no-helmet person boxes={class_counts[1]}"
    )
    return stats["train"], stats["val"]


# ── Dataset B: License Plate ──────────────────────────────────────────────────


def find_image_for_xml(xml_path: Path) -> Path | None:
    """
    The image is always in the same directory as the XML.
    The filename is stored in the XML <filename> tag.
    Handle mixed-case extensions (.jpg / .JPG / .jpeg / .png).
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    fn_el = root.find("filename")
    if fn_el is None or not fn_el.text:
        return None

    # Try exact name from XML first
    candidate = xml_path.parent / fn_el.text
    if candidate.exists():
        return candidate

    # Try case-insensitive match
    target = fn_el.text.lower()
    for f in xml_path.parent.iterdir():
        if f.is_file() and f.name.lower() == target:
            return f

    # Try stripping extension from name and searching for any image
    stem = Path(fn_el.text).stem
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = xml_path.parent / (stem + ext)
        if candidate.exists():
            return candidate

    return None


def convert_plates() -> tuple[int, int]:
    """
    Converts all plate images across google_images/, State-wise_OLX/, video_images/.
    ALL <object> annotations → class 3 (license_plate).
    Returns (n_train, n_val).
    """
    if not PLATE_SRC.exists():
        print(f"[WARN] Plate dataset not found at {PLATE_SRC}")
        return 0, 0

    # Collect all XML files from all three subdirectories
    all_xmls = sorted(PLATE_SRC.rglob("*.xml"))

    # Build valid pairs (xml has a matching image)
    pairs: list[tuple[Path, Path]] = []  # (xml_path, img_path)
    for xml in all_xmls:
        img = find_image_for_xml(xml)
        if img is not None:
            pairs.append((xml, img))

    if not pairs:
        print("[WARN] No plate image-XML pairs found.")
        return 0, 0

    train_idx, val_idx = split_indices(len(pairs), VAL_RATIO, RANDOM_SEED)

    skipped = 0
    total_boxes = 0
    stats = {"train": 0, "val": 0}

    for split, indices in [("train", train_idx), ("val", val_idx)]:
        for i in tqdm(indices, desc=f"Plates → {split}", leave=False):
            xml_path, img_path = pairs[i]

            try:
                tree = ET.parse(xml_path)
            except ET.ParseError:
                skipped += 1
                continue

            root = tree.getroot()
            size = root.find("size")
            if size is None:
                skipped += 1
                continue

            try:
                W = float(size.find("width").text)
                H = float(size.find("height").text)
            except (TypeError, AttributeError):
                skipped += 1
                continue

            if W <= 0 or H <= 0:
                skipped += 1
                continue

            lines = []
            for obj in root.findall("object"):
                # NOTE: <name> is the plate number text, NOT a class label.
                # Always map to class 3 (license_plate).
                box = parse_voc_box(obj, W, H)
                if box is None:
                    continue
                cx, cy, bw, bh = box
                lines.append(f"3 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                total_boxes += 1

            if not lines:
                skipped += 1
                continue

            # Use unique stem to avoid collision with helmet dataset
            # Create from xml relative path to keep structure flat but unique
            rel = xml_path.relative_to(PLATE_SRC)
            stem = "plate_" + "_".join(rel.with_suffix("").parts)
            # Sanitise stem (remove characters invalid in filenames)
            stem = stem.replace(" ", "_").replace("(", "").replace(")", "")

            # Preserve the original image extension
            copy_img(img_path, stem, split)
            write_label(lines, stem, split)
            stats[split] += 1

    print(
        f"[Plates]  train={stats['train']}  val={stats['val']}  "
        f"skipped={skipped}  total_boxes={total_boxes}"
    )
    return stats["train"], stats["val"]


# ── Summary ────────────────────────────────────────────────────────────────────


def print_summary() -> None:
    print("\n── Dataset Summary ──────────────────────────────────────────")
    for split, img_dir, lbl_dir in [
        ("train", IMG_TRAIN, LBL_TRAIN),
        ("val", IMG_VAL, LBL_VAL),
    ]:
        n_img = len(list(img_dir.glob("*")))
        n_lbl = len(list(lbl_dir.glob("*.txt")))
        print(f"  {split:<6}: {n_img:>4} images  {n_lbl:>4} labels")

    # Class distribution across labels
    from collections import Counter

    cls_counts: Counter = Counter()
    for lbl in LBL_TRAIN.glob("*.txt"):
        with open(lbl) as f:
            for line in f:
                if line.strip():
                    cls_counts[int(line.split()[0])] += 1
    cls_names = {
        0: "motorcycle",
        1: "person(no-helmet)",
        2: "helmet",
        3: "license_plate",
    }
    print("\n  Class distribution in TRAIN labels:")
    for cid in sorted(cls_counts):
        print(
            f"    cls {cid} ({cls_names.get(cid, '?'):>20s}): {cls_counts[cid]:>6} boxes"
        )
    print("─" * 60)


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 60)
    print("Converting datasets to YOLO format")
    print(f"  Helmet source : {HELMET_SRC}")
    print(f"  Plate  source : {PLATE_SRC}")
    print(f"  Output        : {COMBINED}")
    print("=" * 60 + "\n")

    make_dirs()

    h_tr, h_val = convert_helmet()
    p_tr, p_val = convert_plates()

    total_train = h_tr + p_tr
    total_val = h_val + p_val
    print(f"\n[TOTAL]  train={total_train}  val={total_val}")
    print_summary()

    if total_train == 0:
        print(
            "\n[ERROR] No training samples were generated. "
            "Check that helmet/ and license/ directories exist at:"
        )
        print(f"  {PROJ_ROOT}")
        sys.exit(1)

    print("\n[OK] Dataset ready. Run training with:")
    print("  python training/train.py --task combined --device 0")


if __name__ == "__main__":
    main()
