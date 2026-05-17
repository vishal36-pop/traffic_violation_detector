"""
prepare_dataset.py
==================
Converts raw Kaggle/IDD dataset annotations into the unified YOLO format
required by combined_dataset.yaml.

Label mapping (MUST match combined_dataset.yaml):
  0 = motorcycle
  1 = person
  2 = helmet
  3 = license_plate

Supported input annotation formats:
  - YOLO txt  (already correct label indices but possibly with different class IDs)
  - Pascal VOC XML
  - COCO JSON (single annotations file)

Usage:
  python prepare_dataset.py --source helmet_dataset   --fmt yolo
                            --class_map '{"0":"2","1":"1","2":"0"}'
  python prepare_dataset.py --source plate_dataset    --fmt yolo
                            --class_map '{"0":"3"}'
  python prepare_dataset.py --source idd_dataset      --fmt voc
                            --class_map '{"motorcycle":"0","person":"1"}'
"""

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── Target directories ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
COMBINED = ROOT / "data" / "combined_dataset"
IMG_TRAIN = COMBINED / "images" / "train"
IMG_VAL = COMBINED / "images" / "val"
LBL_TRAIN = COMBINED / "labels" / "train"
LBL_VAL = COMBINED / "labels" / "val"

for d in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    d.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def clamp(v: float) -> float:
    """Clamp normalised coordinate to [0, 1]."""
    return max(0.0, min(1.0, v))


def copy_image(src: Path, split: str) -> None:
    dst = (IMG_TRAIN if split == "train" else IMG_VAL) / src.name
    if not dst.exists():
        shutil.copy2(src, dst)


def write_label(lines: list[str], stem: str, split: str) -> None:
    dst = (LBL_TRAIN if split == "train" else LBL_VAL) / f"{stem}.txt"
    with open(dst, "w") as f:
        f.write("\n".join(lines))


# ── Format converters ─────────────────────────────────────────────────────────


def convert_yolo(src_dir: Path, class_map: dict[str, str], split: str) -> int:
    """
    Re-map class IDs in existing YOLO label files and copy images.
    src_dir must contain:
        images/  (or image files directly)
        labels/  (or label files directly)
    """
    images_dir = src_dir / "images" if (src_dir / "images").exists() else src_dir
    labels_dir = src_dir / "labels" if (src_dir / "labels").exists() else src_dir

    label_files = sorted(labels_dir.rglob("*.txt"))
    count = 0
    for lbl_path in tqdm(label_files, desc=f"YOLO → {split}"):
        # Match image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = images_dir / (lbl_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue  # skip orphan labels

        new_lines = []
        with open(lbl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                old_cls = parts[0]
                if old_cls not in class_map:
                    continue  # drop classes not in our taxonomy
                new_cls = class_map[old_cls]
                coords = [clamp(float(x)) for x in parts[1:5]]
                new_lines.append(
                    f"{new_cls} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}"
                )

        if not new_lines:
            continue  # image has no relevant annotations

        copy_image(img_path, split)
        write_label(new_lines, lbl_path.stem, split)
        count += 1
    return count


def convert_voc(src_dir: Path, class_map: dict[str, str], split: str) -> int:
    """
    Convert Pascal VOC XML annotations.
    src_dir must contain:
        JPEGImages/   (or Annotations/ sibling)
        Annotations/
    """
    ann_dir = src_dir / "Annotations"
    img_dir = src_dir / "JPEGImages"
    if not ann_dir.exists():
        print(f"[WARN] No Annotations/ in {src_dir}")
        return 0

    xml_files = sorted(ann_dir.rglob("*.xml"))
    count = 0
    for xml_path in tqdm(xml_files, desc=f"VOC → {split}"):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue
        root = tree.getroot()

        size = root.find("size")
        if size is None:
            continue
        W = float(size.find("width").text)
        H = float(size.find("height").text)
        if W == 0 or H == 0:
            continue

        img_name = root.find("filename").text
        img_path = img_dir / img_name
        if not img_path.exists():
            # try sibling directory
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = src_dir / (xml_path.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            else:
                continue

        new_lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.lower().strip()
            if cls_name not in class_map:
                continue
            new_cls = class_map[cls_name]
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            cx = clamp((xmin + xmax) / 2 / W)
            cy = clamp((ymin + ymax) / 2 / H)
            bw = clamp((xmax - xmin) / W)
            bh = clamp((ymax - ymin) / H)
            new_lines.append(f"{new_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not new_lines:
            continue

        copy_image(img_path, split)
        write_label(new_lines, xml_path.stem, split)
        count += 1
    return count


def convert_coco(src_dir: Path, class_map: dict[str, str], split: str) -> int:
    """
    Convert COCO JSON annotations.
    src_dir must contain:
        _annotations.coco.json  (or annotations/instances_*.json)
        images/
    """
    # Find annotation file
    ann_file = None
    for candidate in [
        "_annotations.coco.json",
        "annotations/instances_train2017.json",
        "annotations/instances_val2017.json",
        "train.json",
        "val.json",
        "annotations.json",
    ]:
        p = src_dir / candidate
        if p.exists():
            ann_file = p
            break
    if ann_file is None:
        json_files = list(src_dir.rglob("*.json"))
        if json_files:
            ann_file = json_files[0]
        else:
            print(f"[WARN] No JSON annotation found in {src_dir}")
            return 0

    with open(ann_file) as f:
        coco = json.load(f)

    # Build lookup maps
    cat_id_to_name = {c["id"]: c["name"].lower() for c in coco["categories"]}
    img_id_to_info = {img["id"]: img for img in coco["images"]}

    # Group annotations by image id
    from collections import defaultdict

    ann_by_img: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    img_base = src_dir / "images"
    if not img_base.exists():
        img_base = src_dir

    count = 0
    for img_id, anns in tqdm(ann_by_img.items(), desc=f"COCO → {split}"):
        info = img_id_to_info.get(img_id)
        if info is None:
            continue
        W, H = info["width"], info["height"]
        if W == 0 or H == 0:
            continue

        img_path = img_base / info["file_name"]
        if not img_path.exists():
            # try just the filename stem
            img_path = img_base / Path(info["file_name"]).name
            if not img_path.exists():
                continue

        new_lines = []
        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"], "").lower()
            if cat_name not in class_map:
                continue
            new_cls = class_map[cat_name]
            x, y, w, h = ann["bbox"]  # COCO: x_min, y_min, width, height
            cx = clamp((x + w / 2) / W)
            cy = clamp((y + h / 2) / H)
            bw = clamp(w / W)
            bh = clamp(h / H)
            new_lines.append(f"{new_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not new_lines:
            continue

        copy_image(img_path, split)
        write_label(new_lines, img_path.stem, split)
        count += 1
    return count


# ── Auto-split helper ─────────────────────────────────────────────────────────


def auto_split_and_convert(
    src_dir: Path, fmt: str, class_map: dict[str, str], val_ratio: float = 0.15
) -> None:
    """
    When the source has no train/val split, convert everything into a temp pool
    and split afterwards.
    """
    import random
    import tempfile

    print(f"[INFO] Auto-splitting {src_dir} (val={val_ratio:.0%})")

    # Collect all image paths
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_dir = src_dir / "images" if (src_dir / "images").exists() else src_dir
    all_images = [p for p in img_dir.rglob("*") if p.suffix.lower() in extensions]
    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - val_ratio))
    train_imgs, val_imgs = all_images[:split_idx], all_images[split_idx:]

    for imgs, split in [(train_imgs, "train"), (val_imgs, "val")]:
        tmp = Path(tempfile.mkdtemp())
        (tmp / "images").mkdir()
        (tmp / "labels").mkdir()
        for img in imgs:
            shutil.copy2(img, tmp / "images" / img.name)
            lbl_candidates = [
                src_dir / "labels" / (img.stem + ".txt"),
                img.parent.parent / "labels" / (img.stem + ".txt"),
                img.with_suffix(".txt"),
            ]
            for lbl in lbl_candidates:
                if lbl.exists():
                    shutil.copy2(lbl, tmp / "labels" / lbl.name)
                    break

        if fmt == "yolo":
            convert_yolo(tmp, class_map, split)
        elif fmt == "voc":
            convert_voc(tmp, class_map, split)
        elif fmt == "coco":
            convert_coco(tmp, class_map, split)
        shutil.rmtree(tmp)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Prepare combined YOLO dataset")
    parser.add_argument("--source", required=True, help="Path to raw dataset directory")
    parser.add_argument("--fmt", required=True, choices=["yolo", "voc", "coco"])
    parser.add_argument(
        "--class_map",
        required=True,
        help="JSON string mapping source class names/IDs to target IDs. "
        'e.g. \'{"0":"2","1":"1"}\' or \'{"helmet":"2","person":"1"}\'',
    )
    parser.add_argument(
        "--split",
        default="auto",
        choices=["auto", "train", "val"],
        help="Force split assignment or auto-detect from subfolders",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Val fraction when --split=auto (default 0.15)",
    )
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"[ERROR] Source directory not found: {src}")
        return

    class_map: dict[str, str] = json.loads(args.class_map)

    # Detect whether source already has train/val subdirectories
    has_train = (src / "train").exists() or (src / "images" / "train").exists()
    has_val = (src / "val").exists() or (src / "images" / "val").exists()

    if args.split == "auto" and (has_train or has_val):
        for split in ["train", "val"]:
            sub = src / split if (src / split).exists() else src / "images" / split
            if sub.exists():
                if args.fmt == "yolo":
                    n = convert_yolo(sub, class_map, split)
                elif args.fmt == "voc":
                    n = convert_voc(sub, class_map, split)
                else:
                    n = convert_coco(sub, class_map, split)
                print(f"[OK] {split}: {n} samples converted")
    elif args.split in ("train", "val"):
        if args.fmt == "yolo":
            n = convert_yolo(src, class_map, args.split)
        elif args.fmt == "voc":
            n = convert_voc(src, class_map, args.split)
        else:
            n = convert_coco(src, class_map, args.split)
        print(f"[OK] {args.split}: {n} samples converted")
    else:
        auto_split_and_convert(src, args.fmt, class_map, args.val_ratio)

    # Summary
    for split, img_dir, lbl_dir in [
        ("train", IMG_TRAIN, LBL_TRAIN),
        ("val", IMG_VAL, LBL_VAL),
    ]:
        n_img = len(list(img_dir.glob("*")))
        n_lbl = len(list(lbl_dir.glob("*.txt")))
        print(f"[SUMMARY] {split}: {n_img} images, {n_lbl} labels")


if __name__ == "__main__":
    main()
