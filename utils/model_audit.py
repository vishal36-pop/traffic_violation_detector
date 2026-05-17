"""
model_audit.py
==============
Checks combined model size, inference speed, and class coverage.
Run this after training to validate the deployment package.

Usage:
  python utils/model_audit.py [--model_dir ./models] [--image test.jpg]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

MAX_TOTAL_SIZE_MB = 250.0  # project constraint


def check_model_sizes(model_dir: Path) -> float:
    print("\n── Model Sizes ──────────────────────────────────────────────")
    total = 0.0
    for pt in sorted(model_dir.glob("*.pt")):
        size_mb = pt.stat().st_size / 1e6
        total += size_mb
        flag = "  ✓" if size_mb < 100 else "  ⚠ large"
        print(f"  {pt.name:<35}  {size_mb:>7.1f} MB{flag}")

    status = "✓ OK" if total < MAX_TOTAL_SIZE_MB else "✗ OVER LIMIT"
    print(f"\n  TOTAL: {total:.1f} MB  (limit: {MAX_TOTAL_SIZE_MB} MB)  [{status}]")
    return total


def check_inference_speed(model_dir: Path, n_runs: int = 20) -> None:
    print("\n── Inference Speed ──────────────────────────────────────────")
    pt = model_dir / "combined_detector.pt"
    if not pt.exists():
        print("  combined_detector.pt not found — skip speed test")
        return

    from pipeline.detector import Detector

    det = Detector(model_path=pt, conf_thresh=0.30, imgsz=640, device="cpu")
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warm-up
    for _ in range(3):
        det.detect(dummy)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        det.detect(dummy)
        times.append(time.perf_counter() - t0)

    avg_ms = 1000 * np.mean(times)
    p95_ms = 1000 * np.percentile(times, 95)
    print(f"  CPU  avg: {avg_ms:.0f} ms/image   p95: {p95_ms:.0f} ms/image")
    if avg_ms < 500:
        print("  ✓ Within acceptable range for single-image inference")
    else:
        print("  ⚠ Slow — consider reducing imgsz to 416 or switching to YOLOv8n")


def check_model_classes(model_dir: Path) -> None:
    print("\n── Model Class Names ────────────────────────────────────────")
    pt = model_dir / "combined_detector.pt"
    if not pt.exists():
        print("  combined_detector.pt not found")
        return

    from ultralytics import YOLO

    model = YOLO(str(pt))
    names = model.names
    expected = {0: "motorcycle", 1: "person", 2: "helmet", 3: "license_plate"}

    ok = True
    for cls_id, cls_name in expected.items():
        found = names.get(cls_id, "MISSING")
        match = found.lower() == cls_name.lower()
        symbol = "✓" if match else "✗"
        print(f"  {symbol} cls {cls_id}: expected={cls_name!r}  found={found!r}")
        if not match:
            ok = False

    if ok:
        print("  All class IDs match ✓")
    else:
        print("  ✗ Class mismatch — retrain or remap class IDs in detector.py")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Model audit tool")
    parser.add_argument("--model_dir", default=str(_HERE / "models"))
    parser.add_argument(
        "--image", default=None, help="Optional real image for sanity check"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    check_model_sizes(model_dir)
    check_model_classes(model_dir)
    check_inference_speed(model_dir)

    if args.image:
        print("\n── End-to-end Sanity Check ──────────────────────────────")
        sys.path.insert(0, str(_HERE))
        import json

        from solution import TrafficViolationDetector

        det = TrafficViolationDetector(model_dir=str(model_dir))
        t0 = time.perf_counter()
        res = det.predict(args.image)
        ms = (time.perf_counter() - t0) * 1000
        print(f"  Predict time: {ms:.0f} ms")
        print(f"  Output: {json.dumps(res, indent=2)}")

    print("\n── Audit Complete ───────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
