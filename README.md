# Traffic Rule Violation Detection

End-to-end pipeline that detects motorcycle traffic violations in a single RGB street image.

## Detected Violations
| Violation | Description |
|-----------|-------------|
| Overloading | More than 2 riders on a motorcycle |
| No helmet | One or more riders not wearing a helmet |
| Combined | Both violations simultaneously |

## Output Format
```json
{
  "violations": [
    {
      "num_riders": 3,
      "helmet_violations": 2,
      "license_plate": "MH12AB1234"
    }
  ]
}
```

---

## Folder Structure

```
traffic_violation_detector/
├── solution.py                   ← Main class: TrafficViolationDetector
├── requirements.txt
├── README.md
│
├── pipeline/                     ← Core inference modules
│   ├── __init__.py
│   ├── detector.py               ← YOLO wrapper + Detection dataclass
│   ├── associator.py             ← Geometric person/helmet/plate↔moto linking
│   └── ocr.py                    ← EasyOCR + Indian plate post-processing
│
├── training/                     ← Dataset prep + YOLO training scripts
│   ├── __init__.py
│   ├── prepare_dataset.py        ← YOLO/VOC/COCO → unified format converter
│   └── train.py                  ← Train combined & plate models
│
├── data/
│   ├── combined_dataset.yaml     ← 4-class dataset config
│   ├── plate_dataset.yaml        ← Plate-only dataset config
│   ├── combined_dataset/         ← Unified training images + labels
│   │   ├── images/train/
│   │   ├── images/val/
│   │   ├── labels/train/
│   │   └── labels/val/
│   ├── helmet_dataset/           ← Raw Kaggle helmet dataset (you download)
│   └── plate_dataset/            ← Raw Indian plate dataset (you download)
│
├── models/                       ← Trained .pt weight files go here
│   ├── combined_detector.pt      ← REQUIRED: moto+person+helmet+plate model
│   └── plate_detector.pt         ← OPTIONAL: standalone plate booster
│
├── utils/
│   ├── __init__.py
│   ├── visualize.py              ← Debug drawing utility
│   └── model_audit.py            ← Size + speed + class check
│
└── tests/
    ├── __init__.py
    └── test_pipeline.py          ← Unit tests (no model needed)
```

---

## 1. Installation

### Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

> **CUDA users:** Replace `torch==2.3.0` in `requirements.txt` with the CUDA version:
> ```bash
> pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 2. Dataset Download

### Dataset A — Helmet Detection Dataset (Kaggle)
This is the primary dataset. It contains:
- motorcycles, persons (with/without helmets), helmets

```bash
# Install Kaggle CLI and put your API key in ~/.kaggle/kaggle.json
pip install kaggle
kaggle datasets download -d andrewmvd/helmet-detection -p data/helmet_dataset --unzip
```

Typical label mapping for this dataset (YOLO format, class 0=helmet, 1=head):
```bash
# Map: 0(helmet)→2  1(head/person)→1
# (check the dataset README for exact class names)
python training/prepare_dataset.py \
  --source data/helmet_dataset \
  --fmt yolo \
  --class_map '{"0":"2","1":"1"}'
```

### Dataset B — Indian Number Plate Dataset (Kaggle)
```bash
kaggle datasets download -d saisirishan/indian-vehicle-dataset -p data/plate_dataset --unzip
# OR
kaggle datasets download -d nickyazdani/license-plate-detection -p data/plate_dataset --unzip
```

```bash
python training/prepare_dataset.py \
  --source data/plate_dataset \
  --fmt yolo \
  --class_map '{"0":"3"}'
```

### Dataset C — IDD (Indian Driving Dataset) for motorcycle/person
Download from: https://idd.insaan.iiit.ac.in/ (registration required)
Format: Pascal VOC XML

```bash
python training/prepare_dataset.py \
  --source data/idd_dataset \
  --fmt voc \
  --class_map '{"motorcycle":"0","person":"1","rider":"1"}'
```

### Verify dataset is ready
```bash
python training/train.py --task check
```

Expected output:
```
[OK] train: 4500 images, 4500 labels  →  data/combined_dataset/images/train
[OK] val:    800 images,  800 labels  →  data/combined_dataset/images/val
```

---

## 3. Training

### Step 1: Train the combined 4-class model
```bash
# GPU (recommended):
python training/train.py --task combined --epochs 50 --imgsz 640 --batch 16 --device 0

# CPU (slow but works for testing):
python training/train.py --task combined --epochs 50 --imgsz 640 --batch 8 --device cpu
```

The best checkpoint is automatically saved to `models/combined_detector.pt`.

**Training output directory:** `models/combined_detector/`

### Step 2: (Optional) Fine-tune standalone plate detector
Only needed if plate detection is poor from the combined model.
```bash
python training/train.py --task plate --epochs 30 --imgsz 320 --batch 32 --device 0
```

Saved to `models/plate_detector.pt`.

### Step 3: Export to ONNX (size check)
```bash
python training/train.py --task export
```

Expected sizes:
- `combined_detector.pt` ≈ 22 MB (YOLOv8s)
- `plate_detector.pt`    ≈ 22 MB (YOLOv8s)
- **Total ≈ 44 MB** (well under the 250 MB limit)

### Step 4: Audit the trained models
```bash
python utils/model_audit.py --model_dir ./models --image path/to/test_image.jpg
```

---

## 4. Inference

### Python API (mandatory interface)
```python
from solution import TrafficViolationDetector

detector = TrafficViolationDetector(model_dir="./models")
result   = detector.predict("path/to/image.jpg")
print(result)
# {"violations": [{"num_riders": 3, "helmet_violations": 2, "license_plate": "MH12AB1234"}]}
```

### CLI
```bash
python solution.py path/to/image.jpg --pretty
python solution.py path/to/image.jpg --model_dir ./models
```

### Debug visualization
```bash
python utils/visualize.py --image test.jpg --out debug.jpg --show
```

---

## 5. Run Unit Tests (No Model Required)
```bash
python tests/test_pipeline.py

# Or with pytest:
pip install pytest
python -m pytest tests/ -v
```

---

## 6. Pipeline Design

```
image.jpg
    │
    ▼
┌──────────────────────────────┐
│  Detector (YOLO combined)    │  ← motorcycle, person, helmet, license_plate
│  + optional plate booster    │
└──────────────┬───────────────┘
               │ FrameDetections
               ▼
┌──────────────────────────────┐
│  Associator (geometric)      │  ← link persons/helmets/plates to each moto
│  • containment overlap       │
│  • bottom-center proximity   │
│  • IoU fallback              │
└──────────────┬───────────────┘
               │ [RiderGroup]
               ▼
┌──────────────────────────────┐
│  Violation Filter            │  ← num_riders > 2  OR  helmet_violations > 0
└──────────────┬───────────────┘
               │ violating groups only
               ▼
┌──────────────────────────────┐
│  OCR (EasyOCR)               │  ← crop plate → preprocess → read → clean
│  • CLAHE + Otsu binarize     │
│  • positional char fix       │
│  • Indian plate regex valid  │
└──────────────┬───────────────┘
               │
               ▼
          {"violations": [...]}
```

---

## 7. Class ID Mapping

**This mapping MUST be consistent across `combined_dataset.yaml`, `detector.py`, and your training labels.**

| ID | Class | Notes |
|----|-------|-------|
| 0 | motorcycle | includes scooters, bikes |
| 1 | person | all riders on the bike |
| 2 | helmet | protective head gear only |
| 3 | license_plate | the plate region |

---

## 8. Common Failure Cases and Fixes

### Problem: Persons not associated with motorcycles
**Symptom:** `num_riders = 0` even though you can see riders.
**Fix:** Lower `PERSON_IN_MOTO_RATIO` in `associator.py` (try `0.15`), or
         increase `PERSON_PROX_FACTOR` (try `0.90`).

### Problem: Helmets counted for wrong person
**Symptom:** Helmet violations incorrectly reported.
**Fix:** Adjust `HELMET_HEAD_FRAC` in `associator.py`. Default is `0.40` (top 40% of person box).

### Problem: "N/A" or empty license plate
**Symptom:** `"license_plate": ""`
**Causes:**
  1. Plate not detected → train more plate data or lower `conf_thresh` on plate detector
  2. Plate too small → increase `imgsz` or add padding in `PlateOCR._crop()`
  3. Blurry plate → increase `mag_ratio` in EasyOCR call
**Fix:** Use `utils/visualize.py` to see which detections are being made.

### Problem: Multiple persons claimed by wrong motorcycle
**Symptom:** One bike shows 0 riders, another shows 4.
**Fix:** The `claimed_persons` set in `associate()` prevents double-counting.
         Make sure `PERSON_IN_MOTO_RATIO` is not too low.

### Problem: Model size over 250 MB
**Fix:** Use `YOLOv8n` instead of `YOLOv8s` in `train.py` (`YOLO("yolov8n.pt")`).
         YOLOv8n is ~6 MB. You can fit 4 of them and stay under 25 MB total.

### Problem: CUDA out of memory during training
**Fix:** Reduce `--batch` size. Start with `--batch 8`. For 4GB VRAM, use `--batch 4 --imgsz 416`.

### Problem: cv2.imread returns None on Windows with non-ASCII path
**Fix:** Already handled in `solution.py._load_image()` using `np.fromfile + cv2.imdecode`.

### Problem: EasyOCR downloads models at runtime
**Fix:** Run inference once with internet, then it caches models in `~/.EasyOCR/`.
         For offline: copy the model files from `~/.EasyOCR/model/` to the target machine.

---

## 9. Model Size Reference

| Model | Params | Size |
|-------|--------|------|
| YOLOv8n (nano) | 3.2M | ~6 MB |
| YOLOv8s (small) | 11.2M | ~22 MB |
| YOLOv8m (medium) | 25.9M | ~50 MB |
| EasyOCR (english) | — | ~60 MB cached |

**Recommended for this project:** 2× YOLOv8s + EasyOCR ≈ 44 + 60 = 104 MB total.

---

## 10. Optimization Tips

### Speed
- Use `imgsz=416` instead of `640` for ~2× speedup with minimal accuracy loss.
- Use `YOLOv8n` instead of `YOLOv8s` for ~3× speedup.
- Set `conf_thresh=0.35` to reduce false positives and NMS overhead.
- For GPU: inference is <50ms. For CPU: expect 300–600ms/image.

### Accuracy
- More data always wins — combine 3+ helmet datasets.
- Use mosaic augmentation (`mosaic=1.0`) to handle scale variation.
- Copy-paste augmentation (`copy_paste=0.1`) helps with occluded bikes.
- If helmet detection is weak, fine-tune separately on helmet-only data.

### Plate OCR
- CLAHE + Otsu binarization already handles most lighting conditions.
- If plates are yellow (commercial vehicles), add HSV yellow-mask preprocessing.
- For very small plates, crop with 10% padding and upsample to 128px height.
