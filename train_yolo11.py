import os
import torch
from ultralytics import YOLO

def main():
    # Check available GPU
    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "0":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Initialize the YOLO11 model from the last checkpoint
    checkpoint_path = "C:\\Users\\VishalReddyK\\OneDrive\\Documents\\semester 4\\Computer Vision\\project\\runs\\detect\\models\\combined_detector_yolo11\\weights\\last.pt"
    model = YOLO(checkpoint_path)

    # Train the model for 40 epochs
    dataset_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "combined_dataset.yaml"))
    print(f"Using dataset: {dataset_yaml}")
    
    results = model.train(
        data=dataset_yaml,
        epochs=5,
        batch=8,           # Lowered to 8 to avoid Out-of-Memory (OOM) with the yolo11l model
        imgsz=640,
        device=device,
        project="models",
        name="combined_detector_yolo11",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.001,
        # Small object augmentations (for helmets and plates)
        mosaic=1.0,
        copy_paste=0.15,
        mixup=0.05
    )

    # Run validation to get final metrics
    metrics = model.val()
    print("\nTraining complete! See charts in the models/combined_detector_yolo11 folder.")

if __name__ == "__main__":
    main()
