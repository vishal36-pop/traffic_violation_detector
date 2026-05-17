import os
import shutil
import zipfile
from pathlib import Path

def create_submission_zip():
    source_dir = Path(r"C:\Users\VishalReddyK\OneDrive\Documents\semester 4\Computer Vision\project\traffic_violation_detector")
    temp_dir = source_dir / "Traffic_Violation_Submission"
    zip_path = source_dir / "Traffic_Violation_Submission.zip"
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(exist_ok=True)

    # Files to copy from root
    root_files = [
        "solution.py", "requirements.txt", "README.md",
        "CV_Course_Project_Traffic_Violation - Copy.pdf"
    ]
    
    for f in root_files:
        src = source_dir / f
        if src.exists():
            shutil.copy2(src, temp_dir / f)

    # Directories to copy fully
    def ignore_func(d, files):
        return [f for f in files if f == '__pycache__']

    # We must include pipeline/ since solution.py depends on it
    dirs_to_copy = ["pipeline"]
    for d in dirs_to_copy:
        src = source_dir / d
        if src.exists():
            shutil.copytree(src, temp_dir / d, ignore=ignore_func, dirs_exist_ok=True)

    # Specific handling for models/
    models_dest = temp_dir / "models"
    models_dest.mkdir(exist_ok=True)
    # yolo11s.pt is now required by solution.py inside models/
    for model_file in ["combined_detector.pt", "plate_detector.pt", "yolo11s.pt"]:
        src = source_dir / "models" / model_file
        if src.exists():
            shutil.copy2(src, models_dest / model_file)

    print(f"Creating zip file at {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                # Use YOUR_ROLL_NUMBER as the root folder inside the zip per the PDF spec
                arcname = Path("YOUR_ROLL_NUMBER") / file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)
                
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Done! Submission zip is ready at: {zip_path}")

if __name__ == "__main__":
    create_submission_zip()
