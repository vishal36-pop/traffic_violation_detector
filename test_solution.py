import cv2
import numpy as np
from solution import TrafficViolationDetector

def main():
    print("Loading TrafficViolationDetector...")
    detector = TrafficViolationDetector()
    
    print("Loading image test_bus_scene.png...")
    img = cv2.imread("test_bus_scene.png")
    if img is None:
        print("Failed to load image!")
        return
        
    print("Running process_frame...")
    result = detector.process_frame(img)
    
    print(f"Detected {len(result.motorcycles)} motorcycles.")
    for i, bike in enumerate(result.motorcycles):
        violation = len(bike.riders) > 2 or any(not r.has_helmet for r in bike.riders)
        print(f"  Bike {i+1}: {len(bike.riders)} persons, {sum(1 for r in bike.riders if r.has_helmet)} helmets. Violation: {violation}")
        
if __name__ == "__main__":
    main()
