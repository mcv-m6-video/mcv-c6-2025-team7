import argparse
import numpy as np
import json
from pathlib import Path
from sort import Sort
from ultralytics import YOLO
import cv2
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cosine
from torchreid import models
from torchreid import utils
from torchreid import data

parser = argparse.ArgumentParser(description="Object tracking script")
parser.add_argument("--sequence", required=True, help="Nombre de la secuencia (ej. S01)")
parser.add_argument("--case", required=True, help="Nombre del caso (ej. c002)")
parser.add_argument("--visualize", action="store_true", help="Visualizar el seguimiento")
parser.add_argument("--parked", type=bool, default=False, help="Si es True, elimina los coches estacionados, si es False, los considera")
args = parser.parse_args()

sequence = args.sequence.lower()
case = args.case.lower()

# Camera ID
camera_id = args.case.lower()[-1]  # Unique camera identifier
visualize = args.visualize
parked = args.parked
seq_case_name = f"{sequence}_{case}"

# Camera relationships based on spatial layout
camera_transitions = {
    "0": ["1"],
    "1": ["0", "2", "3"],
    "2": ["1", "3"],
    "3": ["2", "4", "5"],
    "4": ["3"],
    "5": ["3"]
}

# Output directories
output_dir = Path(f"output/{seq_case_name}")
output_dir.mkdir(parents=True, exist_ok=True)

detections_json = output_dir / "detections.json"
features_json = output_dir / "features.json"
crop_dir = output_dir / "crops"
crop_dir.mkdir(parents=True, exist_ok=True)

roi_path = Path(f"aic19-track1-mtmc-train/train/{sequence}/{case}/roi.jpg")
roi_mask = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
roi_mask = roi_mask.astype(np.uint8)

# Load model and tracker
model = YOLO("yolov8l.pt")
tracker = Sort()

# Load VERI-Wild ReID model
reid_model = models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
utils.load_pretrained_weights(reid_model, 'osnet_x1_0_imagenet.pth')
reid_model.eval()
_, reid_transform = data.transforms.build_transforms(height=256, width=128, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

# Store vehicle last known positions
vehicle_positions = {}

def extract_features(image_path):
    try:
        if not Path(image_path).exists():
            print(f"Warning: Image file not found: {image_path}")
            return [0] * 512
        
        image = Image.open(image_path).convert("RGB")
        image = reid_transform(image).unsqueeze(0)
        with torch.no_grad():
            feature = reid_model(image)
        return feature.squeeze().cpu().numpy().tolist()
    except Exception as e:
        print(f"Error extracting features for {image_path}: {e}")
        return [0] * 512

def determine_next_cameras(track_id, x1, x2):
    if track_id in vehicle_positions:
        prev_x1, prev_x2 = vehicle_positions[track_id]
        movement = (x1 + x2) / 2 - (prev_x1 + prev_x2) / 2
        if movement > 5:
            return camera_transitions.get(camera_id, [])  # Moving forward
        elif movement < -5:
            return [c for c in camera_transitions if camera_id in camera_transitions[c]]  # Moving backward
    return []  # No movement detected


video_path = Path(f"aic19-track1-mtmc-train/train/{sequence}/{case}/vdo.avi")
cap = cv2.VideoCapture(str(video_path))

tracking_results = []
feature_results = []
# frame_idx = 0

# Initialize movement history
movement_history = {}
max_no_move_frames = 100  # Threshold to consider a car as parked
tolerance = 5  # Small movement tolerance

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

    # # Custom mask
    # height, width, _ = frame.shape

    # # Create a mask with white from the middle to bottom, black from top to middle
    # roi_mask2 = np.zeros((height, width), dtype=np.uint8)
    # roi_mask2[int(height//4):, :] = 255  # Make the bottom half white

    # mask = cv2.bitwise_and(roi_mask, roi_mask2)

    # # Apply the ROI mask to the current frame
    # frame = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imwrite("roi.jpg", mask)

    results = model(frame)
    detections = []
    
    for det in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) in [2, 7]:
            detection_mask = np.zeros_like(roi_mask)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            detection_mask[y1:y2, x1:x2] = 255

            if np.sum(roi_mask & detection_mask) > 0:
                detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections)
    tracked_objects = tracker.update(detections) if detections.shape[0] > 0 else []

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)

        # Movement tracking logic
        if obj_id not in movement_history:
            movement_history[obj_id] = {'last_position': (x1, y1), 'no_move_count': 0, 'moved_before': False}
        else:
            prev_x, prev_y = movement_history[obj_id]['last_position']
            distance_moved = np.sqrt((x1 - prev_x) ** 2 + (y1 - prev_y) ** 2)

            if distance_moved <= tolerance:
                movement_history[obj_id]['no_move_count'] += 1
            else:
                movement_history[obj_id]['moved_before'] = True
                movement_history[obj_id]['no_move_count'] = 0

            movement_history[obj_id]['last_position'] = (x1, y1)

        # Check if car is parked
        if parked and (not movement_history[obj_id]['moved_before'] or movement_history[obj_id]['no_move_count'] >= max_no_move_frames):
            continue  # Skip parked cars

        vehicle_positions[obj_id] = (x1, x2)
        next_expected = camera_transitions[camera_id]
        
        track_data = {
            "frame": frame_idx,
            "track_id": obj_id,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "camera_id": camera_id,
            "next_expected_cameras": next_expected
        }
        tracking_results.append(track_data)

        # Save cropped image for ReID
        try:
            # Save cropped image for ReID
            crop_img = frame[y1:y2, x1:x2]
            
            # Check if the crop is valid (not empty)
            if crop_img.size == 0 or crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                print(f"Warning: Invalid crop at frame {frame_idx}, track {obj_id}")
                continue
                
            crop_path = crop_dir / f"{frame_idx}_{obj_id}.jpg"
            cv2.imwrite(str(crop_path), crop_img)
            
            # Verify the file was saved
            if not crop_path.exists():
                print(f"Warning: Failed to save crop at {crop_path}")
                continue
                
            # Extract features for ReID
            features = extract_features(crop_path)
            feature_results.append({
                "frame": frame_idx,
                "track_id": obj_id,
                "camera_id": camera_id,
                "features": features
            })
        except Exception as e:
            print(f"Error processing detection at frame {frame_idx}, track {obj_id}: {e}")

    if visualize:
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)

            # Check if we should exclude the parked objects
            if parked and obj_id in movement_history:
                # Skip parked objects if they have never moved or have been stationary for too long
                if not movement_history[obj_id]['moved_before'] or movement_history[obj_id]['no_move_count'] >= max_no_move_frames:
                    continue  # Skip parked objects

            # Draw the bounding box and ID for moved or newly moving objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green color for tracked objects
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

with open(detections_json, "w") as f:
    json.dump(tracking_results, f, indent=4)

with open(features_json, "w") as f:
    json.dump(feature_results, f, indent=4)

print(f"Tracking results saved to {detections_json}")
print(f"Feature extraction results saved to {features_json}")
