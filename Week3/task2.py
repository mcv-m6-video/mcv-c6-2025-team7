import argparse
import numpy as np
from pathlib import Path
from sort import Sort
from ultralytics import YOLO
import cv2
import shutil

parser = argparse.ArgumentParser(description="Object tracking script")
parser.add_argument("--sequence", required=True, help="Nombre de la secuencia (ej. S01)")
parser.add_argument("--case", required=True, help="Nombre del caso (ej. c002)")
parser.add_argument("--visualize", action="store_true", help="Visualizar el seguimiento")
parser.add_argument("--parked", type=bool, default=False, help="Si es True, elimina los coches estacionados, si es False, los considera")

args = parser.parse_args()
sequence = args.sequence.lower()  # Convertimos a minúsculas por consistencia
case = args.case.lower()
visualize = args.visualize
parked = args.parked  # Store the value of the --parked argument
seq_case_name = f"{sequence}_{case}"


gt_dir = Path(f"TrackEval/data/gt/mot_challenge/{seq_case_name}-train/{seq_case_name}-01")
gt_dir.mkdir(parents=True, exist_ok=True)

seqinfo_path = gt_dir / "seqinfo.ini"
if not seqinfo_path.exists():
    seq_length = 1996  
    with open(seqinfo_path, "w") as f:
        f.write(f"[Sequence]\nname={seq_case_name}\nseqLength={seq_length}\n")

txt_files = ["all", "test", "train"]
seqmap_path = Path(f"TrackEval/data/gt/mot_challenge/seqmaps")
for txt in txt_files:
    txt_path = seqmap_path / f"{seq_case_name}-{txt}.txt"
    if not txt_path.exists():
        with open(txt_path, "w") as f:
            f.write("name\n")
            f.write(f"{seq_case_name}-01\n")


gt_subdir = gt_dir / "gt"
gt_subdir.mkdir(parents=True, exist_ok=True)

source_gt_file = Path(f"aic19-track1-mtmc-train/train/{sequence}/{case}/gt/gt.txt")
destination_gt_file = gt_subdir / "gt.txt"

if source_gt_file.exists() and not destination_gt_file.exists():
    shutil.copy(source_gt_file, destination_gt_file)
    print(f"Copied ground truth file to {destination_gt_file}")

output_dir = Path(f"TrackEval/data/trackers/mot_challenge/{seq_case_name}-train/PerfectTracker/data")
output_dir.mkdir(parents=True, exist_ok=True)

if parked:
    output_file = output_dir / f"{seq_case_name}-01_parked.txt"
else:
    output_file = output_dir / f"{seq_case_name}-01.txt"


if output_file.exists():
    output_file.unlink()


model = YOLO("yolov8l.pt")

tracker = Sort()

video_path = Path(f"aic19-track1-mtmc-train/train/{sequence}/{case}/vdo.avi")
cap = cv2.VideoCapture(str(video_path))

roi_path = Path(f"aic19-track1-mtmc-train/train/{sequence}/{case}/roi.jpg")
roi_mask = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
roi_mask = roi_mask.astype(np.uint8)

use_roi = True

gt_file = Path(f"aic19-track1-mtmc-train/train/{sequence}/{case}/gt/gt.txt")

gt_data = {}
if gt_file.exists():
    with open(gt_file, "r") as f:
        for line in f:
            line = line.strip().split(',')
            frame_idx = int(line[0])
            obj_id = int(line[1])
            x1, y1, width, height = map(int, line[2:6])

            if frame_idx not in gt_data:
                gt_data[frame_idx] = []
            gt_data[frame_idx].append([x1, y1, x1 + width, y1 + height, obj_id])

# Initialize movement history and parked detection logic
movement_history = {}
max_no_move_frames = 10  # Number of frames an object can stay without moving to be considered "parked"
tolerance = 10  # Tolerance for small movements (in pixels)

def save_detections_mots(output_txt, tracked_objects, frame_idx, movement_history, max_no_move_frames, parked):
    with open(output_txt, "a") as f:
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)

            # Check if we should exclude the parked objects
            if parked and obj_id in movement_history:
                # Skip parked objects if they have never moved or have been stationary for too long
                if not movement_history[obj_id]['moved_before'] or movement_history[obj_id]['no_move_count'] >= max_no_move_frames:
                    continue  # Skip parked objects

            # Save the detection if it's not a parked car
            width = x2 - x1
            height = y2 - y1
            conf = 1  # Fixed confidence
            f.write(f"{frame_idx + 1}, {obj_id}, {x1}, {y1}, {width}, {height}, {conf:.2f}, -1, -1, -1\n")

if parked:
    output_video = output_dir / f"{seq_case_name}-01_parked.avi"
else:
    output_video = output_dir / f"{seq_case_name}-01.avi"
    
if visualize:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    assert out.isOpened(), f"Error: Cannot open video writer for {output_video}"

# Procesar el video
while cap.isOpened():
    print(f"Processing frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

    results = model(frame)

    detections = []
    for det in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 2 or int(cls) == 7:
            if use_roi:
                detection_mask = np.zeros_like(roi_mask)
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                detection_mask[y1:y2, x1:x2] = 255

                if np.sum(roi_mask & detection_mask) > 0:
                    detections.append([x1, y1, x2, y2, conf])
            else:
                detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections)

    if detections.shape[0] > 0:
        tracked_objects = tracker.update(detections)
    else:
        tracked_objects = []

    # Update movement history for each object
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        if obj_id not in movement_history:
            movement_history[obj_id] = {'last_position': (x1, y1), 'no_move_count': 0, 'moved_before': False}
        else:
            prev_position = movement_history[obj_id]['last_position']
            # Calculate the distance moved using Euclidean distance (with tolerance)
            distance_moved = np.sqrt((x1 - prev_position[0]) ** 2 + (y1 - prev_position[1]) ** 2)
            if distance_moved <= tolerance:
                movement_history[obj_id]['no_move_count'] += 1
            else:
                # Mark the object as moved at least once
                movement_history[obj_id]['moved_before'] = True
                movement_history[obj_id]['no_move_count'] = 0
                
            movement_history[obj_id]['last_position'] = (x1, y1)

    save_detections_mots(output_file, tracked_objects, frame_idx, movement_history, max_no_move_frames, parked)

    if visualize:
        # Draw ground truth bounding boxes
        if frame_idx in gt_data:
            for gt_bbox in gt_data[frame_idx]:
                x1, y1, x2, y2, obj_id = gt_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for ground truth
                cv2.putText(frame, f"GT {obj_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Draw results from tracker, considering the --parked argument
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

        out.write(frame)
        # Show frame
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()