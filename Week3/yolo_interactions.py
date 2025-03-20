"""
YOLO Object Detection and Finetuning Script
===========================================

This script provides functionalities for finetuning a YOLO model and using it to predict 
object bounding boxes in video sequences. It supports two primary modes of operation:

1. **Finetune Mode**: Trains the YOLO model using a specified strategy, batch size, and 
   number of epochs.
2. **Predict Mode**: Uses a trained YOLO model to detect objects in a video and saves 
   the results in a JSON file.

Dependencies:
-------------
- `ultralytics` (for YOLO model operations)
- `OpenCV` (for video processing)
- `argparse` (for command-line arguments handling)
- `json` (for saving predictions)

Functions:
----------
- `detect_with_yolo(model, frame, conf_thresh: float) -> tuple`:
    Detects objects in a given video frame using a YOLO model and draws bounding boxes.
- `finetune_yolo(output_folder: Path, strategy: str, batch_size: int, epochs: int) -> None`:
    Finetunes a YOLO model with specified training parameters.
- `predict_frames_with_yolo(video_path: Path, weights_path: Path, output_file: Path, conf_thresh: float = 0.7) -> None`:
    Runs object detection on video frames and saves results in JSON format.

Usage:
------
Run the script from the command line with the appropriate mode and arguments.

Finetune Example:
    ```
    python script.py finetune --output_folder yolo_output --strategy B --batch_size 16 --epochs 50
    ```

Predict Example:
    ```
    python script.py predict --video_path input.mp4 --weights_path best_weights.pt --output_file predictions.json --conf_thresh 0.5
    ```
"""
import numpy as np
import argparse
from ultralytics import YOLO
from pathlib import Path
import cv2
import json
from sort import Sort


sort_tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

def convert_ndarray_to_list(obj):
    """ 
    Recursively converts numpy arrays to lists for JSON serialization 
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    
    else:
        return obj

def detect_and_track_with_yolo(model, frame, conf_thresh=0.7):
    """
    Uses Sort to track the detected objects with yolo
    """
    detections, annotated_frame = detect_with_yolo(model, frame, conf_thresh)
    
    if len(detections) == 0:
        return np.empty((0, 5)), annotated_frame

    sort_dets = []

    for det in detections:
        bbox, _, conf = det
        x1, y1, x2, y2 = bbox
        sort_dets.append([x1, y1, x2, y2, conf])

    # Update the SORT tracker
    sort_dets = np.array(sort_dets)
    tracked_objects = sort_tracker.update(sort_dets)  # tracked_objects: each row [x1, y1, x2, y2, track_id]
    
    # Annotate the frame with the track IDs
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj

        cv2.putText(annotated_frame, f"ID: {int(track_id)}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

    return tracked_objects, annotated_frame

def detect_with_yolo(model, frame, conf_thresh: float=0.7) -> tuple:
    """
    Execute the model (Yolo finetuned) to detect the bboxes from the video.
    """
    results = model(frame, verbose=False)
    bboxes = []

    for detection in results:
        for box in detection.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            if class_id == 0 and confidence >= conf_thresh:
                bboxes.append(((x1, y1, x2, y2), class_id, confidence))

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (203, 204, 170), 2)
                cv2.putText(frame, f"car: {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (203, 204, 170), 2)
                
    return bboxes, frame

def finetune_yolo(output_folder: Path, strategy: str, batch_size: int, epochs: int) -> None:
    """
    Finetune YOLO over a defined fold strategy.
    """
    model = YOLO("yolov8n.pt")
    model.train(
        data="config.yaml",
        epochs=epochs,
        batch=batch_size,
        freeze=10,
        project=f'{output_folder}',
        name=f"{strategy}",
    )

def predict_frames_with_yolo(video_path: Path, weights_path: Path, output_file: Path, conf_thresh: float = 0.7) -> None:
    """
    Predict object bboxes from a video sequence with YOLO.
    """
    model = YOLO(weights_path)
    
    frame_id = 0
    predicions = {}

    cap = cv2.VideoCapture(str(video_path))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame_id += 1

        frame_predicts, frame = detect_and_track_with_yolo(model, frame, conf_thresh=conf_thresh)
        predicions[frame_id] = frame_predicts

    cap.release()

    with open(output_file, "w") as f:
        json.dump(convert_ndarray_to_list(predicions), f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run YOLO operations: finetune or predict."
    )
    
    # Create subparsers for each mode
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Choose the operation mode: finetune or predict.")

    # Subparser for finetune mode
    finetune_parser = subparsers.add_parser("finetune", help="Finetune the YOLO model.")
    finetune_parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path("yolo_ft_output"),
        help="Path to the output folder for training results. (default: yolo_ft_output)",
    )
    finetune_parser.add_argument(
        "--strategy",
        type=str,
        default="A",
        help="Strategy label for this run. (default: A)",
    )
    finetune_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training. (default: 32)",
    )
    finetune_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for training. (default: 100)",
    )

    # Subparser for predict mode
    predict_parser = subparsers.add_parser("predict", help="Predict object bounding boxes from a video using YOLO.")
    predict_parser.add_argument(
        "--video_path",
        type=Path,
        default=Path('output.avi'),
        help="Path to the input video file.",
    )
    predict_parser.add_argument(
        "--weights_path",
        type=Path,
        default=Path('y8_ft_default.pt'),
        help="Path to the YOLO weights file.",
    )
    predict_parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to save the JSON output with predictions.",
    )
    predict_parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.7,
        help="Confidence threshold for detections. (default: 0.7)",
    )

    args = parser.parse_args()

    if args.mode == "finetune":
        finetune_yolo(
            output_folder=args.output_folder,
            strategy=args.strategy,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )
    elif args.mode == "predict":
        predict_frames_with_yolo(
            video_path=args.video_path,
            weights_path=args.weights_path,
            output_file=args.output_file,
            conf_thresh=args.conf_thresh,
        )
