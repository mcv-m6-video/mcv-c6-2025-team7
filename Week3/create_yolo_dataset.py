"""
Video Frame Extractor

This script processes a given video file, extracting its frames and categorizing them
into two separate folders: 'train' (first 25% of frames) and 'test' (remaining 75%).
It ensures the required directories exist before saving the extracted frames.

Modules:
    - cv2: Used for video processing and frame extraction.
    - math: Utilized for calculating the training frame threshold.
    - argparse: Handles command-line argument parsing.
    - pathlib.Path: Manages file paths and directory creation.

Functions:
    - create_dir(path: Path): Ensures a directory exists, creating it if necessary.
    - main(video_path): Processes the video, extracts frames, and distributes them
      into training and testing sets.

Usage:
    Run the script from the command line with the path to the video file:
    
    ```
    python script.py /path/to/video.mp4
    ```

    This will generate two folders:
      - dataset/frames/train/ → containing the first 25% of frames
      - dataset/frames/test/ → containing the remaining 75% of frames
"""

import cv2
import math
import argparse
from pathlib import Path

def create_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def main(video_path):
    video_path = Path(video_path)
    # Create output folders using pathlib
    train_folder = Path("dataset/frames/train")
    test_folder = Path("dataset/frames/test")
    create_dir(train_folder)
    create_dir(test_folder)

    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("Error: Video has no frames.")
        return

    # Calculate threshold for training set (first 25%)
    train_threshold = math.floor(total_frames * 0.25)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = f"AICity_frame_{frame_id}.jpg"
        # Determine destination folder based on frame id
        if frame_id < train_threshold:
            out_path = train_folder / frame_filename
        else:
            out_path = test_folder / frame_filename

        # Save frame to disk
        cv2.imwrite(str(out_path), frame)
        print(f"Saved frame {frame_id} to {out_path}")
        frame_id += 1

    cap.release()
    print("Finished processing all frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split video frames into training (25%) and test (75%) folders."
    )
    parser.add_argument("video", help="Path to the video file")
    args = parser.parse_args()
    main(args.video)
