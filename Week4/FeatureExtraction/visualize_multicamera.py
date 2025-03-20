import cv2
import json
import numpy as np
from pathlib import Path
import random

# Specify your sequence and cases
sequence = "s03"
cases = ["c010", "c011", "c012", "c013", "c014", "c015"]  # Your two cameras

# Load matched tracks
matched_tracks_path = Path(f"output/{sequence}_multicamera_tracks_knn.json")
try:
    with open(matched_tracks_path, "r") as f:
        matched_tracks = json.load(f)
    print(f"Loaded {len(matched_tracks)} matched tracks")
except Exception as e:
    print(f"Error loading matched tracks: {e}")
    matched_tracks = {}

# Create global ID mapping
global_id_map = {}
next_global_id = 1

for source_track, target_track in matched_tracks.items():
    cam1, id1 = source_track.split("_")
    cam2, id2 = target_track.split("_")
    id1, id2 = int(id1), int(id2)
    
    # If either track already has a global ID, use that
    global_id = None
    if (cam1, id1) in global_id_map:
        global_id = global_id_map[(cam1, id1)]
    elif (cam2, id2) in global_id_map:
        global_id = global_id_map[(cam2, id2)]
    else:
        global_id = next_global_id
        next_global_id += 1
    
    global_id_map[(cam1, id1)] = global_id
    global_id_map[(cam2, id2)] = global_id

print(f"Created {len(global_id_map)} global ID mappings")

# Load detection results
detection_results = {}
for case in cases:
    camera_id = case[-1]
    detection_path = Path(f"output/{sequence}_{case}/detections.json")
    
    if detection_path.exists():
        try:
            with open(detection_path, "r") as f:
                detections = json.load(f)
                detection_results[camera_id] = {}
                
                # Group by frame for quick lookup
                for det in detections:
                    frame_idx = det["frame"]
                    if frame_idx not in detection_results[camera_id]:
                        detection_results[camera_id][frame_idx] = []
                    detection_results[camera_id][frame_idx].append(det)
                
                print(f"Loaded detections for camera {camera_id}: {len(detections)} detections")
        except Exception as e:
            print(f"Error loading detections for camera {camera_id}: {e}")

# Create random colors for global IDs
random.seed(42)
colors = {}
for global_id in range(1, next_global_id + 1000):  # Create more colors than needed
    colors[global_id] = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )

# Open video files
video_captures = {}
for case in cases:
    camera_id = case[-1]
    video_path = Path(f"aic19-track1-mtmc-train/train/{sequence}/{case}/vdo.avi")
    
    if video_path.exists():
        video_captures[camera_id] = cv2.VideoCapture(str(video_path))
        print(f"Opened video for camera {camera_id}: {video_path}")
    else:
        print(f"Video file not found: {video_path}")

if not video_captures:
    print("No videos could be opened. Check paths.")
    exit(1)




# Visualization loop
frame_idx = 0
running = True
target_width = 640  # Desired width for the resized frames
target_height = 360  # Desired height for the resized frames

while running:
    frames = {}
    # Read frames from all cameras
    for camera_id, cap in video_captures.items():
        ret, frame = cap.read()
        if ret:
            # Resize frame to target size
            resized_frame = cv2.resize(frame, (target_width, target_height))

            # Store the resizing factors for later use
            frame_height, frame_width = frame.shape[:2]
            resize_factor_x = target_width / frame_width
            resize_factor_y = target_height / frame_height

            frames[camera_id] = (resized_frame, resize_factor_x, resize_factor_y)
        else:
            print(f"End of video for camera {camera_id}")
    if not frames:
        print("No more frames to read")
        break

    # Draw detections and IDs on each frame
    for camera_id, (frame, resize_factor_x, resize_factor_y) in frames.items():
        # Add frame counter and camera ID
        cv2.putText(frame, f"Camera {camera_id} - Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw detections if available
        if camera_id in detection_results and frame_idx in detection_results[camera_id]:
            for det in detection_results[camera_id][frame_idx]:
                track_id = det["track_id"]
                bbox = det["bbox"]
                x, y, w, h = bbox

                # Scale bounding box coordinates
                x = int(x * resize_factor_x)
                y = int(y * resize_factor_y)
                w = int(w * resize_factor_x)
                h = int(h * resize_factor_y)

                # Get global ID if available
                if (camera_id, track_id) not in global_id_map:
                    continue
                global_id = global_id_map[(camera_id, track_id)]
                color = colors[global_id]
                label = f"Global ID: {global_id}"

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                # Draw label
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Create a canvas for side-by-side visualization in a 3x2 grid
    max_height = max(frame.shape[0] for frame, _, _ in frames.values())
    max_width = max(frame.shape[1] for frame, _, _ in frames.values())

    # Create canvas for the 3x2 grid
    canvas_height = max_height * 2  # 2 rows
    canvas_width = max_width * 3  # 3 columns
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place frames in the grid
    for i, (camera_id, (frame, _, _)) in enumerate(frames.items()):
        row = i // 3  # Determine row (0 or 1)
        col = i % 3  # Determine column (0, 1, or 2)
        h, w = frame.shape[:2]
        canvas[row * max_height:row * max_height + h, col * max_width:col * max_width + w] = frame

    # Fill empty slots with black frames if there are fewer than 6 videos
    for i in range(len(frames), 6):
        row = i // 3
        col = i % 3
        canvas[row * max_height:(row + 1) * max_height, col * max_width:(col + 1) * max_width] = 0

    # Show the combined visualization
    cv2.imshow("Multi-Camera Tracking (Mosaic)", canvas)

    # Handle keyboard input
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        running = False
    elif key == ord(' '):  # Space to pause/resume
        cv2.waitKey(0)
    elif key == ord('n'):  # 'n' to advance one frame
        pass  # Just continue to next frame
    frame_idx += 1

# Release resources
for cap in video_captures.values():
    cap.release()
cv2.destroyAllWindows()
