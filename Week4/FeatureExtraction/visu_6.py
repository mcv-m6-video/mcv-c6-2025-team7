import cv2
import numpy as np
import time

# Video details
videos = []
sequence = "s03"
cases = ["c010", "c011", "c012", "c013", "c014", "c015"]  # Cameras

# Custom start delays (in seconds)
custom_delays = {
    "c010": 8.715,
    "c011": 8.457,
    "c012": 5.879,
    "c013": 0,
    "c014": 5.042,
    "c015": 8.492,
}

# Target FPS
target_fps = 10
frame_interval = 1 / target_fps  # Time between frames in seconds

# Gather video paths
for case in cases:
    video_path = f"../aic19-track1-mtmc-train/train/S03/videos/{sequence}_{case}.avi"
    videos.append(video_path)

# Open video captures
captures = [cv2.VideoCapture(video) for video in videos]
print(len(captures))
# Retrieve frame counts and frame rates
frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0 for cap in captures]
fps_list = [int(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else target_fps for cap in captures]

# Calculate automatic delays (sync all videos to finish at the same time)
max_duration = max(frame_counts[i] / fps_list[i] for i in range(len(captures)))
delays = [
    custom_delays[cases[i]] if custom_delays[cases[i]] is not None else (max_duration - (frame_counts[i] / fps_list[i]))
    for i in range(len(captures))
]

# Resize parameters
video_width, video_height = 320, 240

# Initialize frame counters
frame_indices = [0] * len(captures)

# Start time for elapsed time calculation
start_time = time.time()

while True:
    current_time = time.time()
    elapsed_time = current_time - start_time

    frames = []
    for i, cap in enumerate(captures):
        if not cap.isOpened():
            frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        elif elapsed_time < delays[i]:  # Still in delay period
            frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        else:
            ret, frame = cap.read()
            if not ret:  # Video ended
                frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (video_width, video_height))

        # Draw frame number in red
        frame_number_text = f"Frame: {frame_indices[i]}"
        cv2.putText(frame, frame_number_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        case_text = cases[i]  # Get the case name (e.g., "c010")
        text_size, _ = cv2.getTextSize(case_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_width = text_size[0]

        x_pos = video_width - text_width - 10  # Position it 10px from the right
        y_pos = 30  # Position at 30px from the top

        cv2.putText(frame, case_text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        frames.append(frame)

        # Increment frame counter only if delay is over
        if elapsed_time >= delays[i]:
            frame_indices[i] += 1

    # Create mosaic (2 rows x 3 columns)
    row1 = np.hstack(frames[:3])
    row2 = np.hstack(frames[3:])
    mosaic = np.vstack([row1, row2])

    # Display time elapsed
    cv2.putText(mosaic, f"Time: {elapsed_time:.2f} s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Synchronized Mosaic with Time Overlay", mosaic)

    # Wait to match FPS
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(frame_interval)

# Release resources
for cap in captures:
    cap.release()
cv2.destroyAllWindows()
