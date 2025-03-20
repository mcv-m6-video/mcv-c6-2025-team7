import cv2
from ultralytics import YOLO
from yolo_interactions import detect_and_track_with_yolo
from utils import extract_features, track_features_dense, compute_optical_flow
import numpy as np
from pathlib import Path
import torch

"""
Task 1.2 Pipeline
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_path = Path("AICity_data/train/S03/c010/vdo.avi")
weights_path = Path("y8_ft_default.pt")
output_path = Path("task1_2.avi")

# Initialize video capture
cap = cv2.VideoCapture(str(video_path))  # Ensure path is a string
assert cap.isOpened(), f"Error: Cannot open video {video_path}"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
assert out.isOpened(), f"Error: Cannot open video writer for {output_path}"

# Initialize YOLO model, and optical flow parameters
model = YOLO(weights_path).to(device)
prev_frame = None
tracking_data = {}  # Dictionary to hold tracking information (id -> trajectory)
kalman_filters = {}  # Store Kalman filters for each object
refined_detections = []
frame_idx = 0

def draw_bbox(frame, detection):
    """Draws bounding box on the frame."""
    x1, y1, x2, y2, track_id= detection

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    color = (0, 255, 0)  # Green for detection
    cv2.rectangle(frame, (x1, y1), (x2, y2), color)


def draw_tracking_line(frame, track):
    """
    Draws the trajectory of the tracked object on the frame with a fading effect.
    The oldest segments are drawn with lower opacity and the most recent with higher opacity.
    """
    num_points = len(track)
    if num_points < 2:
        return
    # Loop over each segment in the track
    for i in range(1, num_points):
        # Compute an alpha that increases with the segment index:
        # For example: oldest segment has alpha=0.2, newest has alpha=1.0
        alpha = 0.2 + 0.8 * ((i - 1) / (num_points - 2)) if num_points > 2 else 1.0
        prev_pt = tuple(map(int, track[i - 1]))
        current_pt = tuple(map(int, track[i]))
        # Create a temporary image to draw the line
        line_img = np.zeros_like(frame, dtype=np.uint8)
        cv2.line(line_img, prev_pt, current_pt, (0, 0, 255), 2)
        # Blend the line onto the frame with the computed alpha
        frame[:] = cv2.addWeighted(frame, 1.0, line_img, alpha, 0)


def refine_bbox(detection, tracked_points):
    """Refines the bounding box based on the movement of tracked points."""
    if tracked_points is not None and len(tracked_points) > 0:
        # Update bounding box using the centroid of the tracked points
        centroid = np.mean(tracked_points, axis=0)
        x, y, w, h, track_id = detection  # Assuming detection is (bbox, class_id, confidence)
        # Update the bbox around the centroid
        new_x = int(centroid[0] - w / 2)
        new_y = int(centroid[1] - h / 2)
        return (new_x, new_y, w, h), track_id
    return detection  # If no tracking, return original detection


def write_to_file(tracking_data):
    """
    Writes the tracking data into a text file named "tracking_data.txt" in CSV-like format.
    Each line in the file will have the following format:
        track_id, x, y
    where:
        - track_id is the ID of the tracked object.
        - x, y are the estimated positions (centroid) for that object.

    Parameters:
        tracking_data (dict): A dictionary where each key is a track ID and each value is a list
                              of (x, y) positions representing the object's trajectory.

    Returns:
        result (dict): A dictionary containing information about the output file.
    """
    output_filename = "tracking_data.txt"
    with open(output_filename, "w") as f:
        f.write("track_id, x, y\n")
        for track_id, positions in tracking_data.items():
            for (x, y) in positions:
                # Format the coordinates to two decimal places
                f.write(f"{track_id}, {x:.2f}, {y:.2f}\n")
    # Return a result dictionary to pass to the save_results function
    return {"tracking_file": output_filename, "num_tracks": len(tracking_data)}


def save_results(results):
    """Saves the evaluation results to a file."""
    with open("evaluation_results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")


# Main pipeline
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx +=1

    detections, frame = detect_and_track_with_yolo(model, frame)  # conf_thresh = 0.7 default
    # Step 2: If previous frame exists, compute optical flow
    if prev_frame is not None:
        gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow between prev_frame and current frame
        flow_vectors = compute_optical_flow(gray_prev_frame, gray_frame)

        # Step 3: For each detection, refine tracking using optical flow
        for detection in detections:
            # bbox, class_id, confidence = detection
            # print(f"loop: {detection}")
            x1, y1, x2, y2, track_id = detection

            feature_points = extract_features(gray_frame, region=(x1, y1, x2, y2))

            # Track these points using the computed optical flow
            tracked_points = track_features_dense(feature_points, flow_vectors)

            # Aggregate tracked points to compute a measurement (e.g., centroid)
            if tracked_points is not None and len(tracked_points) > 0:
                centroid = np.mean(tracked_points, axis=0)  # (x, y) average

                # # Check if this detection has been tracked before
                # car_id = int(detection[1])  # Example: using class_id as car_id (or another method to track ID)

                if track_id not in kalman_filters:
                    # Initialize Kalman filter for this car_id
                    kalman_filter = cv2.KalmanFilter(4, 2)  # (x, y, vx, vy), (x, y)
                    initial_state = np.array([[centroid[0]], [centroid[1]], [0], [0]], dtype=np.float32)
                    kalman_filter.statePre = initial_state.copy()
                    kalman_filter.statePost = initial_state.copy()
                    kalman_filter.transitionMatrix = np.eye(4, dtype=np.float32)  # State transition matrix
                    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]],
                                                               dtype=np.float32)  # Measurement matrix
                    kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2  # Process noise
                    kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1  # Measurement noise
                    kalman_filters[track_id] = kalman_filter

                kalman_filter = kalman_filters[track_id]
                kalman_state = kalman_filter.predict()  # Predict next state
                kalman_state = kalman_filter.correct(
                    np.array([[centroid[0]], [centroid[1]]]))  # Correct with new measurement
                kalman_state = kalman_state.flatten().tolist()

                # Store the updated centroid (car trajectory) in tracking data
                if track_id not in tracking_data:
                    tracking_data[track_id] = []
                tracking_data[track_id].append((kalman_state[0], kalman_state[1]))  # Store the tracked position

             # Refine bbox based on optical flow information
            refined_detection, track_id = refine_bbox(detection, tracked_points)
            # refined_detection is (new_x, new_y, w, h)
            x, y, w, h = refined_detection
            # Save detection in desired MOT format: frame, track_id, x, y, w, h, 1, -1, -1, -1
            refined_detections.append((frame_idx, int(track_id), int(x), int(y), int(w), int(h), 1, -1, -1, -1))
            # Update the detection in the list if needed (for visualization)
            detection = (x, y, x+w, y+h, track_id)

    # Step 4: Visualization and logging
    for detection in detections:
        draw_bbox(frame, detection)  # Draw YOLO (or refined) bounding box on frame
    for track_id, track in tracking_data.items():
        draw_tracking_line(frame, track)  # Visualize car trajectory

    out.write(frame)
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Prepare for next iteration
    prev_frame = frame

cap.release()
out.release()
cv2.destroyAllWindows()

# Save refined detections to a file in MOT format
with open("refined_tracking.txt", "w") as f:
    for det in refined_detections:
        line = f"{det[0]}, {det[1]}, {det[2]}, {det[3]}, {det[4]}, {det[5]}, {det[6]}, {det[7]}, {det[8]}, {det[9]}\n"
        f.write(line)

# After processing the video, compute evaluation metrics (accuracy, precision, recall, etc.)
results = write_to_file(tracking_data)
save_results(results)
