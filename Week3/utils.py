import cv2
import numpy as np

def extract_features(frame, region):
    """
    Extracts robust keypoints from a specified region in the frame using the Shi-Tomasi method.

    Parameters:
        frame (numpy.ndarray): The full frame from the video in grayscale.
        region (tuple): A tuple of (x1, y1, x2, y2) defining the bounding box of the car.

    Returns:
        keypoints (numpy.ndarray): Array of detected keypoints in the region.
    """
    x1, y1, x2, y2 = map(int, region)

    roi = frame[y1:y2, x1:x2]
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    
    # Set parameters for Shi-Tomasi corner detection
    max_corners = 100       # Maximum number of corners to return
    quality_level = 0.02    # Minimal accepted quality of image corners
    min_distance = 7        # Minimum possible Euclidean distance between corners
    block_size = 7          # Size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    
    corners = cv2.goodFeaturesToTrack(roi, maxCorners=max_corners,
                                      qualityLevel=quality_level, 
                                      minDistance=min_distance, 
                                      blockSize=block_size)
    
    # If no features are found, retry with a lower threshold
    if corners is None or len(corners) == 0:
        print("No feature points found, trying with lower quality threshold...")
        corners = cv2.goodFeaturesToTrack(roi, maxCorners=50,
                                          qualityLevel=0.005,  # Lower threshold
                                          minDistance=5, 
                                          blockSize=5)
        
    # The corners are relative to the ROI, so adjust them to the frame's coordinate space.
    if corners is not None:
        corners += np.array([[x1, y1]], dtype=np.float32)
    
    return corners


def track_features_dense(feature_points, flow_field):
    """
    Tracks feature points using a precomputed dense optical flow field.
    
    Parameters:
        feature_points (numpy.ndarray): Array of keypoints (shape: (N, 1, 2) or (N, 2)).
        flow_field (numpy.ndarray): Dense optical flow field computed over the current frame,
                                    where each element [y, x] is a vector [dx, dy].
        
    Returns:
        tracked_points (numpy.ndarray): Updated keypoints based on the flow field, 
                                        formatted as an array of shape (N, 2).
    """
    # Ensure the feature_points are in shape (N, 2)
    if feature_points.ndim == 3:
        points = feature_points.reshape(-1, 2)
    else:
        points = feature_points

    new_points = []
    
    for pt in points:
        x, y = pt.ravel()
        # Convert coordinates to integer indices
        x_int, y_int = int(round(x)), int(round(y))
        
        # Check boundaries
        if y_int < flow_field.shape[0] and x_int < flow_field.shape[1]:
            dx, dy = flow_field[y_int, x_int]
            new_points.append([x + dx, y + dy])
    
    if new_points:
        tracked_points = np.array(new_points, dtype=np.float32)
    else:
        tracked_points = np.empty((0, 2), dtype=np.float32)
    
    return tracked_points


def compute_optical_flow(prev_frame, frame):
    return cv2.calcOpticalFlowFarneback(
        prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
