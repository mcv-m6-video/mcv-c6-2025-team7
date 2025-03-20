import json
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

# Specify sequence and cases
sequence = "s03"
cases = ["c010", "c011", "c012", "c013", "c014", "c015"]  # Your two cameras

# Camera transitions (simplified for just two cameras)
camera_transitions = {
    "0": ["1"],
    "1": ["0", "2"],
    "2": ["1", "3"],
    "3": ["2", "4", "5"],
    "4": ["3"],
    "5": ["3"]
}


def match_across_cameras_knn(features_data, time_constraint=50, similarity_threshold=0.7, k=3):
    print("Starting cross-camera matching using KNN...")
    
    matched_tracks = {}
    
    # Extract camera IDs
    cam_ids = list(features_data.keys())
    if len(cam_ids) < 2:
        print(f"Not enough cameras to match. Found: {cam_ids}")
        return matched_tracks
    
    # Process each camera
    for cam_id1 in cam_ids:
        if cam_id1 not in camera_transitions:
            continue
            
        print(f"Processing camera {cam_id1}...")
        tracks1 = features_data[cam_id1]
        
        # Group features by track ID
        track_features1 = {}
        for track in tracks1:
            track_id = track["track_id"]
            if track_id not in track_features1:
                track_features1[track_id] = []
            track_features1[track_id].append(track["features"])
        
        # Process neighboring cameras
        for cam_id2 in camera_transitions[cam_id1]:
            if cam_id2 not in features_data:
                continue
                
            print(f"  Comparing with camera {cam_id2}...")
            tracks2 = features_data[cam_id2]
            
            track_features2 = {}
            for track in tracks2:
                track_id = track["track_id"]
                if track_id not in track_features2:
                    track_features2[track_id] = []
                track_features2[track_id].append(track["features"])
            
            # Prepare data for KNN
            all_features2 = []
            track_labels2 = []
            
            for track_id2, feature_list in track_features2.items():
                for feature in feature_list:
                    all_features2.append(feature)
                    track_labels2.append(track_id2)  # Assign track ID as label
            
            if len(all_features2) == 0:
                continue
            
            all_features2 = np.array(all_features2)
            scaler = StandardScaler()
            all_features2 = scaler.fit_transform(all_features2)
            # Train KNN classifier
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
            knn.fit(all_features2, track_labels2)
            
            # Match tracks
            for track_id1, feature_list1 in tqdm(track_features1.items(), desc=f"Matching camera {cam_id1} to {cam_id2}"):
                global_track_id1 = f"{cam_id1}_{track_id1}"
                
                if global_track_id1 in matched_tracks:
                    continue
                
                best_match = None
                best_votes = 0
                
                # Predict using KNN
                for feature in feature_list1:
                    neighbor_ids = knn.predict([feature])  # Get predicted track ID
                    neighbor_id = neighbor_ids[0]
                    
                    # Count votes
                    votes = np.sum(knn.predict_proba([feature]) > similarity_threshold)
                    
                    if votes > best_votes:
                        best_votes = votes
                        best_match = f"{cam_id2}_{neighbor_id}"
                
                # If match found, record it
                if best_match:
                    matched_tracks[global_track_id1] = best_match
            
            print(f"  Found {len(matched_tracks)} matches so far")

    print(f"Cross-camera matching completed. Total matches: {len(matched_tracks)}")
    
    return matched_tracks

import numpy as np

def match_objects_across_cameras(tracking_data, threshold=0.7):
    """
    Matches objects across different cameras based on feature similarity.

    Parameters:
    - tracking_data: dict {camera_id: list of dicts with 'frame', 'track_id', 'camera_id', 'features'}
    - threshold: similarity threshold (higher = more strict, lower = more flexible)

    Returns:
    - matches: List of sets, each containing (camera_id, track_id) tuples that represent the same object.
    """
    
    # Step 1: Flatten the tracking data
    all_objects = []  # List of (camera_id, track_id, features)
    
    for cam_id, objects in tracking_data.items():
        for obj in objects:
            all_objects.append((obj["camera_id"], obj["track_id"], np.array(obj["features"])))
    
    num_objects = len(all_objects)
    if num_objects == 0:
        return []  # No objects to match
    
    # Step 2: Compute similarity matrix
    similarity_matrix = np.zeros((num_objects, num_objects))
    
    for i in range(num_objects):
        for j in range(i + 1, num_objects):  # Compare each pair once
            f1 = all_objects[i][2]
            f2 = all_objects[j][2]
            
            # Compute cosine similarity
            similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrix
    
    # Step 3: Group objects based on threshold
    matches = []  # List of sets containing matched objects
    visited = set()
    
    for i in range(num_objects):
        if i in visited:
            continue  # Skip already matched objects
        
        matched_set = {(all_objects[i][0], all_objects[i][1])}  # Initialize new group
        
        for j in range(num_objects):
            if i != j and similarity_matrix[i, j] > threshold:
                matched_set.add((all_objects[j][0], all_objects[j][1]))
                visited.add(j)
        
        matches.append(matched_set)
    
    return matches

# Load feature data
all_features = {}
for case in cases:
    camera_id = case[-1]
    seq_case_name = f"{sequence}_{case}"
    feature_path = Path(f"output/{seq_case_name}/features.json")
    
    if feature_path.exists():
        print(f"Loading features from {feature_path}...")
        with open(feature_path, "r") as f:
            features = json.load(f)
            print(f"Loaded {len(features)} feature entries for camera {camera_id}")
            all_features[camera_id] = features
    else:
        print(f"Warning: Feature file not found: {feature_path}")
#print(f"Keys in all_features: {list(all_features.keys())}")  # Print out camera IDs (keys)
#print(f"Number of cameras: {len(all_features)}")  # Total number of cameras

matches = match_objects_across_cameras(all_features)
print(matches)  # Print out camera IDs (keys)

'''
if len(all_features) < 2:
    print("Not enough feature data loaded to perform matching")
else:
    # Perform multi-camera tracking with KNN
    matched_tracks = match_across_cameras_knn(all_features, time_constraint=100, similarity_threshold=0.99, k=5)

    # Save the results
    output_path = Path(f"output/{sequence}_multicamera_tracks_knn.json")
    with open(output_path, "w") as f:
        json.dump(matched_tracks, f, indent=4)

    print(f"Multi-camera tracking with KNN completed. Results saved to {output_path}")
'''