import trackeval
import numpy as np
import json
import xml.etree.ElementTree as ET
from pathlib import Path

#--------------------------------------
#   METRIC COMPUTATION PART
#--------------------------------------


def compute_iou(bboxA, bboxB):
    """
    Compute Intersection over Union for two bounding boxes.
    """
    xA = max(bboxA[0], bboxB[0])
    yA = max(bboxA[1], bboxB[1])
    xB = min(bboxA[2], bboxB[2])
    yB = min(bboxA[3], bboxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (bboxA[2] - bboxA[0]) * (bboxA[3] - bboxA[1])
    boxBArea = (bboxB[2] - bboxB[0]) * (bboxB[3] - bboxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0.0

def group_ids_and_similarity(gt_data, pred_data):
    """
    Build lists of IDs and similarity (IoU) matrices for each frame.
    Here, gt_data and pred_data are expected to be in TrackEval format:
    {
      "frames": {
          frame_number: [ {"id": track_id, "bbox": [x1, y1, x2, y2]}, ... ],
          ...
      }
    }
    """
    gt_ids_list = []
    tracker_ids_list = []
    similarity_scores_list = []
    
    # Use the keys inside the "frames" dictionary.
    all_frames = sorted(set(gt_data["frames"].keys()) | set(pred_data["frames"].keys()))
    
    # Build mapping of unique IDs to contiguous indices.
    all_gt_ids = sorted({det["id"] for dets in gt_data["frames"].values() for det in dets})
    all_tracker_ids = sorted({det["id"] for dets in pred_data["frames"].values() for det in dets})

    gt_id_map = {tid: idx for idx, tid in enumerate(all_gt_ids)}
    tracker_id_map = {tid: idx for idx, tid in enumerate(all_tracker_ids)}

    # Counters for total detections
    num_gt_dets = sum(len(dets) for dets in gt_data["frames"].values())
    num_tracker_dets = sum(len(dets) for dets in pred_data["frames"].values())  
    
    for frame in all_frames:
        frame_gt = gt_data["frames"].get(frame, [])
        frame_tracker = pred_data["frames"].get(frame, [])
        
        # Map the IDs to contiguous indices.
        gt_ids = np.array([gt_id_map[det["id"]] for det in frame_gt], dtype=int) if frame_gt else np.array([])
        tracker_ids = np.array([tracker_id_map[det["id"]] for det in frame_tracker], dtype=int) if frame_tracker else np.array([])
        gt_ids_list.append(gt_ids)
        tracker_ids_list.append(tracker_ids)
      
        # Compute similarity (IoU) matrix.
        if frame_gt and frame_tracker:
            sim_matrix = np.zeros((len(frame_gt), len(frame_tracker)))
            for i, det in enumerate(frame_gt):
                bbox_gt = det["bbox"]
                for j, det_tr in enumerate(frame_tracker):
                    bbox_tr = det_tr["bbox"]
                    sim_matrix[i, j] = compute_iou(bbox_gt, bbox_tr)
        else:
            sim_matrix = np.empty((len(frame_gt), len(frame_tracker)))
        
        similarity_scores_list.append(sim_matrix)
    
    data = {
        'num_gt_data': num_gt_dets,
        'num_pred_data': num_tracker_dets,
        'gt_ids': gt_ids_list,
        'tracker_ids': tracker_ids_list,
        'similarity_scores': similarity_scores_list,
        'num_gt_ids': len(all_gt_ids),
        'num_tracker_ids': len(all_tracker_ids)
    }
    
    return data

def evaluate_tracking_with_trackeval(gt_data, pred_data):
    """
    Evaluates tracking performance using TrackEval's HOTA and IDF1 metrics.

    Parameters:
        ground_truth (dict): Dictionary mapping frame numbers to ground truth detections.
                             Each detection is (track_id, bbox).
        predictions (dict): Dictionary mapping frame numbers to predicted detections.
                            Each detection is (track_id, bbox).

    Returns:
        metrics (dict): Dictionary containing the computed HOTA and IDF1 scores.
    """
    hota_metric = trackeval.metrics.hota.HOTA()
    identity_metric = trackeval.metrics.identity.Identity()

    data = group_ids_and_similarity(gt_data, pred_data)

    hota_results = hota_metric.eval_sequence(data)
    hota_score = hota_results["HOTA"]

    idf1_results = identity_metric.eval_sequence(data)
    idf1_score = idf1_results["IDF1"]

    return {
        'HOTA': hota_score,
        'IDF1': idf1_score
    }

#---------------------------------------
#   FILE CONVERSION TO TRACKEVAL PART
#---------------------------------------

def convert_to_trackeval_format(ground_truth, predictions):
    """
    Converts tracking data into TrackEval's required format.

    Parameters:
        ground_truth (dict): Mapping frame numbers to ground truth detections (track_id, bbox).
        predictions (dict): Mapping frame numbers to predicted detections (track_id, bbox).

    Returns:
        gt_data (dict): TrackEval-compatible ground truth.
        pred_data (dict): TrackEval-compatible predictions.
    """
    gt_data = {"frames": {}}
    pred_data = {"frames": {}}

    for frame, detections in ground_truth.items():
        print(f"Frame: {frame}, Detections: {detections}") 
        gt_data["frames"][frame] = [{"id": track_id, "bbox": bbox} for track_id, bbox in detections]
        
    for frame, detections in predictions.items():
        pred_data["frames"][frame] = [{"id": track_id, "bbox": bbox} for track_id, bbox in detections]

    return gt_data, pred_data


def convert_tracked_json_to_trackeval(json_path):
    """
    Converts a tracked JSON file to TrackEval format.

    Parameters:
        json_path (str): Path to the JSON file containing tracked predictions.

    Returns:
        pred_data (dict): TrackEval-compatible predictions dictionary.
    """
    pred_data = {"frames": {}}

    # Load JSON file
    with open(json_path, "r") as f:
        predictions = json.load(f)

    # Iterate over frames
    for frame_str, detections in predictions.items():
        frame = int(frame_str)  # Convert string key to int
        pred_data["frames"][frame] = []

        for det in detections:
            x1, y1, x2, y2, track_id = det  # Extract values

            pred_data["frames"][frame].append({
                "id": int(track_id),  # TrackEval requires integer IDs
                "bbox": [x1, y1, x2, y2]
            })

    return pred_data


def parse_ground_truth_xml(xml_path):
    """
    Parses the XML ground truth file into TrackEval format.
    
    Parameters:
        xml_path (str): Path to the XML file.
    
    Returns:
        gt_data (dict): TrackEval-compatible ground truth dictionary.
    """
    gt_data = {"frames": {}}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Iterate through all <track> elements
    for track in root.findall("track"):
        track_id = int(track.get("id"))

        for box in track.findall("box"):
            frame = int(box.get("frame"))

            if int(box.get("outside")) == 1:  # Skip boxes marked as 'outside'
                continue

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            bbox = [xtl, ytl, xbr, ybr]

            if frame not in gt_data["frames"]:
                gt_data["frames"][frame] = []

            gt_data["frames"][frame].append({"id": track_id, "bbox": bbox})

    return gt_data

if __name__ == "__main__":
    # Define paths to your ground truth XML and tracked predictions JSON
    gt_xml_path = Path("TrackEval/TrackEval/data/gt/mot_challenge/s03_c010-train/s03_c010-01/gt/gt.txt.txt")
    tracked_json_path = Path("TrackEval/TrackEval/data/trackers/mot_challenge/PerfectTracker/data/s03_c010-01.txt")
    
    # # Convert ground truth and predictions to TrackEval format
    # gt_data = parse_ground_truth_xml(gt_xml_path)
    # pred_data = convert_tracked_json_to_trackeval(tracked_json_path)
    
    # Evaluate tracking metrics
    metrics = evaluate_tracking_with_trackeval(gt_xml_path, tracked_json_path)
    print("HOTA Score:", metrics['HOTA'])
    print("IDF1 Score:", metrics['IDF1'])