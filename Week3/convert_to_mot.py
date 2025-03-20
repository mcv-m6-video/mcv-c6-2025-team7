from pathlib import Path
import json

def write_predictions_to_mot_format(pred_json_path, output_file):
    """
    Reads a JSON file of predictions and writes them in MOT challenge format.
    
    Expected JSON format (example):
    {
        "1": [
            [912.08, 93.16, 971.69, 144.48, 8.0],
            [878.52, 108.01, 929.03, 144.83, 7.0],
            ...
        ],
        "2": [
            [912.16, 92.99, 971.63, 144.41, 8.0],
            ...
        ],
        ...
    }
    
    The MOT format for each line is:
      frame, track_id, x, y, w, h, -1, -1, -1, -1
    """
    # Load the JSON file
    with open(pred_json_path, "r") as f:
        pred_data = json.load(f)
    
    # Open the output file for writing
    with open(output_file, "w") as f_out:
        # Sort frames numerically (in case the keys are strings)
        for frame_str in sorted(pred_data.keys(), key=int):
            frame = int(frame_str)
            for detection in pred_data[frame_str]:
                # Expecting each detection to be [x1, y1, x2, y2, track_id]
                if len(detection) < 5:
                    continue  # Skip if not in the expected format
                x1, y1, x2, y2, track_id = detection
                width = x2 - x1
                height = y2 - y1
                # Format the line: using two decimals for coordinates, three for width, two for height
                line = f"{frame}, {int(track_id)}, {x1:.2f}, {y1:.2f}, {width:.3f}, {height:.2f}, 1.0, -1, -1, -1\n"
                f_out.write(line)


import xml.etree.ElementTree as ET

def parse_and_write_gt_to_mot_format(xml_path, output_file):
    """
    Parse the XML ground truth file and write the detections to a text file in MOT Challenge format.
    
    Expected XML structure (CVAT format):
      <annotations>
        ...
        <track id="..." label="car">
          <box frame="..." xtl="..." ytl="..." xbr="..." ybr="..." outside="0" ...>
            <attribute name="parked">false</attribute>
          </box>
          ...
        </track>
        ...
      </annotations>
    
    Each output line will have the format:
      frame, track_id, x, y, w, h, -1, -1, -1, -1
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    with open(output_file, "w") as f:
        # Iterate over each <track> element
        for track in root.findall("track"):
            track_id = int(track.get("id"))
            label = track.get("label")
            # Optionally, only process tracks labeled as "car"
            if label != "car":
                continue
            # Process each box in the track
            for box in track.findall("box"):
                frame = int(box.get("frame"))
                # Skip if the object is marked as outside (not visible)
                if int(box.get("outside")) == 1:
                    continue
                xtl = float(box.get("xtl"))
                ytl = float(box.get("ytl"))
                xbr = float(box.get("xbr"))
                ybr = float(box.get("ybr"))
                w = xbr - xtl
                h = ybr - ytl
                # Write line in MOT format:
                # frame, track_id, x, y, w, h, -1, -1, -1, -1
                line = f"{frame}, {track_id}, {xtl:.2f}, {ytl:.2f}, {w:.2f}, {h:.2f}, 1.0, -1, -1, -1\n"
                f.write(line)

# Example usage:
# xml_gt_path = Path("dataset/ai_challenge_s03_c010-full_annotation.xml")
# output_gt_file = "gt_mot.txt"
# parse_and_write_gt_to_mot_format(xml_gt_path, output_gt_file)
# print("Ground truth MOT file written to", output_gt_file)


# Example usage:
# Suppose you have already parsed your XML ground truth:
# gt_data = parse_ground_truth_xml("path/to/annotation.xml")
# And you want to write it to 'gt_mot.txt'

# Example usage:
write_predictions_to_mot_format("ft_yolo_preds_tracks.json", "s03_c010-01.txt")
