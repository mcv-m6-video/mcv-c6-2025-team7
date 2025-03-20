import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import math

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image dimensions from meta/original_size
    meta = root.find('meta')
    task = meta.find('task')
    original_size = task.find('original_size')
    img_width = float(original_size.find('width').text)
    img_height = float(original_size.find('height').text)
    
    # Get total number of frames from task/size (if available)
    total_frames = int(task.find('size').text)
    
    # Dictionary mapping frame number to list of YOLO-format boxes
    # YOLO format: class_id x_center y_center width height (all normalized)
    annotations = {frame: [] for frame in range(total_frames)}
    
    # Process each track. Only tracks with label "car" are processed.
    for track in root.findall('track'):
        if track.attrib.get('label') != 'car':
            continue
        for box in track.findall('box'):
            frame = int(box.attrib.get('frame'))
            # Read coordinates as floats
            xtl = float(box.attrib.get('xtl'))
            ytl = float(box.attrib.get('ytl'))
            xbr = float(box.attrib.get('xbr'))
            ybr = float(box.attrib.get('ybr'))
            
            # Compute normalized center coordinates and box dimensions
            x_center = ((xtl + xbr) / 2.0) / img_width
            y_center = ((ytl + ybr) / 2.0) / img_height
            box_width = (xbr - xtl) / img_width
            box_height = (ybr - ytl) / img_height
            
            # For YOLO, assign class id 0 for cars
            annotations[frame].append((0, x_center, y_center, box_width, box_height))
    
    return annotations, total_frames

def save_annotations(annotations, total_frames, train_folder: Path, test_folder: Path):
    # Create both output directories
    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)
    
    # Define the threshold: first 25% frames for training, rest for test
    train_threshold = math.floor(total_frames * 0.25)
    
    for frame in range(total_frames):
        filename = f"AICity_frame_{frame}.txt"
        # Determine destination folder based on frame id
        if frame < train_threshold:
            out_path = train_folder / filename
        else:
            out_path = test_folder / filename
        
        with out_path.open('w') as f:
            for ann in annotations.get(frame, []):
                class_id, x_center, y_center, width, height = ann
                # Write each bounding box on a new line in YOLO format
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        print(f"Saved annotations for frame {frame} to {out_path}")

def main(xml_path, train_output, test_output):
    xml_path = Path(xml_path)
    train_folder = Path(train_output)
    test_folder = Path(test_output)
    
    annotations, total_frames = parse_xml(xml_path)
    save_annotations(annotations, total_frames, train_folder, test_folder)
    print("Finished processing annotations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert XML annotations to per-frame YOLO-format text files for car detections and split into training (25%) and test (75%) folders."
    )
    parser.add_argument("xml", help="Path to the XML ground truth file")
    parser.add_argument("--train_output", default="training_annotations", help="Directory to save training annotation text files")
    parser.add_argument("--test_output", default="test_annotations", help="Directory to save test annotation text files")
    args = parser.parse_args()
    main(args.xml, args.train_output, args.test_output)
