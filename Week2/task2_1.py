import torch
import torchvision.transforms as T
import torchvision.ops as ops
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from transformers import DetrForObjectDetection



class ObjectTracker:
    def __init__(self, iou_threshold=0.5):
        self.tracked_objects = {}  # Stores {ID: (bbox, frame_last_seen)}
        self.next_id = 1
        self.iou_threshold = iou_threshold
    
    def update(self, detections, frame_idx):
        updated_tracks = {}
        assigned_ids = set()
        
        for det in detections:
            best_iou = 0
            best_id = None
            for obj_id, (prev_bbox, last_frame) in self.tracked_objects.items():
                iou = compute_iou(prev_bbox, det['bbox'])
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_id = obj_id
            
            if best_id is not None:
                updated_tracks[best_id] = (det['bbox'], frame_idx)
                assigned_ids.add(best_id)
            else:
                updated_tracks[self.next_id] = (det['bbox'], frame_idx)
                assigned_ids.add(self.next_id)
                self.next_id += 1
        
        # Remove old objects if not seen in recent frames
        self.tracked_objects = {
            obj_id: (bbox, last_frame)
            for obj_id, (bbox, last_frame) in updated_tracks.items()
            if frame_idx - last_frame <= 5  # Keep objects for n frames
        }
        
        return updated_tracks



# Load the pretrained DETR model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DETR model and move it to GPU
detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
detr.eval()


# Define transformation for input images
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((800, 800)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# COCO Class labels for DETR
CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Function to compute Intersection over Union (IoU)
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area != 0 else 0

# Function to process a frame through DETR
def detect_objects(frame):
    img = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = detr(img)  # This returns a dictionary-like object
    
    # Access logits and bounding boxes directly from the outputs
    return {
        "logits": outputs.logits,  # Directly access logits
        "bbox": outputs.pred_boxes  # Directly access predicted bounding boxes
    }




# Function to draw bounding boxes
def draw_boxes(frame, ground_truth):    
    # Draw ground truth boxes in GREEN
    for gt in ground_truth:
        x1, y1, x2, y2 = gt['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for ground truth
        cv2.putText(frame, gt['label'], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame


# Function to parse CVAT-style XML annotations
import xml.etree.ElementTree as ET

def parse_cvat_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = {}

    for track in root.findall(".//track"):
        label = track.get("label")
        if label != "car":  # Filter for cars only
            continue

        for box in track.findall("box"):
            frame = int(box.get("frame"))
            xtl, ytl, xbr, ybr = map(float, [box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")])

            # Check if the car is parked
            parked_attr = box.find(".//attribute[@name='parked']")
            is_parked = False  # Default to false (moving)
            if parked_attr is not None and parked_attr.text.strip().lower() == "true":
                is_parked = True  # Mark as parked
            
            # Store both parked and moving cars
            if frame not in annotations:
                annotations[frame] = []
            
            annotations[frame].append({
                "label": label,
                "bbox": (int(xtl), int(ytl), int(xbr), int(ybr)),
                "parked": is_parked  # Add parked/moving info
            })

    return annotations

def save_detections_mots(output_txt, tracked_objects, frame_idx):
    with open(output_txt, "a") as f:
        for obj_id, (bbox, _) in tracked_objects.items():
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            conf = 1  # Fixed confidence
            f.write(f"{frame_idx + 1}, {obj_id}, {x1}, {y1}, {width}, {height}, {conf:.2f}, -1, -1, -1\n")


def apply_nms(detections, iou_threshold=0.4):
    if len(detections) == 0:
        return []
    
    boxes = torch.tensor([d['bbox'] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d['score'] for d in detections], dtype=torch.float32)
    
    keep_idx = ops.nms(boxes, scores, iou_threshold)
    return [detections[i] for i in keep_idx]


def filter_and_transform_detections(frame, outputs, threshold=0.7):
    h, w, _ = frame.shape
    probas = outputs['logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    boxes = outputs['bbox'][0, keep].cpu().numpy()
    labels = probas.argmax(-1)[keep].cpu().numpy()
    scores = probas.max(-1).values[keep].cpu().numpy()
    
    transformed_annotations = []
    for box, label, score in zip(boxes, labels, scores):
        label_name = CLASSES[label]
        if label_name == "car":
            x_center, y_center, width, height = box
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            
            transformed_annotations.append({
                'label': label_name,
                'bbox': (x1, y1, x2, y2),
                'score': float(score)
            })
    
    return apply_nms(transformed_annotations)


tracker = ObjectTracker(iou_threshold=0.5)

def process_video(video_path, output_path, annotation_file, output_txt):
    cap = cv2.VideoCapture(str(video_path))  # Ensure path is a string
    assert cap.isOpened(), f"Error: Cannot open video {video_path}"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    assert out.isOpened(), f"Error: Cannot open video writer for {output_path}"

    annotations = parse_cvat_xml(annotation_file)
    frame_idx = 0
    
    with open(output_txt, "w") as f:
        f.write("")  # Clear the output file

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        ground_truth = annotations.get(frame_idx, [])
        
        # Get the detections from DETR
        detections = detect_objects(frame)  # This will return the format suitable for tracking
        filtered_detections = filter_and_transform_detections(frame, detections, threshold=0.7)
        
        # Update the tracker with the detections
        tracked_objects = tracker.update(filtered_detections, frame_idx)

        # Save the MOTS format detections
        save_detections_mots(output_txt, tracked_objects, frame_idx)

        # Draw bounding boxes on the frame
        for obj_id, (bbox, _) in tracked_objects.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  
            cv2.putText(frame, f'ID {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        frame = draw_boxes(frame, ground_truth)
        
        out.write(frame)
        
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


    
# Example usage
if __name__ == "__main__":
    video_path = Path("AICity_data") / "train" / "S03" / "c010" / "vdo.avi"
    output_txt = "TrackEval\data/trackers\mot_challenge\s03_c010-train\PerfectTracker\data\s03_c010-01.txt"
    process_video(video_path, "output.avi", "ai_challenge_s03_c010-full_annotation.xml", output_txt=output_txt)
