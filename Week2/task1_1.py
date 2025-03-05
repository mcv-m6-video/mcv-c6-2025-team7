import torch
import torchvision.transforms as T
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from transformers import DetrForObjectDetection
from ultralytics import YOLO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pretrained DETR model
#model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
model = YOLO("models/yolov8n.pt")
model.eval()

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

def compute_miou(detections, ground_truths):
    ious = [max([compute_iou(det, gt) for gt in ground_truths] or [0]) for det in detections]
    return np.mean(ious) if ious else 0

def compute_ap(recall, precision):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return np.trapz(precision, recall)

def compute_map50(detected_boxes, gt_boxes):
    tp, fp, fn = 0, 0, len(gt_boxes)
    ious = [max([compute_iou(det, gt) for gt in gt_boxes] or [0]) for det in detected_boxes]
    for iou in ious:
        if iou >= 0.5:
            tp += 1
            fn -= 1
        else:
            fp += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return compute_ap(np.array([recall]), np.array([precision]))


# Function to process a frame through DETR
def detect_objects(frame):
    img = transform(frame).unsqueeze(0).to(device) # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        print(outputs)
    return outputs

def detect_objects_yolo(frame):
    results = model(frame)  # Direct inference
    return results

def draw_boxes_yolo(frame, results, ground_truth, threshold=0.3):
    detected_bboxes = []  # Store detected bounding boxes

    # Draw detected boxes in RED
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get bounding box
            conf = box.conf[0].item()  # Confidence score
            label = int(box.cls[0].item())  # Class index
           

            if conf > threshold and model.names[label] == "car":
                print(x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for detected cars
                cv2.putText(frame, model.names[label], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                detected_bboxes.append((x1, y1, x2, y2))

    # Draw ground truth boxes in GREEN
    for gt in ground_truth:
        x1, y1, x2, y2 = gt['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for ground truth
        cv2.putText(frame, gt['label'], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, detected_bboxes

# Function to draw bounding boxes
def draw_boxes(frame, outputs, ground_truth, threshold=0.7):
    h, w, _ = frame.shape
    probas = outputs['logits'].softmax(-1)[0, :, :-1]  # Remove background class
    keep = probas.max(-1).values > threshold
    boxes = outputs['pred_boxes'][0, keep]  # Filter boxes
    labels = probas.argmax(-1)[keep]
    detected_bboxes = []  # List to store detected bounding boxes

    # Draw detected boxes in RED for all detected cars
    for box, label in zip(boxes, labels):
        x_center, y_center, width, height = box.cpu().numpy()
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)
        
        label_name = CLASSES[label]
        if label_name == "car":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for detected cars
            cv2.putText(frame, label_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            detected_bboxes.append((x1, y1, x2, y2))  # Store detected bounding box

    # Draw ground truth boxes in GREEN
    for gt in ground_truth:
        x1, y1, x2, y2 = gt['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for ground truth
        cv2.putText(frame, gt['label'], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, detected_bboxes

# Function to parse CVAT-style XML annotations
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


# Function to process video
def process_video(video_path, output_path, annotation_file):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    annotations = parse_cvat_xml(annotation_file)
    frame_idx = 0

    all_detections = []
    all_ground_truths = []
    miou_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        outputs = detect_objects_yolo(frame)
        ground_truth = annotations.get(frame_idx, [])
        frame, detected_boxes= draw_boxes_yolo(frame, outputs, ground_truth)
        out.write(frame)

        #print(detected_boxes)
        gt_boxes = [gt['bbox'] for gt in ground_truth]
        #print(gt_boxes)

        # Store detections & ground truths for overall evaluation
        all_detections.extend(detected_boxes)
        all_ground_truths.extend(gt_boxes)

        # Compute mIoU per frame
        frame_miou = compute_miou(detected_boxes, gt_boxes)
        miou_list.append(frame_miou)

        print(f"Frame {frame_idx}: mIoU = {frame_miou:.4f}")

        out.write(frame)
        cv2.imshow('DETR Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # Compute Overall Metrics
    overall_miou = np.mean(miou_list) if miou_list else 0
    overall_map50 = compute_map50(all_detections, all_ground_truths)

    print(f"Overall mIoU: {overall_miou:.4f}")
    print(f"Overall mAP@50: {overall_map50:.4f}")

# Example usage
if __name__ == "__main__":
    video_path = Path("AICity_data") / "train" / "S03" / "c010" / "vdo.avi"
    process_video(video_path, "yolo2.avi", "ai_challenge_s03_c010-full_annotation.xml")
