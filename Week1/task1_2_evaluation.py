
from utils import convert_xml_to_coco, segment_foreground, convert_detections_to_coco

import json
from pathlib import Path
import cv2

"""
Detectron 2 works with python 3.9, cuda 11.3 and torch 1.10, at least for my ubuntu 22.04
I am mentioning it because I spent a lot of time looking for compatibility, in case you
also need it.
"""
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import load_coco_json
from detectron2.data import DatasetCatalog, MetadataCatalog

"""
As fas as I know, Detectron works only with COCO formated datasets,
so the main idea was to convert the ground truth annotations and the \
detections into coco format
"""


# Convert XML annotations and save them as a COCO formated JSON
xml_file_path = "ai_challenge_s03_c010-full_annotation.xml"
coco_gt = convert_xml_to_coco(xml_file_path, {"car-bike": 1})

with open("ground_truth.json", "w") as f:
    json.dump(coco_gt, f, indent=4)

print("Ground truth annotations saved as ground_truth.json")

# Convert bounding boxes JSON to COCO formated JSON
with open("detected_bounding_boxes.json", "r") as f:
    detected_boxes = json.load(f) 

# Convert JSON keys (which are strings) to integers for consistency
detected_boxes = {int(frame): boxes for frame, boxes in detected_boxes.items()}

convert_detections_to_coco(
    detections="detected_bounding_boxes.json",
    ground_truth="ground_truth.json",
    output_json="detections_coco.json",
    categories={"car-bike": 1}
)

print("Predicted detections saved as detections_coco.json")

"""
The newest version from chat GPT suggested to create a dummy model here to feed the COCOEvaluator.
I have not tried it as it seemed too much
"""
# Compute AP with Detectron2
# Register dataset
DatasetCatalog.clear()
DatasetCatalog.register("ground_truth", lambda: load_coco_json("ground_truth_coco_format.json", ""))
DatasetCatalog.register("detections", lambda: load_coco_json("detections_coco.json", ""))

# Load metadata
MetadataCatalog.get("ground_truth").thing_classes = ["car-bike"]

# Initialize evaluator
evaluator = COCOEvaluator("ground_truth", ("bbox",), False, output_dir="./")
evaluator.iou_thresholds = [0.5] 
evaluator.reset()

# Evaluate using Detectron2
results = evaluator.evaluate()

# Save results to a text file
output_txt_path = "evaluation_results.txt"
with open(output_txt_path, "w") as file:
    file.write(json.dumps(results, indent=4))  # Format results in readable JSON

print(f"The obtained results are: {results} \n which are saved to {output_txt_path}")
