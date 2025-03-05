import torch
import cv2
from typing import Any, Optional
from transformers import DetrForObjectDetection

def prepare_frame_for_detr(
    frame,
    frame_idx: int,
    annotations: dict[int, list[dict]],
    transform,
    device: torch.device,
    car_category_id: int
) -> tuple[Optional[torch.Tensor], Optional[list[dict]]]:
    """
    Utility that:
      - transforms a BGR frame into the model input tensor
      - extracts bounding boxes for “car” from annotations[frame_idx]
      - converts them to the huggingface DETR format
    Returns (input_tensor, labels_list):
      - input_tensor: shape [1, C, H, W] or None if no boxes
      - labels_list: a list of length=1: [{"class_labels": Tensor, "boxes": Tensor}] or None if no boxes
    """
    h, w, _ = frame.shape

    input_tensor = transform(frame).unsqueeze(0).to(device)

    # Get CVAT annotations for this frame
    gt_list = annotations.get(frame_idx, [])
    boxes_list = []
    labels_list = []

    for ann in gt_list:
        if ann["label"] == "car":
            (x1, y1, x2, y2) = ann["bbox"]
            # Convert to normalized corner format [x0, y0, x1, y1]
            boxes_list.append([x1 / w, y1 / h, x2 / w, y2 / h])
            labels_list.append(car_category_id)

    # If no car boxes skip frame
    if not boxes_list:
        return None, None

    boxes_tensor = torch.tensor(boxes_list, dtype=torch.float, device=device)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=device)

    # HuggingFace DETR expects a list of length = batch_size.
    labels_dict = [{
        "class_labels": labels_tensor,
        "boxes": boxes_tensor
    }]

    return input_tensor, labels_dict


class Trainer:
    def __init__(
        self,
        video_path: str,
        train_frames_idx: list,
        valid_frames_idx: list,
        test_frames_idx: list,
        annotation_file: str,
        transform,  
        car_category_id: int = 1
    ) -> None:
        """
        :param video_path: path to a video file (e.g. "vdo.avi")
        :param annotation_file: path to CVAT XML (e.g. "ai_challenge_s03_c010-full_annotation.xml")
        :param transform: a transform that takes a NumPy BGR frame -> PyTorch tensor
        :param car_category_id: label ID used for 'car'
        """
        self.video_path = video_path
        self.annotations = annotation_file
        self.transform = transform
        self.car_category_id = car_category_id
        self.train_frames_idx = train_frames_idx
        self.valid_frames_idx = valid_frames_idx
        self.test_frames_idx = test_frames_idx

    def train(
        self,
        model: DetrForObjectDetection,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> tuple[float, float]:
        """
        Runs a training pass on every frame in `self.video_path`.
        For each frame:
          1) transform the frame
          2) build bounding-box tensors for 'car' from the CVAT annotations
          3) pass them through the model
          4) compute loss, backprop, step

        Returns (avg_loss, avg_accuracy). 
        Accuracy for detection tasks is often 0 or omitted, but we return it for API consistency.
        """
        
        model.train().to(device)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video:", self.video_path)
            return 0.0, 0.0

        frame_idx = 0
        total_loss = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in self.train_frames_idx:
                # Prepare frame for Detr
                input_tensor, labels_list = prepare_frame_for_detr(
                    frame=frame,
                    frame_idx=frame_idx,
                    annotations=self.annotations,
                    transform=self.transform,
                    device=device,
                    car_category_id=self.car_category_id
                )

                # If no annotations for this frame, skip
                if input_tensor is None:
                    frame_idx += 1
                    continue

                # Forward pass & compute loss
                optimizer.zero_grad()
                outputs = model(pixel_values=input_tensor, labels=labels_list)
                loss = outputs.loss

                # 5. Backprop
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            frame_idx += 1

        cap.release()

        frames_used = len(self.train_frames_idx)
        avg_loss = total_loss / frames_used if frames_used > 0 else 0.0
        return avg_loss, 0.0


    def validation(
        self,
        model: DetrForObjectDetection,
        device: torch.device
    ) -> tuple[float, float]:
        """
        Evaluates the model on each frame in self.video_path (just like 'Trainer', but no backprop).
        Returns (avg_loss, avg_accuracy).
        
        In detection tasks, "accuracy" is usually replaced by mAP or IoU-based metrics. 
        Here we return a placeholder of 0.0 for accuracy to keep the interface consistent.
        """
        model.eval().to(device)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video:", self.video_path)
            return 0.0, 0.0

        frame_idx = 0
        total_loss = 0.0

        # We do not track gradients during validation
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # No more frames or error
                
                if frame_idx in self.valid_frames_idx:
                    input_tensor, labels_list = prepare_frame_for_detr(
                        frame=frame,
                        frame_idx=frame_idx,
                        annotations=self.annotations,
                        transform=self.transform,
                        device=device,
                        car_category_id=self.car_category_id
                    )
                    
                    # If no annotations for this frame, skip
                    if input_tensor is None:
                        frame_idx += 1
                        continue

                    # Forward pass
                    outputs = model(pixel_values=input_tensor, labels=labels_list)
                    loss = outputs.loss

                    total_loss += loss.item()
                frame_idx += 1

        cap.release()

        frames_used = len(self.valid_frames_idx)
        avg_loss = total_loss / frames_used if frames_used > 0 else 0.0
        # For detection tasks, "accuracy" is typically not relevant, so we return 0.0
        return avg_loss, 0.0
    
    def test(
        self,
        model: torch.nn.Module,
        device: torch.device
    ) -> tuple[float, float]:
        """
        Evaluates the model on test frames.
        Returns (avg_loss, avg_accuracy).
        """
        model.eval()  # Set model to evaluation mode
        model.to(device)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening test video:", self.video_path)
            return 0.0, 0.0

        frame_idx = 0
        total_loss = 0.0

        all_predicted_boxes = []  # list for predicted boxes per frame
        # all_ground_truth_boxes = []  # list for ground truth boxes per frame

        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # No more frames
                
                if frame_idx in self.test_frames_idx:
                    # Prepare frame
                    input_tensor, labels_dict = prepare_frame_for_detr(
                        frame=frame,
                        frame_idx=frame_idx,
                        annotations=self.annotations,
                        transform=self.transform,
                        device=device,
                        car_category_id=self.car_category_id
                    )

                    if input_tensor is None:
                        # all_predicted_boxes.append([])
                        # all_ground_truth_boxes.append([])
                        frame_idx += 1
                        continue
                    
                    # gt_boxes = labels_dict[0]['boxes'].cpu().numpy()  # shape: [N, 4]
                    # all_ground_truth_boxes.append(gt_boxes.tolist())
                    
                    outputs = model(pixel_values=input_tensor, labels=labels_dict)

                    h, w, _ = frame.shape
                    probas = outputs['logits'].softmax(-1)[0]
                    keep = probas[:, 1] > 0.7
                    boxes = outputs['pred_boxes'][0, keep]  # Filter boxes
                    labels = probas.argmax(-1)[keep]
                    detected_bboxes = []  # List to store detected bounding boxes

                    # Forward pass (compute loss only, no backprop)

                    loss = outputs.loss
                    total_loss += loss.item()
                    # print(labels)
                    
                    for box, label in zip(boxes, labels):
                        # print("Raw box tensor:", box)
                        # print("Converted to numpy:", box.cpu().numpy())
                        # print("Label value:", label.item())

                        if int(label.item()) == self.car_category_id:
                            x_center, y_center, width, height = box.cpu().numpy()
                            x1 = int((x_center - width / 2) * w)
                            y1 = int((y_center - height / 2) * h)
                            x2 = int((x_center + width / 2) * w)
                            y2 = int((y_center + height / 2) * h)
                            detected_box = (x1, y1, x2, y2)
                            detected_bboxes.append(detected_box)
                            assert isinstance(detected_box, tuple) and len(detected_box) == 4
                    
                    all_predicted_boxes.append(detected_bboxes)  # Store detected bounding box

                    # threshold = 0.5  # adjust as needed
                    # keep = scores > threshold
                    # pred_boxes = outputs.pred_boxes[0, scores].cpu().numpy()  # shape: [M, 4]
                    # all_predicted_boxes.append(pred_boxes.tolist())


                frame_idx += 1

        cap.release()

        frames_used = len(self.test_frames_idx)
        avg_test_loss = total_loss / frames_used if frames_used > 0 else 0.0
        return avg_test_loss, all_predicted_boxes