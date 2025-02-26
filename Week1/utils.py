import json
from pathlib import Path

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, Dict, List  # Python 3.9-compatible typing

from matplotlib import pyplot as plt
from tqdm import tqdm


# ---------------------------------------------------------------------------------
#                               TASK 1 FUNCTIONS
#----------------------------------------------------------------------------------
def get_total_frames(video_path: str) -> int:
    """Get the total number of frames in a video file.

    This function opens a video file using OpenCV, retrieves the total frame count, 
    and then releases the video capture object.

    Args:
        video_path (str): Path to the video file.

    Returns:
        int: The total number of frames in the video.

    Example:
        >>> get_total_frames("video.mp4")
        2400  # Example output
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def get_frames(video_path: str, start: int = 0, end: Optional[int] = None) -> np.ndarray:
    """Extracts video frames as grayscale images and returns them as a NumPy array.

    This function reads frames from the video file between 'start' and 'end' (exclusive),
    converts them to grayscale, and stores them in a NumPy array.

    Args:
        video_path (str): Path to the video file.
        start (int, optional): The starting frame index (default is 0).
        end (int | None, optional): The ending frame index (exclusive). If None, 
            it defaults to the total number of frames in the video.

    Returns:
        np.ndarray: A NumPy array of shape (num_frames, height, width) containing 
        grayscale frames as float32 values.

    Example:
        >>> frames = get_frames("video.mp4", start=10, end=20)
        >>> frames.shape
        (10, 720, 1280)  # Example output (height and width depend on the video)

    Notes:
        - The function reads frames sequentially, so larger ranges may take more time.
        - If 'start' is greater than the total frame count, an empty array is returned.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    if end is None:
        end = get_total_frames(video_path)  # Get total frames if not specified
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    # Read the frames between start and end frame numbers
    for _ in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    
    cap.release()
    return np.array(frames, dtype=np.float32)


def estimate_background(video_path: str, 
                        percent: float = 0.25, 
                        use_median: bool = False
                        ) -> Tuple[Optional[np.ndarray],Optional[np.ndarray], int]:
    """Estimate the background of a video using the first 'percent' of frames.

    This function computes a background model by analyzing a portion of the video frames. 
    The background can be estimated using either the mean or median of the selected frames. 
    Additionally, the function calculates the standard deviation of the frames.

    Args:
        video_path (str): Path to the video file.
        percent (float, optional): The fraction (0 to 1) of the total frames to use 
            for background estimation (default is 0.25).
        use_median (bool, optional): If True, uses the median instead of the mean for 
            background estimation (default is False).

    Returns:
        tuple:
            - np.ndarray | None: The estimated background (mean or median) in grayscale, 
              or None if no frames were read.
            - np.ndarray | None: The standard deviation of the frames, or None if no 
              frames were read.
            - int: The number of frames used for background estimation.

    Example:
        >>> bg, std, num_frames = estimate_background("video.mp4", percent=0.2, use_median=True)
        Total frames: 1000, Using the first 200 for background estimation
        >>> bg.shape, std.shape, num_frames
        ((720, 1280), (720, 1280), 200)  # Example output (depends on video resolution)

    Notes:
        - If 'percent' is too small, the background estimate may be noisy.
        - If no frames are read, the function returns '(None, None, num_frames)'.
        - Ensure the video file is accessible and readable.
    """
    total_frames = get_total_frames(video_path)  # Total frames in the video
    num_bg_frames = int(total_frames * percent)  # Number of frames used for background estimation

    print(f"Total frames: {total_frames}, Using the first {num_bg_frames} for background estimation")

    frames = get_frames(video_path, start=0, end=num_bg_frames)  # Read the first `num_bg_frames` frames

    if len(frames) == 0:  # No frames read
        print("Error: No frames were read for background estimation!")
        return None, None, num_bg_frames

    # Estimate background: Mean or Median
    if use_median:
        mean_bg = np.median(frames, axis=0)
    else:
        mean_bg = np.mean(frames, axis=0)
    
    # Calculate standard deviation for background estimation
    std_bg = np.std(frames, axis=0)

    return mean_bg, std_bg, num_bg_frames

# def segment_foreground(
#     video_path: str,
#     mean_bg: np.ndarray,
#     std_bg: np.ndarray,
#     num_bg_frames: int,
#     alpha: float = 2.5,
#     output_path: Optional[str] = None,
#     show_video: bool = False
# ) -> None:
#     """Segments the foreground in a video based on background estimation.

#     This function processes a video by comparing each frame to a precomputed background model.
#     It identifies foreground pixels where the difference from the background exceeds 
#     `alpha * std_bg`. The resulting binary mask highlights moving objects.

#     The function optionally saves the output as a video and can display the segmentation in real time.

#     Args:
#         video_path (str): Path to the input video file.
#         mean_bg (np.ndarray): The estimated background image (grayscale).
#         std_bg (np.ndarray): The standard deviation of the background model.
#         num_bg_frames (int): The number of frames used to compute the background.
#         alpha (float, optional): Threshold multiplier for background subtraction (default is 2.5).
#         output_path (str | None, optional): Path to save the output segmented video.
#             If None, the video is not saved (default is None).
#         show_video (bool, optional): If True, displays the segmented foreground in real time (default is False).

#     Returns:
#         None: The function does not return a value but can save an output video.

#     Example:
#         >>> segment_foreground("video.mp4", mean_bg, std_bg, num_bg_frames, alpha=3.0, output_path="output.avi", show_video=True)

#     Notes:
#         - The function processes frames starting from `num_bg_frames` to the end of the video.
#         - Foreground detection is based on the absolute difference exceeding `alpha * std_bg + 2`.
#         - Press 'ESC' to stop the video display early.
#     """
#     total_frames = get_total_frames(video_path)  # Total frames in the video
#     frames = get_frames(video_path, start=num_bg_frames, end=total_frames)  # Read frames after background frames

#     # Create the output video writer if saving
#     if output_path:
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi
#         fps = 30  # Frame rate
#         out_video = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

#     for frame in frames:
#         # Create foreground mask by comparing frame with background
#         foreground_mask = np.abs(frame - mean_bg) >= (alpha * (std_bg + 2))
#         fg_binary = (foreground_mask * 255).astype(np.uint8)

#         # Show the foreground binary mask if show_video is True
#         if show_video:
#             cv2.imshow("Foreground Mask", fg_binary)

#         # Write the processed frame to the output video
#         if output_path:
#             out_video.write(cv2.cvtColor(fg_binary, cv2.COLOR_GRAY2BGR))

#         # Break on pressing 'ESC'
#         if cv2.waitKey(30) & 0xFF == 27:
#             break
    
#     cv2.destroyAllWindows()

#     # Release video writer if it was created
#     if output_path:
#         out_video.release()


def segment_foreground(
    video_path: str,
    mean_bg: np.ndarray,
    std_bg: np.ndarray,
    num_bg_frames: int,
    alpha: float = 2.5,
    output_path: str = None,
    bbox_output_json: str = "detected_bounding_boxes.json",
    show_video: bool = False
):
    """Segments foreground and extracts bounding boxes while processing frames."""

    total_frames = get_total_frames(video_path)
    frames = get_frames(video_path, start=num_bg_frames, end=total_frames)

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = 14
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

    detected_bboxes = {}  # Store bounding boxes per frame
    frame_number = num_bg_frames  # Start from the correct frame index

    # -------------------------------------------------------------------------------------
    #    This section is for task 1.2. The idea was to detect the bboxes while modeling to 
    #    have better frame-detected object alignment
    #---------------------------------------------------------------------------------------
    for frame in frames:
        # Foreground detection using background subtraction
        foreground_mask = np.abs(frame - mean_bg) >= (alpha * (std_bg + 2))
        fg_binary = (foreground_mask * 255).astype(np.uint8)

        # Create a black background
        black_background = np.zeros_like(fg_binary)

        # Find contours
        contours, _ = cv2.findContours(fg_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if w > 50 and h > 50 and area > 1100:  # Ignore small noise
                frame_bboxes.append([x, y, x + w, y + h])  # (x_min, y_min, x_max, y_max)

                cv2.rectangle(black_background, (x, y), (x + w, y + h), (255, 255, 255), 2)

        if frame_bboxes:
            detected_bboxes[frame_number] = frame_bboxes  # Save bounding boxes

        # -------------------------------------------------------------------------------------
        #    It finishes here
        #---------------------------------------------------------------------------------------

        if show_video:
            cv2.imshow("Foreground Mask", black_background)  # Show white bbox on black background

        if output_path:
            out_video.write(cv2.cvtColor(fg_binary, cv2.COLOR_GRAY2BGR))

        if cv2.waitKey(30) & 0xFF == 27:
            break

        frame_number += 1  # Increment frame index

    cv2.destroyAllWindows()

    if output_path:
        out_video.release()

    # Also from task 1.2, Save bounding boxes to JSON
    with open(bbox_output_json, "w") as f:
        json.dump(detected_bboxes, f, indent=4)


# ---------------------------------------------------------------------------------
#                               TASK 2 FUNCTIONS
#----------------------------------------------------------------------------------

def convert_xml_to_coco(xml_file: str, categories: Dict[str, int]) -> Dict:
    """Converts an XML annotation file (CVAT format) to COCO format.

    This function parses an XML file containing object annotations, filters out 
    non-car objects and parked cars, and converts the remaining annotations into 
    COCO format.

    Args:
        xml_file (str): Path to the XML annotation file.
        categories (dict[str, int]): Mapping of category names to COCO category IDs.

    Returns:
        dict: A dictionary in COCO format containing:
            - "images" (list[dict]): Metadata for each frame with annotations.
            - "annotations" (list[dict]): Bounding box annotations in COCO format.
            - "categories" (list[dict]): List of COCO category definitions.

    Example:
        >>> categories = {"car": 1}
        >>> coco_data = convert_xml_to_coco("annotations.xml", categories)
        >>> print(coco_data.keys())
        dict_keys(['images', 'annotations', 'categories'])

    Notes:
        - Only annotations labeled as "car" are included.
        - Parked cars are ignored based on the "parked" attribute.
        - The function assumes fixed image dimensions (1920x1080).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    images = []
    annotations = []
    annotation_id = 1
    frame_offset = 536  # 0 to 535 used for background estimation and not in modelled video

    for track in root.findall("track"):
        label = track.get("label")

        for box in track.findall("box"):
            frame = int(box.get("frame"))

            # We ignore them as they are not in modelled video
            if frame < frame_offset:
                continue

            xtl, ytl, xbr, ybr = map(float, [box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")])
            if label == "car":
                parked = box.find("attribute[@name='parked']").text
                if parked == "true":  # Ignore parked cars
                    continue
            
            # Convert all labels into car-bike as we work with only one class
            label = "car-bike"
            
            category_id = categories[label]
          
            image_info = {"id": frame, "file_name": f"{frame}.jpg", "height": 1080, "width": 1920}
            if image_info not in images:
                images.append(image_info)

            bbox = [xtl, ytl, xbr - xtl, ybr - ytl]  # COCO format: (x_min, y_min, width, height)
            annotation = {
                "id": annotation_id,
                "image_id": frame,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
            annotations.append(annotation)
            annotation_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "car-bike"}],
    }

    return coco_format


# def extract_bounding_boxes(video_path: str, 
#                            output_json: str, 
#                            threshold: int=127,
#                            min_area: int=10) -> None:
#     """
#     Extracts bounding boxessss from a binary-segmented video.

#     This function processes an input video, applying grayscale conversion 
#     and thresholding to detect objectsss. It then extracts bounding boxes 
#     around the detected regions and stores them in a JSON file.

#     Args:
#         video_filename (str): The name of the input video file, located in the "Output_Videos" directory.
#         output_json (str, optional): The filename for saving the detected bounding boxes as a JSON file. 
#                                      Defaults to "detected_bounding_boxes.json".
#         threshold (int, optional): The binarization threshold for segmenting the video frames. 
#                                    Defaults to 127.

#     Returns:
#         dict: A dictionary mapping frame numbers to a list of bounding box tuples. 
#               Each tuple is in the format (x_min, y_min, x_max, y_max).

#     Example:
#         >>> boxes = extract_bounding_boxes("example_video.mp4")
#         Processing video: Output_Videos/example_video.mp4
#         Extracted bounding boxes saved as detected_bounding_boxes.json
#     """

#     cap = cv2.VideoCapture(video_path)
#     detected_boxes = {}
#     frame_number = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         frame_bboxes = []
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             area = w * h
            
#             # Ignore very small objects (noise)
#             if w > 1 and h > 1 and area > min_area:  
#                 frame_bboxes.append((x, y, x + w, y + h))  # Store (x_min, y_min, x_max, y_max)

#         # Save bounding boxes for this frame
#         if frame_bboxes:
#             detected_boxes[frame_number] = frame_bboxes

#         frame_number += 1

#     cap.release()

#     # Save detected bounding boxes as a JSON file
#     with open(output_json, "w") as f:
#         json.dump(detected_boxes, f, indent=4)

#     return output_json


def convert_detections_to_coco(
    detections: str, 
    ground_truth: str, 
    output_json: str, 
    categories: Dict[str, int],
    frame_offset: int=535
) -> Dict:
    """Converts object detections into COCO format with ground truth validation.

    This function processes a JSON file of detections and filters them based on 
    valid frame IDs from a ground truth dataset. It then converts the detections 
    into COCO format and saves the output as a JSON file.

    Args:
        detections (str): Path to a JSON file containing detected bounding boxes.
            The JSON should be structured as {frame_id: [[x_min, y_min, x_max, y_max], ...]}.
        ground_truth (str): Path to a ground truth JSON file in COCO format.
            The function ensures that detections only include frames listed in 
            `ground_truth["images"]`.
        output_json (str): Path to save the output JSON file in COCO format.
        categories (Dict[str, int]): Mapping of category names to COCO category IDs.

    Returns:
        Dict: A dictionary in COCO format containing:
            - "images" (list[Dict]): Metadata for each annotated frame.
            - "annotations" (list[Dict]): Bounding box annotations in COCO format.
            - "categories" (list[Dict]): List of COCO category definitions.

    Example:
        >>> categories = {"car-bike": 1}
        >>> coco_data = convert_detections_to_coco("detections.json", "ground_truth.json", "output.json", categories)
        >>> print(coco_data.keys())
        dict_keys(['images', 'annotations', 'categories'])

    Notes:
        - Frames not present in `ground_truth["images"]` are ignored.
        - Bounding boxes are converted to COCO format: (x_min, y_min, width, height).
        - The output JSON file is written to `output_json`.
    """
    # Load detected bounding boxes
    with open(detections, "r") as f:
        detections = json.load(f)

    # Load ground truth image IDs
    with open(ground_truth, "r") as f:
        gt_data = json.load(f)

    gt_image_ids = {img["id"] for img in gt_data["images"]}  # Set of valid frame IDs

    images = []
    annotations = []
    annotation_id = 1

    for frame, boxes in detections.items():
        frame = int(frame)  # Convert frame to integer
        # frame = frame + frame_offset

        if frame not in gt_image_ids:
            print(f"Skipping frame {frame} (not in ground truth).")
            continue  # Skip frames that are not in ground truth

        image_info = {"id": frame, "file_name": f"{frame}.jpg", "height": 1080, "width": 1920}
        images.append(image_info)

        for box in boxes:
            bbox = [box[0], box[1], box[2] - box[0], box[3] - box[1]]  # Convert to COCO format
            annotation = {
                "id": annotation_id,
                "image_id": frame,  # Use corrected frame number
                "category_id": categories["car-bike"],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
            annotations.append(annotation)
            annotation_id += 1

    coco_format = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "car-bike"}]}

    with open(output_json, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"Fixed detections saved to {output_json}")
    return coco_format


def visualize_estimated_background(mean_bg: np.ndarray, std_bg: np.ndarray, save_path: str | None = None) -> None:
    """
    Visualizes the estimated background mean and standard deviation side by side.

    Args:
        mean_bg (np.ndarray): 2D array of the background mean (grayscale).
        std_bg (np.ndarray): 2D array of the background standard deviation.
        save_path (str | None): Path to save the output plot. If None, displays the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))

    # Mean background subplot
    plt.subplot(1, 2, 1)
    plt.title("Estimated Background Mean")
    plt.imshow(mean_bg, cmap='gray')
    plt.colorbar()
    plt.axis('off')

    # Standard deviation subplot
    plt.subplot(1, 2, 2)
    plt.title("Estimated Background Std Dev")
    plt.imshow(std_bg, cmap='hot')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def adaptive_segment_foreground(
    video_path: str,
    mean_bg: np.ndarray,
    std_bg: np.ndarray,
    num_bg_frames: int,
    alpha: float = 2.5,
    rho: float = 0.01,
    output_path: str | None = None,
    show_video: bool = False
) -> None:
    """
    Segments the foreground using adaptive background modeling.

    Args:
        video_path (str): Path to the input video file.
        mean_bg (np.ndarray): The estimated background image (grayscale).
        std_bg (np.ndarray): The standard deviation of the background model.
        num_bg_frames (int): The number of frames used to compute the background.
        alpha (float, optional): Threshold multiplier for background subtraction (default is 2.5).
        rho (float): Learning rate for adaptive modeling (0 < rho <= 1).
        output_path (str | None, optional): Path to save the output segmented video.
            If None, the video is not saved (default is None).
        show_video (bool, optional): If True, displays the segmented foreground in real time (default is False).

    Returns:
        None: The function does not return a value but can save an output video.

    Example:
        >>> segment_foreground("video.mp4", mean_bg, std_bg, num_bg_frames, alpha=3.0, rho=1e-2, output_path="output.avi", show_video=True)

    Notes:
        - The function processes frames starting from `num_bg_frames` to the end of the video.
        - Foreground detection is based on the absolute difference exceeding `alpha * std_bg + 2`.
        - Press 'ESC' to stop the video display early.
    """
    total_frames = get_total_frames(video_path)  # Total frames in the video
    frames = get_frames(video_path, start=num_bg_frames, end=total_frames)  # Read frames after background frames

    # Create the output video writer if saving
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi
        fps = 30  # Frame rate
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

    pbar = tqdm(enumerate(frames), total=total_frames)
    for i, frame in pbar:
        # Create foreground mask by comparing frame with background
        foreground_mask = np.abs(frame - mean_bg) >= (alpha * (std_bg + 2))
        fg_binary = (foreground_mask * 255).astype(np.uint8)

        # Update background model (for background pixels only)
        background_mask = ~foreground_mask
        mean_bg[background_mask] = (rho * frame[background_mask] + (1 - rho) * mean_bg[background_mask])
        std_bg[background_mask] = np.sqrt((rho * (frame[background_mask] - mean_bg[background_mask])**2 +
                                   (1 - rho) * std_bg[background_mask]**2))

        # Uncomment to save estimated backgrounds
        # estimated_background_folder = Path("Output_Videos") / "AdaptativeModelling" / "estimated_background"
        # estimated_background_folder.mkdir(parents=True, exist_ok=True)
        # visualize_estimated_background(
        #     mean_bg, std_bg, save_path=(estimated_background_folder / f"estimated_background_iter_{i}.png").as_posix()
        # )

        # Show the foreground binary mask if show_video is True
        if show_video:
            cv2.imshow("Foreground Mask", fg_binary)

        # Write the processed frame to the output video
        if output_path:
            out_video.write(cv2.cvtColor(fg_binary, cv2.COLOR_GRAY2BGR))

        # Break on pressing 'ESC'
        if cv2.waitKey(30) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()
