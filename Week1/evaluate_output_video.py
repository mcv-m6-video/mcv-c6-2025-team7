import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# --- VOC Evaluation Functions --- (from the second code)
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def voc_eval(preds, gt, ovthresh=0.5):
    class_recs = {}
    npos = 0

    for i, frame in enumerate(gt):
        bbox = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]] for bbox in frame])
        difficult = np.array([False for bbox in frame]).astype(bool)
        det = [False] * len(frame)
        npos = npos + sum(~difficult)
        class_recs[i] = {"bbox": bbox, "difficult": difficult, "det": det}

    image_ids = []
    confidence = []
    BB = []

    for i, frame in enumerate(preds):
        image_ids += [i] * len(preds[i])
        confidence += list(np.random.rand(len(preds[i])))
        BB += [[bbox[0], bbox[1], bbox[2], bbox[3]] for bbox in preds[i]]

    confidence = np.array(confidence)
    BB = np.array(BB).reshape(-1, 4)

    if np.all(confidence != None):
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    iou = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            overlaps = [compute_iou(bb, gt_bbox) for gt_bbox in BBGT]
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            iou[d] = ovmax

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.mean(prec)
    iou = np.mean(iou)

    return rec, prec, ap

def plot_prec_recall_curve(prec, rec, title="Precision-Recall curve", save_name="model.png"):
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.savefig(save_name)
    plt.close()


# --- Ground Truth Parsing --- (from your code)
xml_file = "ai_challenge_s03_c010-full_annotation.xml"
tree = ET.parse(xml_file)
root = tree.getroot()

annotations = {}
for track in root.findall(".//track"):
    label = track.get("label")
    if label != "car":
        continue

    for box in track.findall("box"):
        frame = int(box.get("frame"))
        xtl, ytl, xbr, ybr = map(float, [box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")])

        parked = box.find(".//attribute[@name='parked']")
        if parked is not None and parked.text.strip().lower() == "false":
            if frame not in annotations:
                annotations[frame] = []
            annotations[frame].append((label, int(xtl), int(ytl), int(xbr), int(ybr)))



def get_bbox_from_single_image(image, kernel_open=5, kernel_close=30):
    final_bounding_boxes = []

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_open, kernel_open)))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_close, kernel_close)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    for i in range(1, num_labels):  # Ignore background label (0)
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if 250 < area < 200000:
            ratio = w / h
            if ratio > 0.8:  # To discard persons
                final_bounding_boxes.append([x, y, x + w, y + h])

    return final_bounding_boxes

def evaluate_video(video_file: str):
    # --- Main Evaluation Loop ---
    cap = cv2.VideoCapture(video_file)
    frame_idx = 535
    pred_bboxes = []
    gt_bboxes = []

    roi = cv2.imread("AICity_data/train\S03\c010/roi.jpg", cv2.IMREAD_GRAYSCALE)
    min_area_threshold = 1000


    # Initialize video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change this to another codec if needed
    output_video_file = video_file[:-4] + '_output_with_bboxes.avi'
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, binary_image = cv2.threshold(frame_gray, 127, 255, cv2.THRESH_BINARY)
        binary_image = cv2.bitwise_and(binary_image, roi)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Get predicted bounding boxes using get_bbox_from_single_image
        bounding_boxes = get_bbox_from_single_image(frame)
        pred_bboxes.append(bounding_boxes)

        # Get ground truth bounding boxes
        gt_bboxes.append([bbox[1:] for bbox in annotations.get(frame_idx, [])])

        # Visualize bounding boxes
        for xtl, ytl, xbr, ybr in gt_bboxes[-1]:
            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)  # Green for GT

        for xtl, ytl, xbr, ybr in bounding_boxes:
            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 0, 255), 2)  # Red for detected

        out.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    out.release()
    # Evaluate using VOC
    rec, prec, ap = voc_eval(preds=pred_bboxes, gt=gt_bboxes)
    print(f"Recall: {rec[-1]:.4f}, Precision: {prec[-1]:.4f}, Average Precision: {ap:.4f}")

    # Plot Precision-Recall Curve
    plot_prec_recall_curve(prec, rec, title="Precision-Recall Curve", save_name="pr_curve.png")

    return rec, prec, ap

if __name__ == "__main__":
    video_path = "Output_Videos\AdaptiveModelling/task_2_1_mean_alpha9_rho0.4.avi"
    evaluate_video(video_path)