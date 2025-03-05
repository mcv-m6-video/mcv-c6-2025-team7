from pathlib import Path

import numpy as np
import wandb
from transformers import DetrForObjectDetection

from detr_model import run_model
from task1_1 import parse_cvat_xml


wandb.login(key="50315889c64d6cfeba1b57dc714112418a50e134")


params = {
    'img_size': 800,
    'lr': 0.01,
    'optimizer': 'adadelta',

    'momentum': 0.95,
    'epochs': 50,

    'detr_dim': 256,
    'freeze_backbone': 'False',
    'freeze_transformer': 'False',
    'freeze_bbox_predictor': 'False',
    'extra_layers': 'True',
    'k_fold': 'A'
}

config = dict(params)

execution_name = f'K_Fold_Gerard_1'

wandb.init(
    project='Detr_W1_ft',
    entity='mcv-c6g7',
    name=execution_name,
    config=config,
    reinit=True,
)

annotation_file = "ai_challenge_s03_c010-full_annotation.xml"
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
annotations = parse_cvat_xml(annotation_file)
video_path = Path("AICity_data") / "train" / "S03" / "c010" / "vdo.avi"

### K-Fold Cross-validation

total_frames = 2141
frame_indices = np.arange(total_frames)

k_fold_strategy = "A"
assert k_fold_strategy in ["A", "B", "C"]

if k_fold_strategy == "A":
    # Strategy A: fixed percentage splits
    train_pctg = 0.25
    valid_pctg = 0.00

    train_frames_idx = frame_indices[:int(train_pctg * total_frames)].tolist()
    valid_frames_idx = frame_indices[int(train_pctg * total_frames):int((train_pctg + valid_pctg) * total_frames)].tolist()
    test_frames_idx = frame_indices[int((train_pctg + valid_pctg) * total_frames):].tolist()

    run_model(params,
              model,
              video_path=video_path,
              annotations=annotations,
              trial_number=0,
              train_frames_idx=train_frames_idx,
              valid_frames_idx=valid_frames_idx,
              test_frames_idx=test_frames_idx,
              num_labels=1)

elif k_fold_strategy == "B":
    # Strategy B: Fixed K-Fold (K=4)
    k = 4
    fold_size = total_frames // k
    folds = [frame_indices[i * fold_size:(i + 1) * fold_size].tolist() for i in range(4)]

    # No support for validation frames
    valid_frames_idx = []

    for train_fold_idx in range(k):
        train_frames_idx = folds[train_fold_idx]
        test_frames_idx = []
        for i, fold in enumerate(folds):
            if i != train_fold_idx:
                test_frames_idx += fold

        run_model(params,
                  model,
                  video_path=video_path,
                  annotations=annotations,
                  trial_number=0,
                  train_frames_idx=train_frames_idx,
                  valid_frames_idx=valid_frames_idx,
                  test_frames_idx=test_frames_idx,
                  num_labels=1)

elif k_fold_strategy == "C":
    # Strategy C: Random K-Fold (K=4)
    np.random.seed(42)
    random_frame_indices = np.random.permutation(frame_indices)

    k = 4
    fold_size = total_frames // k
    folds = [random_frame_indices[i * fold_size:(i + 1) * fold_size].tolist() for i in range(4)]

    # No support for validation frames
    valid_frames_idx = []

    for train_fold_idx in range(k):
        train_frames_idx = folds[train_fold_idx]
        test_frames_idx = []
        for i, fold in enumerate(folds):
            if i != train_fold_idx:
                test_frames_idx += fold

        run_model(params,
                  model,
                  video_path=video_path,
                  annotations=annotations,
                  trial_number=0,
                  train_frames_idx=train_frames_idx,
                  valid_frames_idx=valid_frames_idx,
                  test_frames_idx=test_frames_idx,
                  num_labels=1)
