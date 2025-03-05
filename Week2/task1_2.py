import random
from pathlib import Path
import torch
import numpy as np
import wandb
import optuna
from transformers import DetrForObjectDetection

from detr_model import run_model
from task1_1 import parse_cvat_xml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login(key="50315889c64d6cfeba1b57dc714112418a50e134")


def objective_model_cv(trial):
    params = {
        # 'batch_size': trial.suggest_categorical('batch_size', [16]),
        'img_size': trial.suggest_categorical('img_size', [800]),
        'lr': trial.suggest_categorical('lr', [0.0001, 0.001, 0.01, 0.1, 0.2]),  # 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3
        'optimizer': trial.suggest_categorical('optimizer', ['adadelta']),  # adadelta, adam, sgd, RMSprop
        # 'unfroze': trial.suggest_categorical('unfroze', [20]),

        # 'rot': trial.suggest_categorical('rot', [20]),
        # 'sr': trial.suggest_categorical('sr', [0]),
        # 'zr': trial.suggest_categorical('zr', [0.2]),
        # 'hf': trial.suggest_categorical('hf', [0.2]),

        # 'margin': trial.suggest_float('margin', 1.0, 1.0),

        'momentum': trial.suggest_float('momentum', 0.95, 0.95),
        # 'dropout': trial.suggest_categorical('dropout', ['0']),
        'epochs': trial.suggest_categorical('epochs', [50]),
        # 'output': trial.suggest_categorical('output', [2]),

        'detr_dim': trial.suggest_categorical('detr_dim', [256]),
        'freeze_backbone': trial.suggest_categorical('freeze_backbone', ['False']),
        'freeze_transformer': trial.suggest_categorical('freeze_transformer', ['False']),
        'freeze_bbox_predictor': trial.suggest_categorical('freeze_bbox_predictor', ['False']),
        'extra_layers': trial.suggest_categorical('extra_layers', ['True']),
        'k_fold': trial.suggest_categorical('k_fold', ['A'])
    }

    config = dict(trial.params)
    config['trial.number'] = trial.number

    execution_name = f'Detr_ft_xavi_new{str(trial.number)}'

    wandb.init(
        project='Detr_W1_ft',
        entity='mcv-c6g7',
        name=execution_name,
        config=config,
        reinit=True,
    )

    annotation_file = "ai_challenge_s03_c010-full_annotation.xml"
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    annotations = parse_cvat_xml(annotation_file)
    video_path = Path("AICity_data") / "train" / "S03" / "c010" / "vdo.avi"

    total_frames = 2141
    frame_indices = np.arange(total_frames)

    train_pctg = 0.25
    valid_pctg = 0.00

    train_frames_idx = frame_indices[:int(train_pctg * total_frames)].tolist()
    valid_frames_idx = frame_indices[int(train_pctg * total_frames):int((train_pctg + valid_pctg) * total_frames)].tolist()
    test_frames_idx = frame_indices[int((train_pctg + valid_pctg) * total_frames):].tolist()

    ratio = run_model(params,
              model,
              video_path=video_path,
              annotations=annotations,
              trial_number=0,
              train_frames_idx=train_frames_idx,
              valid_frames_idx=valid_frames_idx,
              test_frames_idx=test_frames_idx,
              num_labels=1)

    return ratio


study = optuna.create_study(direction="maximize", study_name='c6-Week1')
study.optimize(objective_model_cv, n_trials=10)