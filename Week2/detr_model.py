
from typing import Union
from pathlib import Path

import wandb
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as T
from transformers import DetrForObjectDetection, PreTrainedModel

from utils.utils import get_optimizer
from utils.train_val import Trainer


# Define a random seed for torch, numpy and cuda. This will ensure reproducibility
# and help obtain same results (wights inits, data splits, etc.). 
# This way we delete the posibility of a run obtaining better results by random chance
# rather than by hyperparameter tweaking.


def freeze_layers(params,
                  model,
                  num_labels: int,
                  hidden_layer_dim: int = 256) -> DetrForObjectDetection:
    
    model.config.num_labels = num_labels
    model.class_labels_classifier = torch.nn.Linear(
        model.config.d_model, 
        model.config.num_labels + 1
    )

    # Freeze model's different components
    if params['freeze_backbone'] == 'True':
        for param in model.model.backbone.parameters():
            param.requires_grad = False

    if params['freeze_transformer'] == 'True':
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        
        for param in model.model.decoder.parameters():
            param.requires_grad = False

    if params['freeze_bbox_predictor'] == 'True':
        for param in model.bbox_predictor.parameters():
            param.requires_grad = False
            
            
    # Replace the classification head
    # Detr use hungarian matching and implementing sodtmax at the end can cause errors.
    if params['extra_layers'] != 'True':
        d_model = model.model.decoder.layers[0].self_attn.embed_dim
        model.class_labels_classifier = nn.Sequential(
            nn.Linear(d_model, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, num_labels + 1)
        )

    return model

def run_model(
        params: dict[str, Union[int, float, str]], 
        model: PreTrainedModel,
        video_path, 
        annotations,
        trial_number: int,
        train_frames_idx: list,
        valid_frames_idx: list,
        test_frames_idx: list,
        num_labels: int = 1
        ) -> None:
    
    seed = 49
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: --> {device}")
    
    model_name = f"best model with parameters_lr_{params['lr']}_op_{params['optimizer']}_ded_{params['detr_dim']}_fold_{params['k_fold']}.pth"
    print(model_name)
    num_epochs = params['epochs']

    model = freeze_layers(
        params,
        model, 
        num_labels, 
        hidden_layer_dim=params['detr_dim'])
    
    model.to(device)
    optimizer = get_optimizer(params, model)


    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((params['img_size'], params['img_size'])),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainer = Trainer(
        video_path,
        train_frames_idx=train_frames_idx,
        valid_frames_idx=valid_frames_idx,
        test_frames_idx=test_frames_idx,
        annotation_file=annotations,
        transform=transform,
        car_category_id=1
    )

    # Early stopping
    patience = 20
    min_delta = 0.001
    best_val_loss = np.inf
    current_patience = 0

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    for epoch in range(num_epochs):
        train_loss, train_accuracy = trainer.train(model, optimizer, device)
        val_loss, val_accuracy = trainer.validation(model, device)
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            current_patience = 0

            # Save the best model
            print("Best model. Saving weights")
            model_path = Path("best_models") / model_name
            torch.save(model.state_dict(), model_path)
        else:
            current_patience += 1
            if current_patience > patience:
                print("Early stopping.")
                break

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss/Accuracy: {train_loss:.4f}/{train_accuracy:.4f} Val Loss/Accuracy: {val_loss:.4f}/{val_accuracy:.4f}')

        #Add info to wandb
        wandb.log({
            'Train Loss': train_loss,
            'Validation Loss': val_loss,
            'Train Accuracy': train_accuracy,
            'Validation Accuracy': val_accuracy,
        })

    model_path = Path("best_models") / model_name
    print(f'Testing the best model with name: {model_name} ...')
    model.load_state_dict(torch.load(model_path))  # Load best saved model

    test_loss, all_pred_boxes = trainer.test(model, device)
    output_filename = f"predicted_boxes_lr_{params['lr']}_op_{params['optimizer']}_ded_{params['detr_dim']}_fold_{params['k_fold']}.txt"
    with open(output_filename, "w") as f:
        for frame_idx, boxes in enumerate(all_pred_boxes):
            f.write(f"Frame {frame_idx}:\n")
            if boxes:  # If there are predictions for this frame
                for box in boxes:
                    # Format each box as a comma-separated string
                    formatted_box = ", ".join(map(str, box))
                    f.write(f"  {formatted_box}\n")
            else:
                f.write("  No detections\n")
            f.write("\n")

    print(f"Predicted boxes saved to {output_filename}")
    print(f"Final Test Loss: {test_loss:.4f}")

    # Log test performance
    wandb.log({
        'Test Loss': test_loss
    })
