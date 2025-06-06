#!/usr/bin/env python3
"""
File containing the main training script.
"""

#Standard imports
import argparse
import torch
import os
import numpy as np
import random
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
import sys
from torch.utils.data import DataLoader
from tabulate import tabulate
from torchinfo import summary

#Local imports
from util.io import load_json, store_json
from util.eval_spotting import evaluate
from dataset.datasets import get_datasets
from model.model_spotting import Model
import wandb
import optuna

def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)    
    parser.add_argument('--early_stopping', type=bool, default=False, help='Enable early stopping')
    parser.add_argument('--optuna', action="store_true", help="Run hyperparameter optimization with Optuna")
    return parser.parse_args()

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
    args.store_dir = config['save_dir'] + '/' + "splits"
    args.labels_dir = config['labels_dir']
    args.store_mode = config['store_mode']
    args.task = config['task']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.dataset = config['dataset']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.only_test = config['only_test']
    args.device = config['device']
    args.num_workers = config['num_workers']
    args.patience = config['patience']

    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def run_training(args, trial):
    # Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if trial:
        execution_name = f'baseline_finetuning{str(trial.number)}'
    
    else:
        execution_name = f'baseline'

    wandb.init(
        project='Week6_baseline',
        entity='mcv-c6g7',
        name=execution_name,
        config=args, reinit=True)

    # Directory for storing / reading model checkpoints
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, val_data, test_data = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run changing "mode" to "load" in the config JSON.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    def worker_init_fn(id):
        random.seed(id + epoch * 100)

    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )
        
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=(2 if args.num_workers > 0 else None),
        worker_init_fn=worker_init_fn
    )

    # Model
    model = Model(args=args)

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        # Warmup schedule
        num_steps_per_epoch = len(train_loader)
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)
        
        losses = []
        best_criterion = float('inf')
        epoch = 0
        patience = args.patience  
        epochs_no_improve = 0

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):

            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler)
            
            val_loss = model.epoch(val_loader)

            better = False
            if val_loss < best_criterion:
                best_criterion = val_loss
                better = True
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
                epoch, train_loss, val_loss))
            if better:
                print('New best mAP epoch!')

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss
            })

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

                if better:
                    torch.save( model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt') )
    
    model_parameters = model.get_model_parameters()
    wandb.log({"model_params": model_parameters})

    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))

    # Evaluation on test split
    map_score, ap_score = evaluate(model, test_data, nms_window = 5)

    # Report results per-class in table
    table = []
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{ap_score[i]*100:.2f}"])

    table_str = tabulate(table, headers=["Class", "Average Precision"], tablefmt="grid")
    avg_str = tabulate([["Mean", f"{np.mean(map_score)*100:.2f}"]],
                    headers=["", "Average Precision"], tablefmt="grid")

    print(table_str)
    print(avg_str)
    # Combine the strings into one text.
    result_text = table_str + "\n\n" + avg_str

    with open(f"results/results_baseline_{trial.number}_{args.batch_size}_{args.learning_rate}_{args.num_epochs}_{args.warm_up_epochs}_{args.patience}.txt", "w") as f:
        f.write(result_text)
    
    wandb.finish()
    print('CORRECTLY FINISHED TRAINING AND INFERENCE')
    model_summary = summary(model, input_size=(args.batch_size, 50, 3, 224, 398), col_names=("output_size", "num_params", "mult_adds"))
    summary_str = str(model_summary)

    with open(f"summary/model_summary_baseline_{trial.number}_{args.batch_size}_{args.learning_rate}_{args.num_epochs}_{args.warm_up_epochs}_{args.patience}.txt", "w") as f:
        f.write(summary_str)

    return best_criterion

def main(args):
    # Load the configuration JSON based on the provided model name.
    config_path = os.path.join('config', args.model + '.json')
    config = load_json(config_path)
    args = update_args(args, config)

    if args.optuna:
        def objective(trial):
            # For each trial, reload the configuration and update args.
            config_trial = load_json(config_path)
            new_args = update_args(argparse.Namespace(model=args.model, seed=args.seed, optuna=False), config_trial)
            # Override hyperparameters using Optuna suggestions.
            new_args.batch_size = trial.suggest_categorical("batch_size", [4, 6])
            new_args.stride = trial.suggest_categorical("stride", [2])
            # new_args.learning_rate = trial.suggest_categorical("learning_rate", [.0008, 5e-4, 1e-4, 1e-3, 1e-2]) 
            new_args.learning_rate = trial.suggest_categorical("learning_rate", [.0008, .0004, .0001, .001, 8e-5, 4e-5, 1e-5 ]) 
            new_args.num_epochs = trial.suggest_categorical("num_epochs", [15, 20, 25, 30])  # [15, 20, 25, 30]
            # new_args.warm_up_epochs = trial.suggest_categorical("warm_up_epochs", [1, 3, 5])
            new_args.warm_up_epochs = trial.suggest_categorical("warm_up_epochs", [3, 5, 7])

            patience = int(round(new_args.num_epochs / 2))
            if patience == 0 or not patience:
                patience = 10

            new_args.patience = trial.suggest_categorical("patience", [patience])
    
            # Run training for this trial.
            metric = run_training(new_args, trial)
            return metric

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        run_training(args, trial=None)

if __name__ == '__main__':
    main(get_args())