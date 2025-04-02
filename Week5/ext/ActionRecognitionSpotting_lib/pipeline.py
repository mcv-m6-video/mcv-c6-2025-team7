#!/usr/bin/env python3
"""
Main training script for T-DEED with argparse, wandb, and Optuna integration.
"""

import argparse
import os
import sys
import torch
import random
import numpy as np
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
import wandb
import optuna
from tabulate import tabulate

from util.io import load_json, store_json
from util.eval_classification import evaluate
from datasets.datasets import get_datasets
from model.model_classification import Model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help="Model name; used to load config JSON (e.g., config/<model>.json)")
    parser.add_argument('--seed', type=int, default=1, help="Random seed")
    parser.add_argument('--optuna', action="store_true", help="Run hyperparameter optimization with Optuna")
    return parser.parse_args()

def update_args(args, config):
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model  # Append model name.
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
    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({} epochs) + Cosine Annealing LR ({} epochs)'.format(
        args.warm_up_epochs, cosine_epochs))
    return ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer, num_steps_per_epoch * cosine_epochs)
    ])

def run_training(args, trial):
    # Set seeds for reproducibility.
    print('Setting seed to:', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    execution_name = f'xavi_final{str(trial.number)}'

    wandb.init(
        project='Week5_ft_final',
        entity='mcv-c6g7',
        name=execution_name,
        config=args, reinit=True)
    # wandb.config.update({"store_dir": args.store_dir}, allow_val_change=True)
    # wandb.config.update({"save_dir": args.save_dir}, allow_val_change=True)

    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load datasets.
    classes, train_data, val_data, test_data = get_datasets(args)
    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Re-run with store_mode set to "load" in the config JSON.')
        sys.exit('Datasets have been stored. Exiting.')
    else:
        print('Datasets loaded successfully.')

    # Define a worker initialization function for reproducibility.
    def worker_init_fn(worker_id):
        random.seed(worker_id)

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

    # Initialize the model.
    model = Model(args=args)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    best_metric = float('inf')
    losses = []

    if not args.only_test:
        num_steps_per_epoch = len(train_loader)
        lr_scheduler = get_lr_scheduler(args, optimizer, num_steps_per_epoch)

        print('START TRAINING EPOCHS')
        for epoch in range(args.num_epochs):
            # Train one epoch.
            train_loss = model.epoch(train_loader, optimizer, scaler, lr_scheduler=lr_scheduler)
            # Validate.
            val_loss = model.epoch(val_loader)
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print('[Epoch {}] Train loss: {:.5f} | Val loss: {:.5f}'.format(epoch, train_loss, val_loss))
            if val_loss < best_metric:
                best_metric = val_loss
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_best.pt'))
                print('New best metric at epoch {}: {:.5f}'.format(epoch, val_loss))
            losses.append({'epoch': epoch, 'train': train_loss, 'val': val_loss})
            store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)
    else:
        print("Only test mode; skipping training.")

    model_parameters = model.get_model_parameters()
    wandb.log({"model_params": model_parameters})

    print('START INFERENCE')
    model.load(torch.load(os.path.join(ckpt_dir, 'checkpoint_best.pt')))
    ap_score = evaluate(model, test_data)
    table = []
    for i, class_name in enumerate(classes.keys()):
        table.append([class_name, f"{ap_score[i]*100:.2f}"])
    # print(tabulate(table, headers=["Class", "Average Precision"], tablefmt="grid"))
    # print(tabulate([["Average", f"{np.mean(ap_score)*100:.2f}"]],
    #                headers=["", "Average Precision"], tablefmt="grid"))

    # Generate the tabulated strings.
    table_str = tabulate(table, headers=["Class", "Average Precision"], tablefmt="grid")
    avg_str = tabulate([["Average", f"{np.mean(ap_score)*100:.2f}"]],
                    headers=["", "Average Precision"], tablefmt="grid")

    # Print the results to the console.
    print(table_str)
    print(avg_str)

    # Combine the strings into one text.
    result_text = table_str + "\n\n" + avg_str

    # Write the result to a text file.
    with open(f"results/results{trial.number}_{args.batch_size}_{args.learning_rate}_{args.num_epochs}_{args.warm_up_epochs}_{args.pooling_layer}.txt", "w") as f:
        f.write(result_text)

    wandb.finish()
    print('FINISHED TRAINING AND INFERENCE')
    model_summary = summary(model, input_size=(args.batch_size, 50, 3, 224, 398), col_names=("output_size", "num_params", "mult_adds"))
    summary_str = str(model_summary)

    with open(f"summary/model_summary{trial.number}_{args.batch_size}_{args.learning_rate}_{args.num_epochs}_{args.warm_up_epochs}_{args.pooling_layer}.txt", "w") as f:
        f.write(summary_str)

    return best_metric

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
            new_args.batch_size = trial.suggest_categorical("batch_size", [4])
            new_args.stride = trial.suggest_categorical("stride", [2])
            # new_args.learning_rate = trial.suggest_categorical("learning_rate", [.0008, 5e-4, 1e-4, 1e-3, 1e-2]) 
            new_args.learning_rate = trial.suggest_categorical("learning_rate", [.0008]) 
            new_args.num_epochs = trial.suggest_categorical("num_epochs", [5, 10, 15])
            # new_args.warm_up_epochs = trial.suggest_categorical("warm_up_epochs", [1, 3, 5])
            new_args.warm_up_epochs = trial.suggest_categorical("warm_up_epochs", [3])
            new_args.pooling_layer = trial.suggest_categorical("pooling_layer", ["average"])

    
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
        run_training(args)

if __name__ == '__main__':
    args = get_args()
    main(args)