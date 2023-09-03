import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import config

def plot_metrics_from_csv(csv_path, metrics_to_plot=['loss', 'f1', 'auroc'], datasets_to_plot=['train', 'valid', 'test'], folds_to_plot=None):
    df = pd.read_csv(csv_path)
    
    if folds_to_plot is None:
        folds_to_plot = df['fold'].unique()
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 7))
        
        for fold in folds_to_plot:
            for dataset in datasets_to_plot:
                data = df[(df['fold'] == fold) & (df['dataset'] == dataset)]
                plt.plot(data['epoch'], data[metric], label=f"Fold {fold} ({dataset})")
        
        plt.title(f"{metric.capitalize()} across epochs")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()



def plot_metrics_from_dict(path, metrics_dict, metrics_to_plot=['loss', 'f1', 'auroc'], datasets_to_plot=['train', 'valid', 'test'], folds_to_plot=None):
    """
    Plots specified metrics from a nested dictionary.
    
    Parameters:
    - metrics_dict: Nested dictionary containing metric values.
    - metrics_to_plot: List of metric names to be plotted. Defaults to ['loss', 'f1', 'auroc'].
    - datasets_to_plot: List of dataset types (e.g., 'train', 'valid', 'test') to be plotted. Defaults to ['train', 'valid', 'test'].
    - folds_to_plot: List of fold numbers to be plotted. If None, plots all folds.
    """
    
    # If folds are not specified, get all available folds
    if folds_to_plot is None:
        folds_to_plot = metrics_dict.keys()
    
    # Plot each specified metric
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 7))
        
        for fold in folds_to_plot:
            for dataset in datasets_to_plot:
                # Ensure the dataset and metric exist for the fold
                if dataset in metrics_dict[fold] and metric in metrics_dict[fold][dataset]:
                    epochs = list(range(1, len(metrics_dict[fold][dataset][metric]) + 1))
                    plt.plot(epochs, metrics_dict[fold][dataset][metric], label=f"Fold {fold} ({dataset})")
        
        #plt.title(f"{metric.capitalize()} across epochs")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()




def plot_train_valid_fold(json_path, metrics):
    with open(json_path, "r") as file:
        metrics_json = json.load(file)    
    
    base_dir = './plots'
    model_dir = os.path.join(base_dir, 'resnet10')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    fld_dir = os.path.join(model_dir, 'folds_plots')
    if not os.path.exists(fld_dir):
        os.mkdir(fld_dir)
    
    for fold, data in metrics_json.items():
        plt.figure(figsize=(10, 5))
        
        # Plotting training AUROC
        plt.plot(data['train'][metrics], label='Train {metrics}', marker = 'o')
        
        # Plotting validation AUROC
        plt.plot(data['valid'][metrics], label='Valid {metrics}', marker = 'o')
        
        plt.title(f'Fold {fold} - Train & Valid {metrics}')
        plt.xlabel('Epochs')
        plt.ylabel(metrics)
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(fld_dir, f"train_valid_{metrics}_fold_{fold}.png")
        plt.savefig(save_path)
        plt.close()



def plot_train_valid_all_fold(json_path, metrics):
    with open(json_path, "r") as file:
        metrics_json = json.load(file)    

    base_dir = './plots'
    model_dir = os.path.join(base_dir, 'resnet10')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    all_fld_dir = os.path.join(model_dir, 'all_folds')
    if not os.path.exists(all_fld_dir):
        os.mkdir(all_fld_dir)

    plt.figure(figsize=(12, 6))
    for fold, data in metrics_json.items():
        plt.plot(data['train'][metrics], label=f'Train {metrics} Fold {fold}', marker = 'o')
        plt.plot(data['valid'][metrics], linestyle='dashed', label=f'Valid {metrics} Fold {fold}', marker = 'o')

    plt.title(f'Train & Valid {metrics} Across All Folds')
    plt.xlabel('Epochs')
    plt.ylabel(metrics)
    plt.legend()
    plt.grid(True)

    save_dir = os.path.join(all_fld_dir, f"train_valid_{metrics}.png")
    plt.savefig(save_dir)
    plt.close()


def plot_test_metrics(json_filepath, metric_name):
    dataset_type = 'test'
    # Load the metrics from the JSON file
    with open(json_filepath, "r") as file:
        metrics_dict = json.load(file)
    
    # Extract the specified metric for each fold
    folds = sorted(list(metrics_dict.keys()), key=int)  # Ensure the folds are in numerical order
    metric_values = [metrics_dict[fold][dataset_type][metric_name][0] for fold in folds]
    
    # Plot the metric values
    plt.figure(figsize=(10, 6))
    plt.plot(folds, metric_values, marker='o', linestyle='-')
    plt.xlabel("Folds")
    plt.ylabel(f"{metric_name.capitalize()} Score")
    plt.title(f"{metric_name.capitalize()} Score w.r.t Folds")
    plt.xticks(folds)  # This ensures each fold is shown on the x-axis
    plt.grid(True)#, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'./plots/resnet10/test_{metric_name}_plot.png')
    plt.close()