import os
import time
import numpy as np

import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import monai

import config
import wandb
    
    
def evaluate(validation_dl, fold, mod):
    test_time = time.time()
    model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)   
    checkpoint = torch.load(f'./weights/checkpoints/resnet10_{mod}_{fold}.pth')
    model.load_state_dict(checkpoint)
    model.to(config.device)
    model.eval()
    
    preds = []
    true_labels = []            
    for e, batch in enumerate(validation_dl):
        with torch.no_grad():
            model.eval()
            images, targets = batch["image"].to(config.device), batch["target"].to(config.device)

            outputs = model(images)
            targets = targets  # .view(-1, 1)
        
            preds.append(outputs.sigmoid().detach().cpu().numpy())
            true_labels.append(targets.cpu().numpy())
    preds = np.vstack(preds).T[0].tolist()
    true_labels = np.hstack(true_labels).tolist()
    preds_labels = [1 if p > 0.5 else 0 for p in preds]
        
    auc_score = roc_auc_score(true_labels, preds)
    f1 = f1_score(true_labels, preds_labels)
    acc = accuracy_score(true_labels, preds_labels)

    auc_score_adj_best = 0
    for thresh in np.linspace(0, 1, 50):
        auc_score_adj = roc_auc_score(true_labels, list(np.array(preds) > thresh))
        if auc_score_adj > auc_score_adj_best:
            best_thresh = thresh
            auc_score_adj_best = auc_score_adj
    
    if config.WANDB:
        wandb.log({
                'test acc': acc,
                'test f1': f1,
                'test AUROC': auc_score
                })    

        #gradcam
        #gradcam(test_loader, fold)
    
    elapsed_time = time.time() - test_time
    
    print(f'\nEvaluatin complete for {mod} of fold:{fold} in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s | Accuracy: {acc:.3f}% | F1 Score: {f1:.4f} | AUROC: {auc_score:.4f}')

    return acc, f1, auc_score