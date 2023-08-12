import argparse
import os
import time

import monai
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.optim import lr_scheduler
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

import config
from dataset import BrainRSNADataset

import wandb
wandb.login(key='a2a7828ed68b3cba08f2703971162138c680b664')

'''
parser = argparse.ArgumentParser()
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--type", default="FLAIR", type=str)
#parser.add_argument("--model_name", default="b0", type=str)
args = parser.parse_args()'''

'''
data = pd.read_csv("../input/train.csv")
train_df = data[data.fold != args.fold].reset_index(drop=False)
val_df = data[data.fold == args.fold].reset_index(drop=False)
'''

device = torch.device("cuda")
#fscore = F1Score(task='binary')
mod = ['FLAIR', 'T1w', 'T1wCE', 'T2w']

for m in mod:
    wandb.init(
    project="try runs 1", 
    name=f"experiment_{m}", 
    config={
    "learning_rate": 0.0001,
    "architecture": "ResNet10",
    "dataset": "MICAA MRI",
    "epochs": config.N_EPOCHS,
    "Batch size": config.TRAINING_BATCH_SIZE
    })


    dlt = []
    empty_fld = [109, 123, 709]
    df = pd.read_csv("./input/train_labels.csv")
    skf = StratifiedKFold(n_splits=10)
    X = df['BraTS21ID'].values
    Y = df['MGMT_value'].values

    for i in empty_fld:
        j = np.where(X == i)
        dlt.append(j)
        X = np.delete(X, j)
        
    Y = np.delete(Y,dlt)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(Y)), Y), 1):  
    

        losses = []
        train_f_score = []
        test_fscore = []
        start_time = time.time
        
        
        xtrain = X[train_idx]
        ytrain = Y[train_idx]
        xtest = X[test_idx]
        ytest = Y[test_idx]
            
        print(f"train_{m}_{fold}")
        
        train_dataset = BrainRSNADataset(
                                        patient_path='./input/reduced_dataset/',
                                        paths=xtrain, 
                                        targets= ytrain,
                                        mri_type=m,
                                        ds_type=f"train_{m}_{fold}"
                                        )

        valid_dataset = BrainRSNADataset(
                                        patient_path='./input/reduced_dataset/',
                                        paths=xtest,
                                        targets=ytest,
                                        mri_type=m,
                                        is_train=False,
                                        ds_type=f"val_{m}_{fold}"
                                        )


        train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAINING_BATCH_SIZE,
            shuffle=True,
            num_workers=config.n_workers,
            drop_last=False,
            pin_memory=True,
        )

        validation_dl = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.TEST_BATCH_SIZE,
            shuffle=False,
            num_workers=config.n_workers,
            pin_memory=True,
        )


        model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5, last_epoch=-1, verbose=True)

        model.zero_grad()
        model.to(device)
        best_loss = 9999
        best_auc = 0
        criterion = nn.BCEWithLogitsLoss()
        for counter in range(config.N_EPOCHS):

            epoch_iterator_train = tqdm(train_dl)
            tr_loss = 0.0
            preds = []
            true_labels = []
            #case_ids = []
            for step, batch in enumerate(epoch_iterator_train):
                model.train()
                images, targets = batch["image"].to(device), batch["target"].to(device)

                outputs = model(images)
                targets = targets  # .view(-1, 1)
                loss = criterion(outputs.squeeze(1), targets.float())

                loss.backward()
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()

                tr_loss += loss.item()
                epoch_iterator_train.set_postfix(
                    batch_loss=(loss.item()), loss=(tr_loss / (step + 1))
                )

                preds.append(outputs.sigmoid().detach().cpu().numpy())
                true_labels.append(targets.cpu().numpy())
                #case_ids.append(batch["case_id"])
            preds = np.vstack(preds).T[0].tolist()
            true_labels = np.hstack(true_labels).tolist()
            #case_ids = np.hstack(case_ids).tolist()
            auc_score = roc_auc_score(true_labels, preds)
            #f1
            #f1_score = fscore(preds, true_labels)
            
            wandb.log({
                'train loss': tr_loss / (step+1),
                'Train AUC': auc_score, 
                'Train F1 score': f1_score
            })

            auc_score_adj_best = 0
            for thresh in np.linspace(0, 1, 50):
                auc_score_adj = roc_auc_score(true_labels, list(np.array(preds) > thresh))
                if auc_score_adj > auc_score_adj_best:
                    best_thresh = thresh
                    auc_score_adj_best = auc_score_adj

            print(
                f"EPOCH {counter}/{config.N_EPOCHS}: Train average loss: {tr_loss/(step+1)} + F1 Score = {f1_score} AUC SCORE = {auc_score} + AUC SCORE THRESH {best_thresh} = {auc_score_adj_best}"
            )

            if auc_score > best_auc:
                print("Saving the model...")

                all_files = os.listdir("../weights/checkpoints/")

                for f in all_files:
                    if f"resnet10_{mod}_fold{fold}" in f:
                        os.remove(f"../weights/checkpoints/{f}")

                best_auc = auc_score
                torch.save(
                    model.state_dict(),
                    f"../weights/chechpoints/resnet10_{mod}_fold{fold}.pth",
                )

            scheduler.step()  # Update learning rate schedule

            if config.do_valid:
                model.load_state_dict(torch.load(f"../weights/checkpoints/resnet10_{m}_fold{fold}"))           
                with torch.no_grad():
                    val_loss = 0.0
                    preds = []
                    true_labels = []
                    #case_ids = []
                    epoch_iterator_val = tqdm(validation_dl)
                    for step, batch in enumerate(epoch_iterator_val):
                        model.eval()
                        images, targets = batch["image"].to(device), batch["target"].to(device)

                        outputs = model(images)
                        targets = targets  # .view(-1, 1)
                        loss = criterion(outputs.squeeze(1), targets.float())
                        val_loss += loss.item()
                        epoch_iterator_val.set_postfix(
                            batch_loss=(loss.item()), loss=(val_loss / (step + 1))
                        )
                        preds.append(outputs.sigmoid().detach().cpu().numpy())
                        true_labels.append(targets.cpu().numpy())
                        case_ids.append(batch["case_id"])
                preds = np.vstack(preds).T[0].tolist()
                true_labels = np.hstack(true_labels).tolist()
                case_ids = np.hstack(case_ids).tolist()
                auc_score = roc_auc_score(true_labels, preds)
                #f1_score = fscore(preds, true_labels)
                
            wandb.log({'Test AUC': auc_score,
                        'Test F1 score': f1_score})
            
            auc_score_adj_best = 0
            for thresh in np.linspace(0, 1, 50):
                auc_score_adj = roc_auc_score(true_labels, list(np.array(preds) > thresh))
                if auc_score_adj > auc_score_adj_best:
                    best_thresh = thresh
                    auc_score_adj_best = auc_score_adj

            print(
                f"EPOCH {counter}/{config.N_EPOCHS}: Validation average loss: {val_loss/(step+1)} + F1 Score = {f1_score} AUC SCORE = {auc_score} + AUC SCORE THRESH {best_thresh} = {auc_score_adj_best}"
            )
            '''
            if auc_score > best_auc:
                print("Saving the model...")

                all_files = os.listdir("../weights/checkpoints/")

                for f in all_files:
                    if f"resnet10_{mod}_fold{fold}" in f:
                        os.remove(f"../weights/checkpoints/{f}")

                best_auc = auc_score
                torch.save(
                    model.state_dict(),
                    f"../weights/chechpoints/resnet10_{mod}_fold{fold}.pth",
                )'''

    elapsed_time = time.time() - start_time
    '''wandb.log({
        'Avg Train loss': np.mean(losses),
        'Avg Train F1 Score': np.mean(train_f_score),
        'Avg Test F1 Score': np.mean(test_fscore)
    })'''

    #print(best_auc)
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    #print('Avg loss {:.5f}'.format(np.mean(losses)))
    #print('Avg Train f1_score {:.5f}'.format(np.mean(train_f_score)))
    #print('Avg Test f1_score {:.5f}'.format(np.mean(test_fscore)))