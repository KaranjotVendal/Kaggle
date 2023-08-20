import os
import time
import numpy as np
import pandas as pd

import monai
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.optim import lr_scheduler
from sklearn.model_selection import StratifiedKFold

import config
from dataset import BrainRSNADataset
from eval import evaluate
from utils import update_metrics, save_metrics_to_json
from plotting import plot_train_valid_fold, plot_trian_valid_all_fold, plot_test_metrics

import wandb

wandb.login(key='a2a7828ed68b3cba08f2703971162138c680b664')

#mod = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
mod = ['FLAIR']
for m in mod:
    wandb.init(
    project="Kaggle_LB-1 sanity check", 
    name=f"experiment_{m}", 
    config={
    "learning_rate": 0.0001,
    "architecture": "ResNet10",
    "dataset": "MICAA MRI",
    "epochs": config.N_EPOCHS,
    "Batch size": config.TRAINING_BATCH_SIZE
    })

    start_time = time.time()

    dlt = []
    empty_fld = [109, 123, 709]
    df = pd.read_csv("./input/train_labels.csv")
    skf = StratifiedKFold(n_splits=5)
    X = df['BraTS21ID'].values
    Y = df['MGMT_value'].values

    for i in empty_fld:
        j = np.where(X == i)
        dlt.append(j)
        X = np.delete(X, j)
        
    Y = np.delete(Y,dlt)

    metrics = {}
    fold_acc = []
    fold_auroc = []
    fold_f1 = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(Y)), Y), 1):  

        losses = []
        train_f_score = []
        test_fscore = []

        metrics[fold] = {
        'train': {'loss': [], 'acc': [], 'f1': [], 'auroc': []},
        'valid': {'loss': [], 'acc': [], 'f1': [], 'auroc': []},
        'test': {'acc': [], 'f1': [], 'auroc': []}
        }
        
        xtrain = X[train_idx]
        ytrain = Y[train_idx]
        xtest = X[val_idx]
        ytest = Y[val_idx]
            
        print(f"-----------------train_{m}_{fold}-------------------")
        
        train_dataset = BrainRSNADataset(
                                        patient_path='./input/reduced_dataset',
                                        paths=xtrain, 
                                        targets= ytrain,
                                        mri_type=m,
                                        ds_type=f"train_{m}_{fold}"
                                        )
        print('loaded dataset')

        valid_dataset = BrainRSNADataset(
                                        patient_path='./input/reduced_dataset',
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
        print('loaded dataloader')

        validation_dl = torch.utils.data.DataLoader(
                                        valid_dataset,
                                        batch_size=config.TEST_BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=config.n_workers,
                                        pin_memory=True,
                                    )

        model = monai.networks.nets.resnet10(spatial_dims=3, n_input_channels=1, num_classes=1)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.5, last_epoch=-1, verbose=True)

        model.zero_grad()
        model.to(config.device)
        best_loss = 9999
        best_auc = 0
        criterion = nn.BCEWithLogitsLoss()
        
        hist = {'val_loss':[],
            'val_acc':[],
            'val_f1':[],
            'val_auroc':[],
            'train_loss':[],
            'train_acc':[],
            'train_f1': [],
            'train_auroc': [],
        }
           
        for counter in range(config.N_EPOCHS):
            t_time = time.time()

            tr_loss = 0.0
            preds = []
            true_labels = []
            #case_ids = []
            for step, batch in enumerate(train_dl):
                model.train()
                images, targets = batch["image"].to(config.device), batch["target"].to(config.device)

                outputs = model(images)
                targets = targets  # .view(-1, 1)
                loss = criterion(outputs.squeeze(1), targets.float())

                loss.backward()
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()

                tr_loss += loss.item()
            
                preds.append(outputs.sigmoid().detach().cpu().numpy())
                true_labels.append(targets.cpu().numpy())
                #case_ids.append(batch["case_id"])
            preds = np.vstack(preds).T[0].tolist()
            true_labels = np.hstack(true_labels).tolist()
            #case_ids = np.hstack(case_ids).tolist()
            t_loss = tr_loss / (step +1)
            t_auc_score = roc_auc_score(true_labels, preds)
            t_f1 = f1_score(true_labels, preds)
            t_acc = accuracy_score(true_labels, preds)
            
            wandb.log({
                'train loss': t_loss,
                'train auroc': t_auc_score, 
                'train f1': t_f1,
                'train acc': t_acc
            })

            hist['train_loss'].append(t_loss)
            hist['train_acc'].append(t_acc)
            hist['train_f1'].append(t_f1)
            hist['train_auroc'].append(t_auc_score)

            
            train_time = time.time() - t_time
            print(
                f"Train EPOCH {counter + 1}/{config.N_EPOCHS}: Train average loss: {t_loss:.5f} | F1 = {t_f1:.4f} | AUCROC SCORE = {t_auc_score:.4f} | Time: {train_time}"
            )

            scheduler.step()  # Update learning rate schedule

            if config.do_valid:
                v_time = time.time()
                with torch.no_grad():
                    val_loss = 0.0
                    preds = []
                    true_labels = []
                    #case_ids = []
                    for step, batch in enumerate(validation_dl):
                        model.eval()
                        images, targets = batch["image"].to(config.device), batch["target"].to(config.device)
    
                        outputs = model(images)
                        targets = targets  # .view(-1, 1)
                        loss = criterion(outputs.squeeze(1), targets.float())
                        val_loss += loss.item()
                    
                        preds.append(outputs.sigmoid().detach().cpu().numpy())
                        true_labels.append(targets.cpu().numpy())
                        #case_ids.append(batch["case_id"])
                preds = np.vstack(preds).T[0].tolist()
                true_labels = np.hstack(true_labels).tolist()
                #case_ids = np.hstack(case_ids).tolist()
                
                v_loss = val_loss / (step + 1)
                v_auc_score = roc_auc_score(true_labels, preds)
                v_f1 = f1_score(true_labels, preds)
                v_acc = accuracy_score(true_labels, preds)
                
            wandb.log({'valid auroc': v_auc_score,
                        'valid f1': v_f1,
                        'valid acc': v_acc,
                        'valid loss': val_loss
                        })
            
            hist['val_loss'].append(v_loss)
            hist['val_acc'].append(v_acc)
            hist['val_f1'].append(v_f1)
            hist['val_auroc'].append(v_auc_score)

            auc_score_adj_best = 0
            for thresh in np.linspace(0, 1, 50):
                auc_score_adj = roc_auc_score(true_labels, list(np.array(preds) > thresh))
                if auc_score_adj > auc_score_adj_best:
                    best_thresh = thresh
                    auc_score_adj_best = auc_score_adj

            valid_time = time.time() - v_time
            print(
                f" Val EPOCH {counter + 1}/{config.N_EPOCHS}: Validation average loss: {v_loss} + F1 Score = {v_f1} AUC SCORE = {v_auc_score} + AUC SCORE THRESH {best_thresh} = {auc_score_adj_best} | Time: {valid_time}"
            )
            
            if v_auc_score > best_auc:
                print("Saving the model...")

                all_files = os.listdir("../weights/checkpoints/")

                for f in all_files:
                    if f"resnet10_{m}_fold{fold}" in f:
                        os.remove(f"../weights/checkpoints/{f}")

                best_auc = v_auc_score
                torch.save(
                    model.state_dict(),
                    f"../weights/checkpoints/resnet10_{m}_{fold}.pth",
                )
        #evaluation

        test_acc, test_f1, test_auroc = evaluate(validation_dl, fold, m)

        for value in hist['train_loss']:
            update_metrics(metrics, fold, 'train', 'loss', value)
    
        for value in hist['train_acc']:
            update_metrics(metrics, fold, 'train', 'acc', value)
    
        for value in hist['train_f1']:
            update_metrics(metrics, fold, 'train', 'f1', value)
    
        for value in hist['train_auroc']:
            update_metrics(metrics, fold, 'train', 'auroc', value)
    
        for value in hist['val_loss']:
            update_metrics(metrics, fold, 'valid', 'loss', value)
    
        for value in hist['val_acc']:
            update_metrics(metrics, fold, 'valid', 'acc', value)
    
        for value in hist['val_f1']:
            update_metrics(metrics, fold, 'valid', 'f1', value)
    
        for value in hist['val_auroc']:
            update_metrics(metrics, fold, 'valid', 'auroc', value)

        update_metrics(metrics, fold, 'test', 'acc', test_acc)
        update_metrics(metrics, fold, 'test', 'f1', test_f1)
        update_metrics(metrics, fold, 'test', 'auroc', test_auroc)
        
        fold_acc.append(test_acc)
        fold_f1.append(test_f1)
        fold_auroc.append(test_auroc)

        print('\nTraining complete for fold: {fold} {:.0f}m {:.0f}s'.format(fold, elapsed_time // 60, elapsed_time % 60))

    json_path = save_metrics_to_json(metrics, 'resnet10')
    
    #plotting loss
    plot_train_valid_fold(json_path, 'loss')
    plot_trian_valid_all_fold(json_path, 'loss')
    
    #plotting acc
    plot_train_valid_fold(json_path, 'acc')
    plot_trian_valid_all_fold(json_path, 'acc')
    plot_test_metrics(json_path, 'acc')


    #plotting f1
    plot_train_valid_fold(json_path, 'f1')
    plot_trian_valid_all_fold(json_path, 'f1')
    plot_test_metrics(json_path, 'f1')

    #plotting auroc
    plot_train_valid_fold(json_path, 'auroc')
    plot_trian_valid_all_fold(json_path, 'auroc')
    plot_test_metrics(json_path, 'auroc')



    elapsed_time = time.time() - start_time
    
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))

    print('\nfold accuracy:', fold_acc)
    print('\nfold f1_score:',fold_f1)
    print('\nfold auroc:', fold_auroc)
    print('\nStd F1 score:', np.std(np.array(fold_f1)))
    print('\nAVG performance of model:', np.mean(np.array(fold_f1)))

    wandb.log({
        'Avg performance': np.mean(np.array(fold_f1)),
        'Std f1 score': np.std(np.array(fold_f1))
    })
    
    wandb.finish()