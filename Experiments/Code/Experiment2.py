# Experiments 2: Restricted SEV
# Description: This file contains the code for restricted SEV experiments for baseline models and optimized models with error bars

import sys
# Insert the path to the parent directory
sys.path.append('../../')
# Work on restricted SEV
import numpy as np
import pandas as pd
from SEV.data_loader import data_loader
from SEV.Encoder import DataEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from SEV.OptimizedSEV import SimpleLR,CustomDataset,AllOptRestricted,OriginalLoss
from SEV.SEV import SEVPlot,SEV, SEVCount
from torch.utils.data import DataLoader
from SEV.trainer import model_train
from sklearn.metrics import roc_auc_score, accuracy_score
import torch


# load the dataset
X,y,X_neg = data_loader("compas")
# list the restricted features
restricted_features = ["sex=female","age"]

sev_lst_original, test_acc_lst_original, test_auc_lst_original, train_acc_lst_original, train_auc_lst_original = [], [], [], [], []
sev_lst, test_acc_lst, test_auc_lst, train_acc_lst, train_auc_lst = [], [], [], [], []
unrearchable_lst_original = []
unrearchable_lst = []

for i in range(1):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y, train_size=0.8)
    encoded_y_train = np.array(Y_train)
    encoded_y_test = np.array(Y_test)
    encoder = DataEncoder(standard=True)
    merged_data = encoder.fit(X_neg)
    encoded_data_train = encoder.transform(X_train)
    encoded_data_test = encoder.transform(X_test)
    encoded_data_train_arr = np.array(encoded_data_train)
    encoded_data_test_arr = np.array(encoded_data_test)

    sev = SEV(None, encoder, encoded_data_train.columns)
    train_data = CustomDataset(encoded_data_train_arr,encoded_y_train)
    test_data = CustomDataset(encoded_data_test_arr,encoded_y_test)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=False)

    # load the model
    model = LogisticRegression(penalty="l2",solver='liblinear',C=1e-2)
    model.fit(encoded_data_train_arr,encoded_y_train)

    y_pred_train = model.predict_proba(encoded_data_train_arr)[:,1]
    y_pred_test = model.predict_proba(encoded_data_test_arr)[:,1]
    print("Train Accuracy: ",accuracy_score(encoded_y_train,y_pred_train>0.5))
    print("Train AUC: ",roc_auc_score(encoded_y_train,y_pred_train))
    print("Test Accuracy: ",accuracy_score(encoded_y_test,y_pred_test>0.5))
    print("Test AUC: ",roc_auc_score(encoded_y_test,y_pred_test))

    train_acc_lst_original.append(accuracy_score(encoded_y_train,y_pred_train>0.5))
    train_auc_lst_original.append(roc_auc_score(encoded_y_train,y_pred_train))
    test_acc_lst_original.append(accuracy_score(encoded_y_test,y_pred_test>0.5))
    test_auc_lst_original.append(roc_auc_score(encoded_y_test,y_pred_test))

    sev_arr,count = SEVPlot(model,encoder,encoded_data_test,"minus",max_depth=6)
    sev_arr,count = SEVPlot(model,encoder,encoded_data_test,"minus",max_depth=6,strict=restricted_features)
    sev_lst_original.append(np.sum(sev_arr)/count)
    unrearchable_lst_original.append(np.sum(np.array(sev_arr)==7)/count)
    SEVCount(model,encoder,encoded_data_test,"minus",max_depth=6,strict=restricted_features)

    # load the optimized model
    model = SimpleLRmodel = SimpleLR(X.shape[1],sev.data_map,sev.overall_mean,1,0)

    # load the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    # load the criteria
    sev = SEV(None, encoder, encoded_data_train.columns,strict=restricted_features)
    criteria = AllOptRestricted(model,sev.choices)
    original_criteria = OriginalLoss()

    # train the model
    model_train(model,original_criteria,criteria,optimizer,train_loader,num_epochs=100,warm_up=0.8)

    y_pred_train = model.predict_proba(encoded_data_train_arr)[:,1]
    y_pred_test = model.predict_proba(encoded_data_test_arr)[:,1]
    print("Train Accuracy: ",accuracy_score(encoded_y_train,y_pred_train>0.5))
    print("Train AUC: ",roc_auc_score(encoded_y_train,y_pred_train))
    print("Test Accuracy: ",accuracy_score(encoded_y_test,y_pred_test>0.5))
    print("Test AUC: ",roc_auc_score(encoded_y_test,y_pred_test))

    train_acc_lst.append(accuracy_score(encoded_y_train,y_pred_train>0.5))
    train_auc_lst.append(roc_auc_score(encoded_y_train,y_pred_train))
    test_acc_lst.append(accuracy_score(encoded_y_test,y_pred_test>0.5))
    test_auc_lst.append(roc_auc_score(encoded_y_test,y_pred_test))

    # calculate the SEV R
    SEVPlot(model,encoder,encoded_data_test,"minus",max_depth=6)
    sev_arr,count = SEVPlot(model,encoder,encoded_data_test,"minus",max_depth=6,strict=restricted_features)

    sev_lst.append(np.sum(sev_arr)/count)
    unrearchable_lst.append(np.sum(np.array(sev_arr)==7)/count)
    SEVCount(model,encoder,encoded_data_test,"minus",max_depth=6,strict=restricted_features)

# print the performance
print("Train Accuracy, %.4f +- %.4f"%(np.mean(train_acc_lst_original),np.std(train_acc_lst_original)))
print("Train AUC, %.4f +- %.4f"%(np.mean(train_auc_lst_original),np.std(train_auc_lst_original)))
print("Test Accuracy, %.4f +- %.4f"%(np.mean(test_acc_lst_original),np.std(test_acc_lst_original)))
print("Test AUC, %.4f +- %.4f"%(np.mean(test_auc_lst_original),np.std(test_auc_lst_original)))
print("SEV, %.4f +- %.4f"%(np.mean(sev_lst_original),np.std(sev_lst_original)))
print('Unreachable Proportion, %.4f +- %.4f'%(np.mean(unrearchable_lst_original),np.std(unrearchable_lst_original)))

print("-"*30)


print("Train Accuracy, %.4f +- %.4f"%(np.mean(train_acc_lst),np.std(train_acc_lst)))
print("Train AUC, %.4f +- %.4f"%(np.mean(train_auc_lst),np.std(train_auc_lst)))
print("Test Accuracy, %.4f +- %.4f"%(np.mean(test_acc_lst),np.std(test_acc_lst)))
print("Test AUC, %.4f +- %.4f"%(np.mean(test_auc_lst),np.std(test_auc_lst)))
print("SEV, %.4f +- %.4f"%(np.mean(sev_lst),np.std(sev_lst)))
print('Unreachable Proportion, %.4f +- %.4f'%(np.mean(unrearchable_lst),np.std(unrearchable_lst)))


