# Experiments 3: Optimizations for SEV
# Description: This file contains the code for the grid search for different optimization experiments with error bars

import sys
# Insert the path to the parent directory
sys.path.append('../../')
from sklearn.linear_model import LogisticRegression
from SEV.data_loader import data_loader
from SEV.Encoder import DataEncoder
from sklearn.model_selection import train_test_split
from SEV.OptimizedSEV import SimpleLR,CustomDataset,AllOptPlus,OriginalLoss,VolOpt
from SEV.SEV import SEVPlot,SEV, SEVCount
from torch.utils.data import DataLoader
from SEV.trainer import model_train
from SEV.SEV import SEV, SEVPlot
import torch
import numpy as np
import pandas as pd

X,y,X_neg = data_loader("adult")

l1_lst = []
l2_lst = []
volopt_lst = []
all_opt_lst = []
sev_l1_lst, sev_l2_lst, sev_volopt_lst, sev_all_opt_lst = [], [], [], []

for i in range(10):
    # Split the data into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y, train_size=0.8)
    encoder = DataEncoder(standard=True)
    merged_data = encoder.fit(X_neg)
    encoded_y_train = np.array(Y_train)
    encoded_y_test = np.array(Y_test)
    encoded_data_train = encoder.transform(X_train)
    encoded_data_test = encoder.transform(X_test)
    encoded_data_train_arr = np.array(encoded_data_train)
    encoded_data_test_arr = np.array(encoded_data_test)

    # Train the model
    model_l1 = LogisticRegression(penalty="l1",solver='liblinear',C=1e-2)
    model_l2 = LogisticRegression(penalty="l2",solver='liblinear',C=1e-2)
    model_l1.fit(encoded_data_train_arr,encoded_y_train)
    model_l2.fit(encoded_data_train_arr,encoded_y_train)

    # Calculate the SEV
    sev_arr_l1,count = SEVPlot(model_l1,encoder, encoded_data_test,"plus")
    sev_l1_lst.append(np.sum(sev_arr_l1)/count)
    sev_arr_l2,count = SEVPlot(model_l2,encoder, encoded_data_test,"plus")
    sev_l2_lst.append(np.sum(sev_arr_l2)/count)


    sev = SEV(None, encoder, encoded_data_train.columns)
    train_data = CustomDataset(encoded_data_train_arr,encoded_y_train)
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    # Consturct the plots for the optimized models
    model_volopt = SimpleLR(encoded_data_train_arr.shape[1],sev.data_map,sev.overall_mean,0.1,0)
    model_allopt = SimpleLR(encoded_data_train_arr.shape[1],sev.data_map,sev.overall_mean,0.1,0)
    optimizer_volopt = torch.optim.Adam(model_volopt.parameters(), lr=0.1)
    optimizer_allopt = torch.optim.Adam(model_allopt.parameters(), lr=0.1)
    original_criteria = OriginalLoss()
    criteria_volopt = VolOpt(model_volopt)
    criteria_allopt = AllOptPlus(model_allopt)

    model_train(model_volopt,original_criteria,criteria_volopt,optimizer_volopt,train_loader,100,0.7)
    model_train(model_allopt,original_criteria,criteria_allopt,optimizer_allopt,train_loader,100,0.7)

    zero_l1 = np.sum(model_l1.coef_[0] == 0)/len(model_l1.coef_[0])
    zero_l2 = np.sum(model_l2.coef_[0] == 0)/len(model_l2.coef_[0])
    zero_volopt = np.sum(model_volopt.linear.weight.detach().numpy()[0] == 0)/len(model_volopt.linear.weight.detach().numpy()[0])
    zero_allopt = np.sum(model_allopt.linear.weight.detach().numpy()[0] == 0)/len(model_volopt.linear.weight.detach().numpy()[0])

    l1_lst.append(zero_l1)
    l2_lst.append(zero_l2)
    volopt_lst.append(zero_volopt)
    all_opt_lst.append(zero_allopt)

    sev_arr_allopt,count = SEVPlot(model_allopt,encoder, encoded_data_test,"plus")
    sev_all_opt_lst.append(np.sum(sev_arr_allopt)/count)
    sev_arr_volopt,count = SEVPlot(model_volopt,encoder, encoded_data_test,"plus")
    sev_volopt_lst.append(np.sum(sev_arr_volopt)/count)

print("L1: ",np.mean(l1_lst),np.std(l1_lst))
print("L2: ",np.mean(l2_lst),np.std(l2_lst))
print("VolOpt: ",np.mean(volopt_lst),np.std(volopt_lst))
print("AllOpt: ",np.mean(all_opt_lst),np.std(all_opt_lst))
print("SEV L1: ",np.mean(sev_l1_lst),np.std(sev_l1_lst))
print("SEV L2: ",np.mean(sev_l2_lst),np.std(sev_l2_lst))
print("SEV VolOpt: ",np.mean(sev_volopt_lst),np.std(sev_volopt_lst))
print("SEV AllOpt: ",np.mean(sev_all_opt_lst),np.std(sev_all_opt_lst))

