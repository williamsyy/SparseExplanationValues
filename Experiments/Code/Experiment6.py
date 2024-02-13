# Experiments 6: SEV_R Optimizations
# Description: This file contains the code for optimzing the L2 LR for lower SEV_R values in different datsets.

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
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_name', type=str, default="compas")

args = parser.parse_args()
data_name = args.data_name

param_dict = {"compas":["sex=female","age"],"adult":["age","marital-status","relationship","race","sex","native-country","occupation"],"mimic":["age","preiculos"],"german":["Age","Personal-status-sex","Job"]}

# load the dataset
X,y,X_neg = data_loader(data_name)

if data_name == "adult":
    X.columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"]
    X_neg.columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"]

if data_name == "german":
    X.columns = ["Checking-account-status","Duration-in-month","Credit-history","Purpose","Credit-amount","Saving-Account","Present-employment-since","Installment-rate","Personal-status-sex","Other-debetor","Present-resident-since","Property","Age","Other-installment","Housing","Number-of-credit","Job","Liable-People","Telephone","Foreign-worker"]
    X_neg.columns = ["Checking-account-status","Duration-in-month","Credit-history","Purpose","Credit-amount","Saving-Account","Present-employment-since","Installment-rate","Personal-status-sex","Other-debetor","Present-resident-since","Property","Age","Other-installment","Housing","Number-of-credit","Job","Liable-People","Telephone","Foreign-worker"]

acc_original_lst = []
acc_lst = []
auc_original_lst = []
auc_lst = []
mean_sev_r_original_lst = []
mean_sev_r_lst = []
unreachable_original_lst = []
unreachable_lst = []

for _ in range(10):
    # list the restricted features
    restricted_features = param_dict[data_name]

    sev_lst_original, test_acc_lst_original, test_auc_lst_original, train_acc_lst_original, train_auc_lst_original = [], [], [], [], []
    sev_lst, test_acc_lst, test_auc_lst, train_acc_lst, train_auc_lst = [], [], [], [], []

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.8)
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
    model_lr = LogisticRegression(penalty="l2",solver='liblinear',C=1e-2)
    model_lr.fit(encoded_data_train_arr,encoded_y_train)

    # load the optimized model
    model = SimpleLR(encoded_data_train.shape[1],sev.data_map,sev.overall_mean,1,0)

    # load the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

    # load the criteria
    sev = SEV(None, encoder, encoded_data_train.columns,strict=restricted_features)
    criteria = AllOptRestricted(model,sev.choices)
    original_criteria = OriginalLoss()

    # train the model
    model_train(model,original_criteria,criteria,optimizer,train_loader,num_epochs=100,warm_up=0.8)

    explained_features = []
    explained_features_original = []
    unreachable_original = 0
    unreachable = 0

    for i in range(encoded_data_test.shape[0]):
        Xi = pd.DataFrame(encoded_data_test.iloc[i]).T
        if model_lr.predict(Xi) == 1:
            sev = SEV(model_lr, encoder, encoded_data_test.columns,strict=restricted_features)
            sev_num = sev.sev_cal(np.array(Xi), mode="minus")
            if sev_num is None:
                unreachable_original += 1
                sev_lst_original.append(X_train.shape[1])
            else:
                sev_lst_original.append(sev_num)
                old_features = sev.sev_count(np.array(Xi),mode="minus",choice=sev_num)
                explained_features_original += list(old_features)
        if model.predict(Xi) == 1:
            sev = SEV(model, encoder, encoded_data_test.columns,strict=restricted_features)
            sev_num = sev.sev_cal(np.array(Xi), mode="minus")
            if sev_num is None:
                unreachable += 1
                sev_lst_original.append(X_train.shape[1])
            else:
                sev_lst.append(sev_num)
                new_features = sev.sev_count(np.array(Xi),mode="minus",choice=sev_num)
                explained_features += list(new_features)

    y_pred_proba = model.predict_proba(encoded_data_test)
    y_pred_proba_original = model_lr.predict_proba(encoded_data_test)

    print("auc_original:",roc_auc_score(encoded_y_test,y_pred_proba_original[:,1]))
    print("auc:",roc_auc_score(encoded_y_test,y_pred_proba[:,1]))
    auc_original_lst.append(roc_auc_score(encoded_y_test,y_pred_proba_original[:,1]))
    auc_lst.append(roc_auc_score(encoded_y_test,y_pred_proba[:,1]))
    print("acc_original:",accuracy_score(encoded_y_test,model_lr.predict(encoded_data_test)))
    print("acc:",accuracy_score(encoded_y_test,model.predict(encoded_data_test)))
    acc_original_lst.append(accuracy_score(encoded_y_test,model_lr.predict(encoded_data_test)))
    acc_lst.append(accuracy_score(encoded_y_test,model.predict(encoded_data_test)))


    explained_features = pd.Series(explained_features).value_counts()/np.sum(model.predict(encoded_data_test))
    print(dict(explained_features))
    explained_features_original = pd.Series(explained_features_original).value_counts()/np.sum(model_lr.predict(encoded_data_test))
    print(dict(explained_features_original))
    sev_df = pd.Series(sev_lst).value_counts()
    sev_df_original = pd.Series(sev_lst_original).value_counts()

    print(dict(sev_df_original),np.mean(sev_lst_original))
    mean_sev_r_original_lst.append(np.mean(sev_lst_original))
    print(dict(sev_df),np.mean(sev_lst))
    mean_sev_r_lst.append(np.mean(sev_lst))
    print(unreachable_original/np.sum(model_lr.predict(encoded_data_test)))
    unreachable_original_lst.append(unreachable_original/np.sum(model_lr.predict(encoded_data_test)))
    print(unreachable/np.sum(model.predict(encoded_data_test)))
    unreachable_lst.append(unreachable/np.sum(model.predict(encoded_data_test)))

print("Original:")
print("acc:",np.mean(acc_original_lst),np.std(acc_original_lst))
print("auc:",np.mean(auc_original_lst),np.std(auc_original_lst))
print("mean_sev_r:",np.mean(mean_sev_r_original_lst),np.std(mean_sev_r_original_lst))
print("unreachable:",np.mean(unreachable_original_lst),np.std(unreachable_original_lst))
print("SEV:")
print("acc:",np.mean(acc_lst),np.std(acc_lst))
print("auc:",np.mean(auc_lst),np.std(auc_lst))
print("mean_sev_r:",np.mean(mean_sev_r_lst),np.std(mean_sev_r_lst))
print("unreachable:",np.mean(unreachable_lst),np.std(unreachable_lst))




