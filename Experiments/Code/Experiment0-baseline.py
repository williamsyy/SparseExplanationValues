# Experiments 0: Baseline
# Description: This file contains the code for the baseline experiments without error bars for sankey plotting

import sys
# Insert the path to the parent directory
sys.path.append('../../')
# load the required models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
# load the SEV Calculator
from SEV.SEV import SEVPlot
# load the dataloader
from SEV.data_loader import data_loader
# import arguments
import argparse
# load the required packages
import numpy as np
import pandas as pd
# load the special data encoder
from SEV.Encoder import DataEncoder
# load the training and testing split
from sklearn.model_selection import train_test_split
# import copy
from copy import copy
# import time
import time

# parse the arguments
parser = argparse.ArgumentParser(description='Baseline Experiments')
parser.add_argument('--dataset', type=str, default='adult',choices=['adult', 'compas', 'german', 'mimic','diabetes','fico'])
parser.add_argument('--model', type=str, default='l2lr',choices=['l2lr','l1lr', 'mlp', 'gbdt'])
parser.add_argument('--SEV_mode', type=str, default='plus', choices=['plus', 'minus', 'restricted'])
parser.add_argument('--max_depth', type=int, default=6)
parser.add_argument('--max_time', type=int, default=14400)
parser.add_argument('--repeat', type=int, default=1)

args = parser.parse_args()

# load the dataset
X,y,X_neg = data_loader(args.dataset)

# preprocessing the dataset
encoder = DataEncoder(standard=True)
encoder.fit(X_neg)
encoded_X = encoder.transform(X)

def model_loader(data_name, model_name):
    # load the model
    if model_name == 'l2lr':
        if data_name == "german":
            model = LogisticRegression(penalty='l2',solver='liblinear',C=1)
        else:
            model = LogisticRegression(penalty='l2',solver='liblinear',C=1e-2)
    elif model_name == "l1lr":
        if data_name == "german":
            model = LogisticRegression(penalty='l1',solver='liblinear',C=1)
        else:
            model = LogisticRegression(penalty='l1',solver='liblinear',C=1e-2)
    elif model_name == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(128,),early_stopping=True,random_state=42)
    elif model_name == "gbdt":
        model = GradientBoostingClassifier(max_depth=3,n_estimators=200,random_state=42)
    else:
        raise ValueError('Invalid Model Name!')
    return model

sev_lst = []
train_acc_lst, train_auc_lst = [], []
test_acc_lst, test_auc_lst = [], []
time_lst = []

for i in range(args.repeat):
    # start the timer
    start_time = time.time()

    # split the train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.2,random_state=42)
    model = model_loader(args.dataset,args.model)
    # fit the model
    model.fit(X_train,y_train)

    # predict the probability of the train dataset
    y_pred_train = model.predict_proba(X_train)[:,1]
    auc_score_train = roc_auc_score(y_train,y_pred_train)
    acc_score_train = accuracy_score(y_train,y_pred_train>0.5)
    # predict the probability of the test dataset
    y_pred_test = model.predict_proba(X_test)[:,1]
    auc_score_test = roc_auc_score(y_test,y_pred_test)
    acc_score_test = accuracy_score(y_test,y_pred_test>0.5)
    
    # print the results
    print('The train accuracy score is %.4f.'%acc_score_train)
    print('The train auc score is %.4f.'%auc_score_train)
    print("The test accuracy score is %.4f."%acc_score_test)
    print('The test auc score is %.4f.'%auc_score_test)

    # calculate the SEV for test dataset
    sev_arr,positive_count = SEVPlot(model,encoder, X_test, args.SEV_mode, max_depth=6, max_time=args.max_time)

    # save the results
    train_acc_lst.append(acc_score_train)
    train_auc_lst.append(auc_score_train)
    test_acc_lst.append(acc_score_test)
    test_auc_lst.append(auc_score_test)
    sev_lst.append(np.sum(sev_arr)/positive_count)
    time_lst.append(time.time()-start_time)

    # save the temp_result
    temp_result = copy(X_test)
    temp_result['y'] = y_test
    temp_result['sev'] = sev_arr
    temp_result.to_csv("../Results/Exp0_Baseline/data/%s_%s_%s_%d.csv"%(args.dataset,args.model,args.SEV_mode,i),index=False)


print("Train Accuracy, %.4f +- %.4f"%(np.mean(train_acc_lst),np.std(train_acc_lst)))
print("Train AUC, %.4f +- %.4f"%(np.mean(train_auc_lst),np.std(train_auc_lst)))
print("Test Accuracy, %.4f +- %.4f"%(np.mean(test_acc_lst),np.std(test_acc_lst)))
print("Test AUC, %.4f +- %.4f"%(np.mean(test_auc_lst),np.std(test_auc_lst)))
print("SEV, %.4f +- %.4f"%(np.mean(sev_lst),np.std(sev_lst)))
print("Time, %.4f +- %.4f"%(np.mean(time_lst),np.std(time_lst)))

# save the results
result_file = open('../Results/Exp0_Baseline/result.csv','a')
# write the results
result_file.write("".join([args.dataset,',',args.model,',',args.SEV_mode,',',str(args.max_depth),',',str(args.max_time),',',str(args.repeat),',',str(np.round(np.mean(train_acc_lst),4)),',',str(np.round(np.std(train_acc_lst),4)),',',str(np.round(np.mean(train_auc_lst),4)),',',str(np.round(np.std(train_auc_lst),4)),',',str(np.round(np.mean(test_acc_lst),4)),',',str(np.round(np.std(test_acc_lst),4)),',',str(np.round(np.mean(test_auc_lst),4)),',',str(np.round(np.std(test_auc_lst),4)),',',str(np.round(np.mean(sev_lst),4)),',',str(np.round(np.std(sev_lst),4)),',',str(np.round(np.mean(time_lst),4)),',',str(np.round(np.std(time_lst),4)),'\n']))
result_file.close()


