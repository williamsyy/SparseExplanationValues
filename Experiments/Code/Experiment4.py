# Experiments 4: Optimizations
# Description: This file contains the code for calculate the SEV for different optimization experiments with a specific test set for sankey plotting

import sys
# Insert the path to the parent directory
sys.path.append('../../')
import torch
# load the required models
from SEV.OptimizedSEV import SimpleLR, SimpleMLP,SimpleGBDT
# load the required loss functions
from SEV.OptimizedSEV import AllOptPlus,AllOptMinus,VolOpt,OriginalLoss
# load the required dataset
from SEV.OptimizedSEV import CustomDataset
# import the required dataloader
from SEV.data_loader import data_loader
from torch.utils.data import DataLoader
# import trainer
from SEV.trainer import model_train
# load the required packages
import numpy as np
import pandas as pd
# load the special data encoder
from SEV.Encoder import DataEncoder
# load the training and testing split
from sklearn.model_selection import train_test_split
# import argparse
import argparse
# import SEV
from SEV.SEV import SEV,SEVPlot
from sklearn.ensemble import GradientBoostingClassifier
# import evaluation methods
from sklearn.metrics import roc_auc_score, accuracy_score
from copy import copy
torch.manual_seed(42)

# parse the arguments
parser = argparse.ArgumentParser(description='Parameter Search Experiments')
parser.add_argument('--dataset', type=str, default='adult',choices=['adult', 'compas', 'german', 'mimic','diabetes','fico'])
parser.add_argument('--model', type=str, default='l2lr',choices=['lr', 'mlp', 'gbdt'])
parser.add_argument('--Optimized_method', type=str, default='alloptplus', choices=['volopt','alloptplus','alloptminus'])
parser.add_argument('--max_depth', type=int, default=6)
parser.add_argument('--max_time', type=int, default=3600)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=1e-1)
parser.add_argument('--warm_up', type=float, default=0.7)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--sev_penalty', type=float, default=1e-2)
parser.add_argument('--positive_penalty', type=float, default=1e-2)
args = parser.parse_args()

# load the dataset
X,y,X_neg = data_loader(args.dataset)


def model_loader(model_name,sev,sev_penalty,X,y,positive_penalty):
    # load the model
    if model_name == 'lr':
        model = SimpleLR(X.shape[1],sev.data_map,sev.overall_mean,sev_penalty,positive_penalty)
    elif model_name == "mlp":
        model = SimpleMLP(X.shape[1],128,sev.data_map,sev.overall_mean,sev_penalty,positive_penalty)
    elif model_name == "gbdt":
        pre_model = GradientBoostingClassifier(max_depth=3,n_estimators=200)
        pre_model.fit(X,y)
        model = SimpleGBDT(pre_model,sev.data_map,sev.overall_mean,sev_penalty,positive_penalty)
    return model
    
# load the criteria
def criteria_loader(criteria_name,model):
    if criteria_name == 'alloptplus':
        criteria = AllOptPlus(model)
    elif criteria_name == 'volopt':
        criteria = VolOpt(model)
    elif criteria_name == 'alloptminus':
        criteria = AllOptMinus(model)
    else:
        raise ValueError("Invalid criteria name!")
    original_criteria = OriginalLoss()
    return original_criteria, criteria

# select the mode of the experiment
if args.Optimized_method == 'alloptplus':
    mode = 'plus'
elif args.Optimized_method == 'volopt':
    mode = 'plus'
elif args.Optimized_method == 'alloptminus':
    mode = 'minus'
else:
    raise ValueError("Invalid Optimized_method!")

# load the SEV penalty
sev_penalty = args.sev_penalty
positive_penalty = args.positive_penalty

for i in range(10):
    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # preprocess the dataset
    encoded_y_train = np.array(y_train)
    encoded_y_test = np.array(y_test)
    encoder = DataEncoder(standard=True)
    merged_data = encoder.fit(X_neg)
    encoded_data_train = encoder.transform(X_train)
    encoded_data_test = encoder.transform(X_test)
    encoded_data_train_arr = np.array(encoded_data_train)
    encoded_data_test_arr = np.array(encoded_data_test)

    # load the SEV
    sev = SEV(None, encoder, encoded_data_train.columns)

    # load the dataset
    train_dataset = CustomDataset(encoded_data_train_arr, encoded_y_train)

    # load the dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # load the model
    model = model_loader(args.model,sev,sev_penalty,encoded_data_train_arr,encoded_y_train,positive_penalty)

    # load the criteria
    original_criteria, criteria = criteria_loader(args.Optimized_method,model)

    # load the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train the model
    model_train(model, original_criteria, criteria, optimizer, train_loader, args.num_epochs, args.warm_up)
    # evaluate the model
    y_pred_train = model.predict_proba(encoded_data_train_arr)[:,1]
    y_pred = model.predict_proba(encoded_data_test_arr)[:,1]

    train_acc = accuracy_score(encoded_y_train, y_pred_train > 0.5)
    test_acc = accuracy_score(encoded_y_test, y_pred > 0.5)
    train_auc = roc_auc_score(encoded_y_train, y_pred_train)
    test_auc = roc_auc_score(encoded_y_test, y_pred)

    # calculate the SEVPlot
    sev_arr, count = SEVPlot(model,encoder, encoded_data_test,max_depth=args.max_depth,mode=mode,max_time=args.max_time)

    try:
        # save the result in file
        result_file = copy(encoded_data_test)
        result_file['SEV'] = sev_arr
        result_file.to_csv("../Results/Exp4/data/%s_%s_%s_%s.csv"%(args.dataset,args.model,args.Optimized_method,i),index=False)
    except:
        raise ValueError("Invalid input!")


    # write the results in the file
    f = open("../Results/Exp4/result.csv",'a')
    f.write("%s,%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n"%(args.dataset,args.model,args.Optimized_method, sev_penalty,positive_penalty,train_acc,test_acc,train_auc,test_auc,np.sum(sev_arr)/count))
    f.close()


        
        