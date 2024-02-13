# Experiments 8
# Applied Chen,2018 performance on different datasets

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
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
# import SEV
from SEV.SEV import SEV,SEVPlot
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_name', type=str, default="adult")

our_model_performance = {
    "adult": {"Allopt+":[0.84,0.90],"Allopt-":[0.85,0.93]},
    "compas": {"Allopt+":[0.68,0.73],"Allopt-":[0.68,0.74]},
    "mimic": {"Allopt+":[0.89,0.78],"Allopt-":[0.89,0.78]},
    "fico":{ "Allopt+":[0.67,0.78],"Allopt-":[0.67,0.75]},
    "german": {"Allopt+":[0.73,0.80],"Allopt-":[0.76,0.81]},
    "diabetes": {"Allopt+":[0.67,0.78],"Allopt-":[0.67,0.75]}
}

args = parser.parse_args()

train_acc_total = []
test_acc_total = []
train_auc_total = []
test_auc_total = []
original_acc = []
original_auc = []

for random_state in range(1,10):
    print("Current random state: ", random_state)
    # load the dataset
    X,y,X_neg = data_loader(args.data_name)

    # preprocessing the dataset
    encoder = DataEncoder(standard=True)
    encoder.fit(X_neg)
    encoded_X = encoder.transform(X)

    # do the training and testing split
    X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.2, random_state=random_state,stratify=y)

    # generate the sev
    sev = SEV(None, encoder, encoded_X.columns)

    # convert the data into torch tensor
    X_train = torch.from_numpy(X_train.values).float()
    X_test = torch.from_numpy(X_test.values).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()



    # create a mlp model using pytorch
    class SimpleL2X(nn.Module):
        def __init__(self, n_features, k, tau, sev):
            super(SimpleL2X, self).__init__()
            self.linear1 = nn.Linear(n_features, 100)
            self.linear2 = nn.Linear(100, 100)
            self.linear3 = nn.Linear(100, len(sev.data_encoder.original_columns))
            self.softmax = nn.Softmax(dim = 1)
            self.linear1_new = nn.Linear(n_features, 128)
            self.linear2_new = nn.Linear(128, 1)
            self.data_map = torch.tensor(sev.data_map).float()
            self.data_mean = torch.tensor(sev.overall_mean).float()
            self.k = k
            self.tau = tau
            self.sev = sev

        def forward(self, x):
            temp1 = torch.relu(self.linear1(x))
            temp2 = torch.relu(self.linear2(temp1))
            logits = self.linear3(temp2)
            logits_ = torch.unsqueeze(logits, 1)
            uniform = torch.rand(logits.shape[0], self.k, logits.shape[1])
            gumbel = - torch.log(-torch.log(uniform + 1e-10))
            noisy_logits = (gumbel + logits_)/self.tau
            samples = self.softmax(noisy_logits)
            samples = torch.max(samples, axis = 1)[0]
            new_input = x * samples.matmul(self.data_map) + self.data_mean * (1 - samples.matmul(self.data_map))
            new_input = self.linear1_new(new_input)
            new_input = torch.relu(new_input)
            new_input = self.linear2_new(new_input)
            final_output = torch.sigmoid(new_input)
            return final_output.squeeze()
        
        def predict(self,x):
            temp1 = torch.relu(self.linear1(x))
            temp2 = torch.relu(self.linear2(temp1))
            logits = self.linear3(temp2)
            threshold = torch.unsqueeze(torch.topk(logits, self.k, sorted = True)[0][:,-1], -1)
            discrete_logits = torch.where(logits >= threshold, torch.ones_like(logits), torch.zeros_like(logits)).detach()
            new_input = x * discrete_logits.matmul(self.data_map) + self.data_mean * (1 - discrete_logits.matmul(self.data_map))
            new_input = self.linear1_new(new_input)
            new_input = torch.relu(new_input)
            new_input = self.linear2_new(new_input)
            final_output = torch.sigmoid(new_input)
            return final_output.squeeze()

        def create_samples(self, x):
            temp1 = torch.relu(self.linear1(x))
            temp2 = torch.relu(self.linear2(temp1))
            logits = self.linear3(temp2)
            threshold = torch.unsqueeze(torch.topk(logits, self.k, sorted = True)[0][:,-1], -1)
            discrete_logits = torch.where(logits >= threshold, torch.ones_like(logits), torch.zeros_like(logits))
            return discrete_logits.matmul(self.data_map)
        
    class TwoLayerMLP(nn.Module):
        # this is a three layer MLP
        def __init__(self, n_features):
            super(TwoLayerMLP, self).__init__()
            self.linear1 = nn.Linear(n_features, 128)
            self.linear2 = nn.Linear(128, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            temp1 = torch.relu(self.linear1(x))
            logits = self.linear2(temp1)
            return self.sigmoid(logits).squeeze()

    # create a model
    model = TwoLayerMLP(n_features=encoded_X.shape[1])

    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # make a data loader
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    # train the MLP
    for epoch in tqdm(range(100)):
        # print("Starting Epoch {}".format(epoch))
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x)
            loss_value = loss(y_pred, y)
            loss_value.backward()
            optimizer.step()

    # test the model
    y_pred = model(X_test)
    y_pred = y_pred.detach().numpy()
    y_test = y_test.detach().numpy()
    print("Accuracy: {}".format(accuracy_score(y_test, y_pred>0.5)))
    print("AUC: {}".format(roc_auc_score(y_test, y_pred)))
    original_acc.append(accuracy_score(y_test, y_pred>0.5))
    original_auc.append(roc_auc_score(y_test, y_pred))

    train_acc = []
    test_acc = []
    train_auc = []
    test_auc = []

    for k in range(1, len(encoder.original_columns) + 1):
    # for k in range(1,3):
        print("Starting k = {}".format(k))
        model_new = SimpleL2X(n_features=encoded_X.shape[1],k=k, tau=0.01, sev=sev)

        model_new.linear1_new.load_state_dict(model.linear1.state_dict())
        model_new.linear2_new.load_state_dict(model.linear2.state_dict())
        # model_new.linear3.load_state_dict(model.linear3.state_dict())

        loss = nn.BCELoss()
        # optimizer = torch.optim.Adam(model_new.parameters(), lr=0.01)
        params_to_update = list(model_new.linear1.parameters()) + list(model_new.linear2.parameters()) + list(model_new.linear3.parameters()) + list(model_new.softmax.parameters())
        optimizer = torch.optim.Adam(params_to_update, lr=0.01)

        # make a data loader
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)


        # train the model
        for epoch in tqdm(range(100)):
            # print("Starting Epoch {}".format(epoch))
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                y_pred = model_new(x)
                loss_value = loss(y_pred.float(), y)
                loss_value.backward()
                optimizer.step()

        # test the training dataset
        y_pred = model_new.predict(X_train)
        y_pred = y_pred.detach().numpy()
        y_pred_train = y_pred
        y_train_arr = y_train.detach().numpy()
        print("Train Accuracy: {}".format(accuracy_score(y_train_arr, y_pred>0.5)))
        print("Train AUC: {}".format(roc_auc_score(y_train_arr, y_pred)))

        # test the model
        y_pred = model_new.predict(X_test)
        y_pred = y_pred.detach().numpy()
        # y_test = y_test.detach().numpy()
        print("Test Accuracy: {}".format(accuracy_score(y_test, y_pred>0.5)))
        print("Test AUC: {}".format(roc_auc_score(y_test, y_pred)))

        # get the sample weights
        sample_weights = model_new.create_samples(X_test)[y_pred>0.5].detach().numpy()
        sample_select = sample_weights.dot(sev.data_map.T)>0
        # print(sample_select.sum(axis=1).shape)
        # print(sample_select.sum(axis=1).mean())

        # save in list
        train_acc.append(accuracy_score(y_train_arr, y_pred_train>0.5))
        test_acc.append(accuracy_score(y_test, y_pred>0.5))
        train_auc.append(roc_auc_score(y_train_arr, y_pred_train))
        test_auc.append(roc_auc_score(y_test, y_pred))
    
    train_acc_total.append(train_acc)
    test_acc_total.append(test_acc)
    train_auc_total.append(train_auc)
    test_auc_total.append(test_auc)

train_acc_total = np.array(train_acc_total)
test_acc_total = np.array(test_acc_total)
train_auc_total = np.array(train_auc_total)
test_auc_total = np.array(test_auc_total)

plt.errorbar(np.arange(1, len(encoder.original_columns) + 1), train_acc_total.mean(axis=0),yerr=train_acc_total.std(axis=0), label="Train Accuracy",alpha=0.7)
plt.errorbar(np.arange(1, len(encoder.original_columns) + 1), test_acc_total.mean(axis=0),yerr=test_acc_total.std(axis=0),  label="Test Accuracy",alpha=0.7)
plt.axhline(y=np.mean(original_acc), linestyle='--', label="Original Test Accuracy",color='r',alpha=0.5)
plt.axhline(y=our_model_performance[args.data_name]["Allopt+"][0], linestyle='--', label="AllOpt+ Accuracy",color='g',alpha=0.5)
plt.axhline(y=our_model_performance[args.data_name]["Allopt-"][0], linestyle='--', label="AllOpt- Accuracy",color='b',alpha=0.5)
plt.xlabel("Number of Features",fontsize=12)
plt.ylabel("Accuracy",fontsize=12)
plt.xticks(np.arange(1, len(encoder.original_columns) + 1,2),fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.legend()
plt.savefig("../Results/Exp8/accuracy_%s.png"%args.data_name)
plt.show()

plt.close()

plt.errorbar(np.arange(1, len(encoder.original_columns) + 1), train_auc_total.mean(axis=0),yerr=train_auc_total.std(axis=0),  label="Train AUC",alpha=0.7)
plt.errorbar(np.arange(1, len(encoder.original_columns) + 1), test_auc_total.mean(axis=0),yerr=test_auc_total.std(axis=0),  label="Test AUC",alpha=0.7)
plt.axhline(y=np.mean(original_auc), linestyle='--', label="Original Test AUC",color='r',alpha=0.5)
plt.axhline(y=our_model_performance[args.data_name]["Allopt+"][1], linestyle='--', label="AllOpt+ AUC",color='g',alpha=0.5)
plt.axhline(y=our_model_performance[args.data_name]["Allopt-"][1], linestyle='--', label="AllOpt- AUC",color='b',alpha=0.5)
plt.xlabel("Number of Features",fontsize=12)
plt.ylabel("AUC",fontsize=12)
plt.xticks(np.arange(1, len(encoder.original_columns) + 1,2),fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.legend()
plt.savefig("../Results/Exp8/auc_%s.png"%args.data_name)
plt.show()
