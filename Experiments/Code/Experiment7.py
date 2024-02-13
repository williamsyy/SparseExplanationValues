# Experiments 7: Times
# Description: This file contains the code for calculating the median time consumption for different explanation methods: treeshap, kernelshap, lime, dice and sev minus.

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
import shap
from sklearn.ensemble import GradientBoostingClassifier
import time
import lime
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import dice_ml
from dice_ml import Dice
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="compas")
parser.add_argument('--method', type=str, default="sev")

args = parser.parse_args()

# load the dataset
X,y,X_neg = data_loader(args.dataset)

median_lst = []

for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y, train_size=0.8)
    encoded_y_train = np.array(Y_train)
    encoded_y_test = np.array(Y_test)
    encoder = DataEncoder(standard=True)
    merged_data = encoder.fit(X_neg)
    encoded_data_train = encoder.transform(X_train)
    encoded_data_test = encoder.transform(X_test)
    encoded_data_train_arr = np.array(encoded_data_train)
    encoded_data_test_arr = np.array(encoded_data_test)

    model = GradientBoostingClassifier(n_estimators=200,max_depth=3)

    model.fit(encoded_data_train,encoded_y_train)

    encoded_data_test = encoded_data_test[model.predict(encoded_data_test)==1]
    encoded_data_test_save = encoded_data_test.copy()

    if args.method == "kernelshap":
        time_lst = []
        f = lambda x: model.predict_proba(x)[:,1]
        med = encoded_data_train.median().values.reshape((1,encoded_data_train.shape[1]))
        explainer = shap.KernelExplainer(f, med,algorithm="linear")
        for i in tqdm(range(encoded_data_test.shape[0])):
            start_time_sample = time.time()
            explainer = shap.KernelExplainer(f, med,algorithm="linear")
            shap_values = explainer.shap_values(encoded_data_test.iloc[i])
            time_lst.append(time.time()-start_time_sample)
        encoded_data_test_save["kernelSHAP"] = time_lst

    if args.method == "sev":
        time_lst = []
        for i in tqdm(range(encoded_data_test.shape[0])):
            start_time_sample = time.time()
            sev = SEV(model,encoder,encoded_data_test.columns)
            sev_num = sev.sev_cal(np.array(encoded_data_test.iloc[i]).reshape(1,-1),mode="minus",max_depth=6)
            time_lst.append(time.time()-start_time_sample)
        encoded_data_test_save["SEV"] = time_lst

    if args.method == "treeshap":
        time_lst = []
        for i in tqdm(range(encoded_data_test.shape[0])):
            start_time_sample = time.time()
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(encoded_data_test.iloc[i])
            time_lst.append(time.time()-start_time_sample)
        encoded_data_test_save["TreeSHAP"] = time_lst

    if args.method == "lime":
        time_lst = []
        for i in tqdm(range(encoded_data_test.shape[0])):
            start_time_sample = time.time()
            med = encoded_data_train.median().values.reshape((1,encoded_data_train.shape[1]))
            explainer = lime.lime_tabular.LimeTabularExplainer(encoded_data_train, 
                                                            feature_names=encoded_data_train.columns, 
                                                            class_names=np.array(["0","1"]),
                                                            discretize_continuous=False)
            exp = explainer.explain_instance(np.array(encoded_data_test)[i], model.predict_proba, num_features=len(encoded_data_test.columns), top_labels=1)
            time_lst.append(time.time()-start_time_sample)
        encoded_data_test_save["LIME"] = time_lst

    if args.method == 'dice':
        if args.dataset == "adult":
            data = pd.read_csv("../../Data/adult.data",header=None)
            data.columns = data.columns.astype(str)
            target = '14'
            X = data[[i for i in data.columns if i != target]]
            y = data[target].map({" <=50K":0," >50K":1})
            y = np.array(y)
            X_neg = X[y==0]
        elif args.dataset == "compas":
            data = pd.read_csv("../../Data/compas.txt")
            target = "two_year_recid"
            X = data[[i for i in data.columns if i != target]]
            y = data[target]
            y = np.array(y)
            X_neg = X[y==0]
        elif args.dataset == "fico":
            data = pd.read_csv("../../Data/fico.txt")
            target = "RiskPerformance"
            X = data[[i for i in data.columns if i != target]]
            y = data[target]
            y = np.array(y)
            X_neg = X[y==0]
        elif args.dataset == "german":
            data = pd.read_csv("../../Data/german.data",header=None,sep="\s+")
            data.columns = data.columns.astype(str)
            target = '20'
            X = data[[i for i in data.columns if i != target]]
            y = data[target].map({1:0,2:1})
            y = np.array(y)
            X_neg = X[y==0]
        elif args.dataset == "mimic":
            data = pd.read_csv("../../Data/oasis_mimiciii.csv").dropna()
            data = data[["age","preiculos","gcs","heartrate_min","heartrate_max","meanbp_min","meanbp_max","resprate_min","resprate_max","tempc_min","tempc_max","urineoutput","mechvent","electivesurgery","hospital_expire_flag"]]
            X = data[["age","preiculos","gcs","heartrate_min","heartrate_max","meanbp_min","meanbp_max","resprate_min","resprate_max","tempc_min","tempc_max","urineoutput","mechvent","electivesurgery"]]
            target = "hospital_expire_flag"
            y = data["hospital_expire_flag"]
            y = np.array(y)
            X_neg = X[y==0]
        elif args.dataset == "diabetes":
            data = pd.read_csv("../../Data/diabetic_data_new3.csv").dropna()
            data.columns = data.columns.astype(str)
            target = 'readmitted'
            X = data[[i for i in data.columns if i != target]]
            y = data[target].map({'NO':0,'>30':1,'<30':1})
            y = np.array(y)
            X_neg = X[y==0]
        else:
            raise ValueError("Invalid Dataset Name!")
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y, train_size=0.8)
        encoded_y_train = np.array(Y_train)
        encoded_y_test = np.array(Y_test)
        encoder = DataEncoder(standard=True)
        merged_data = encoder.fit(X)
        encoded_data_train = encoder.transform(X_train)
        encoded_data_test = encoder.transform(X_test)

        cate = []
        numer = []
        for key,value in encoder.columns_types.items():
            if value == "binary" or value == "category":
                cate.append(key)
            else:
                numer.append(key)

        print(encoder.columns_types)

        d = dice_ml.Data(dataframe=data, continuous_features=numer, outcome_name=target)

        categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        numerical_transformer = Pipeline(steps=[
            ('standard', StandardScaler())])

        transformations = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, cate),
                ('num', numerical_transformer, numer)])

        clf = Pipeline(steps=[('preprocessor', transformations),
                        ('classifier', GradientBoostingClassifier(n_estimators=200,max_depth=3))])
        model = clf.fit(X_train, Y_train)

        m = dice_ml.Model(model=model, backend="sklearn")
        exp = Dice(d, m, method="random")

        X_test_select = X_test[clf.predict(X_test)==1]
        X_test_save = X_test_select.copy()

        time_lst = []
        for i in range(X_test_select.shape[0]):
            start = time.time()
            dice_exp = exp.generate_counterfactuals(pd.DataFrame(X_test_select.iloc[i]).T, total_CFs=1, desired_class="opposite")
            time_lst.append(time.time()-start)

        X_test_save["DICE"] = time_lst

    print("The median value of %s of %s is %.4f"%(args.method,args.dataset,np.median(time_lst)))
    median_lst.append(np.median(time_lst))
print("The median of %s of %s is %.4f +- %.4f"%(args.method,args.dataset,np.mean(median_lst),np.std(median_lst)))


# time_lst = np.array(time_lst)
# time_lst = time_lst[time_lst<0.35]
# sns.histplot(time_lst,binwidth=0.01,label="SEV",alpha=0.5)
# plt.xlim(-0.05,0.35)
# plt.title("%s - SEV"%args.dataset)
# plt.savefig("../Results/Exp7/SEV_%s.png"%args.dataset,dpi=500)
# encoded_data_test_save.to_csv("../Results/Exp7/%s_1.csv"%args.dataset,index=False)

# plt.legend()
# plt.savefig("diabetes.png",dpi=500)

