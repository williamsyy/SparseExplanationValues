import pandas as pd
import numpy as np

# this function is used to load the dataset
# input:
# data_name (str)
# output:
# X: the feature matrix (pd.DataFrame)
# y: the label vector (np.array)
# X_neg: the negative samples feature matrix (pd.DataFrame)

def data_loader(data_name):
    print("Start Loading the dataset ...")
    if data_name == "adult":
        data = pd.read_csv("../../Data/adult.data",header=None)
        data.columns = data.columns.astype(str)
        target = '14'
        X = data[[i for i in data.columns if i != target]]
        y = data[target].map({" <=50K":0," >50K":1})
        y = np.array(y)
        X_neg = X[y==0]
    elif data_name == "compas":
        data = pd.read_csv("../../Data/compas.txt")
        target = "two_year_recid"
        X = data[[i for i in data.columns if i != target]]
        y = data[target]
        y = np.array(y)
        X_neg = X[y==0]
    elif data_name == "fico":
        data = pd.read_csv("../../Data/fico.txt")
        target = "RiskPerformance"
        X = data[[i for i in data.columns if i != target]]
        y = data[target]
        y = np.array(y)
        X_neg = X[y==0]
    elif data_name == "german":
        data = pd.read_csv("../../Data/german.data",header=None,sep="\s+")
        data.columns = data.columns.astype(str)
        target = '20'
        X = data[[i for i in data.columns if i != target]]
        y = data[target].map({1:0,2:1})
        y = np.array(y)
        X_neg = X[y==0]
    elif data_name == "mimic":
        data = pd.read_csv("../../Data/oasis_mimiciii.csv").dropna()
        X = data[["age","preiculos","gcs","heartrate_min","heartrate_max","meanbp_min","meanbp_max","resprate_min","resprate_max","tempc_min","tempc_max","urineoutput","mechvent","electivesurgery"]]
        y = data["hospital_expire_flag"]
        y = np.array(y)
        X_neg = X[y==0]
    elif data_name == "diabetes":
        data = pd.read_csv("../../Data/diabetic_data_new3.csv").dropna()
        data.columns = data.columns.astype(str)
        target = 'readmitted'
        X = data[[i for i in data.columns if i != target]]
        y = data[target].map({'NO':0,'>30':1,'<30':1})
        y = np.array(y)
        X_neg = X[y==0]
    else:
        raise ValueError("Invalid Dataset Name!")

    print("The dataset is", data_name)
    print("The shape of X is",X.shape)

    return X,y,X_neg