# Experiments 5: DiCE and SEV- Comparison
# This codes aims to generate counterfactual explanations of DiCE and SEV- and see which makes more sense.


import sys
# Insert the path to the parent directory
sys.path.append('../../')
import numpy as np
import pandas as pd
from SEV.SEV import SEV,SEVPlot
from SEV.Encoder import DataEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import argparse
import shap
import lime
import dice_ml
from dice_ml import Dice
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# load the dataset
data_name="compas"
print("Start Loading the dataset ...")
if data_name == "adult":
    data = pd.read_csv("../../Data/adult.data",header=None)
    data.columns = data.columns.astype(str)
    data.columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","Income"]
    target = 'Income'
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
elif data_name == "mimic":
    data = pd.read_csv("../../Data/oasis_mimiciii.csv").dropna().set_index("subject_id")
    data = data[["age","preiculos","gcs","heartrate_min","heartrate_max","meanbp_min","meanbp_max","resprate_min","resprate_max","tempc_min","tempc_max","urineoutput","mechvent","electivesurgery","hospital_expire_flag"]]
    X = data[["age","preiculos","gcs","heartrate_min","heartrate_max","meanbp_min","meanbp_max","resprate_min","resprate_max","tempc_min","tempc_max","urineoutput","mechvent","electivesurgery"]]
    target = "hospital_expire_flag"
    y = data["hospital_expire_flag"]
    y = np.array(y)
    X_neg = X[y==0]

print("The dataset is", data_name)
print("The shape of X is",X.shape)

# split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y, train_size=0.8,random_state=42)
X_test = X_test.iloc[:500]
Y_test = Y_test[:500]
encoded_y_train = np.array(Y_train)
encoded_y_test = np.array(Y_test)
encoder = DataEncoder(standard=True)
merged_data = encoder.fit(X_neg)
encoded_data_train = encoder.transform(X_train)
encoded_data_test = encoder.transform(X_test)
encoded_data_train_arr = np.array(encoded_data_train)
encoded_data_test_arr = np.array(encoded_data_test)

# generate a model for SEV model explanation
model = GradientBoostingClassifier(n_estimators=200,max_depth=3,subsample=0.8,random_state=42)
model.fit(encoded_data_train,encoded_y_train)

# generate a model for DiCE Explanations
cate = []
numer = []
for key,value in encoder.columns_types.items():
    print(key)
    if value == "binary" or value == "category":
        cate.append(key)
    else:
        numer.append(key)

# generate a model for DiCE Explanations
d = dice_ml.Data(dataframe=data, continuous_features=numer, outcome_name=target)

categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop="if_binary",handle_unknown='ignore'))])
numerical_transformer = Pipeline(steps=[
            ('standard', StandardScaler())])
transformations = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, cate),
        ('num', numerical_transformer, numer)])
clf = Pipeline(steps=[('preprocessor', transformations),
                ('classifier', GradientBoostingClassifier(n_estimators=200,max_depth=3,random_state=42))])

model_dice = clf.fit(X_train, Y_train)
m = dice_ml.Model(model=model_dice, backend="sklearn")
exp = Dice(d, m, method="random")

X_test[(clf.predict(X_test)==1)&(model.predict(encoded_data_test)==1)].to_csv("result.csv")
test_X = X_test[(clf.predict(X_test)==1)&(model.predict(encoded_data_test)==1)]
print(X_test[(clf.predict(X_test)==1)&(model.predict(encoded_data_test)==1)])
# generate DiCE explanations
dice_exp = exp.generate_counterfactuals(test_X, total_CFs=1, desired_class="opposite")
for i in range(len(dice_exp._cf_examples_list)):
    original = encoder.transform(test_X.iloc[i])
    sev = SEV(model, encoder, encoded_data_test.columns)
    # generate SEV explanations
    sev_num = sev.sev_cal(np.array(original), mode="minus")
    if sev_num is None:
        continue
    features = sev.sev_count(np.array(original),mode="minus",choice=sev_num)
    print(features,sev_num)
    print(test_X.columns)
    print(dice_exp._cf_examples_list[i].test_instance_df.values[0].astype(str))
    print(dice_exp._cf_examples_list[i].final_cfs_df.values[0].astype(str))
print(X_neg.median())
print(X_neg.mode())
# for i in range(X_test.shape[1]):
#     selected_sample = pd.DataFrame(X_test.iloc[i]).T
#     y_pred = clf.predict(selected_sample)
#     dice_exp = exp.generate_counterfactuals(selected_sample, total_CFs=1, desired_class="opposite")
#     print(dice_exp.cf_examples_list[0].final_cfs_df)
