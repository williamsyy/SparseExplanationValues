# SparseExplanationValues

The introduction of SEV has provided an sparse explanation for each unit based on a binary machine learning classifier and new ways to increase this sparsity explanation without its performance (like Accuracy and AUC). This code is used for **NeurIPS 2023 Submission**, which contains implementations and Experiments for *Sparse Explanations Values (SEV)* and two optimal methods: *All-Opt* and *Vol-Opt*. 

# Development

Guidelines for developers who wants to reproduce the experiments results, modify and test the code.

## Repository Structure

- **SEV**: The implementation of the definition of Sparse Explanations and two optimization methods. 

It consists 5 files. The *SEV.py* file is the implementation of different **SEV** definitions: **SEV+**, **SEV-** and **SEV**$^{\circledR}$. *Encoder.py* file the data encoder for doing one-hot encoding for the general datasets, save the median values for numerical features and mode values for categorical features. *OptimizedSEV.py* includes all the optimization methods: **AllOpt+** for optimizing **SEV+**, **AllOpt-** for optimizing **SEV-**, **AllOptR** for optimizing **SEV**$^{\circledR}$, and **VolOpt** for optimizing **SEV+** in linear classifiers. It also includes the models used in the experiments: Logisitic Regression(**SimpleLR**), 2-layer MLP(**SimpleMLP**) and Graident Boosting Decision Trees(**GBDT**). 

- **Plots**: The code for generating figures in the paper

- **Experiments**: The code for different experiments

The expeirments folder consists of two parts: *Code* and *Scripts*. The code folder contains thepython files for different experiments and scripts folder shows how to submit the experiments through slurm.

## General Usage of SEV

The usage of SEV can reference the Experiment Part of SEV. The general usage of SEV can be used as follows:

```
from SEV.SEV import SEVPlot, SEV
from SEV.data_loader import data_loader
from SEV.Encoder import DataEncoder
from sklearn.linear_model import LogisticRegression

# get the dataset and the negative population
X,y,X_neg = data_loader(args.dataset)

# preprocessing the dataset
encoder = DataEncoder(standard=True)
# fit the encoder with the negative population
encoder.fit(X_neg)
# transform the whole dataset
encoded_X = encoder.transform(X)

# construction the model
lr = LogisiticRegression()
lr.fit(encoded_X,y)

# for explaining the whole dataset, "plus" for SEV+, "minus" for SEV-
SEVPlot(lr,encoder, encoded_X, "plus")

# for explaning the one single instance
sev = SEV(lr,encoder,encoded_X.columns)
# get the number of sev for this instance
sev_num = sev.sev_cal(np.array(encoded_X.iloc[0]).reshape(1,-1),mode="plus")
print("The SEV Value for instance 0 is %d."%sev_num)
# get the features can be used in this explanation
features = sev.sev_count(np.array(encoded_X.iloc[0]).reshape(1,-1),mode="plus",choice=sev_num)
print("The feature used in this explanation are %s."%features)
```

The use of different optimization methods can be seen in Experiment 1 in Experiment Folder.

# Requirements

The following dependencies need to be installed to run the experiments.

```
numpy==1.23
pandas==1.5.2
scikit-learn==1.2.2
pytorch==2.0.0
matplotlib==3.7.1
seaborn==0.12.2
psankey==1.0.1
dice-ml==0.9
shap==0.40.0
lime==0.2.0
```

Note: The sankeyplots are generated based on the *psankey* packages (https://github.com/mandalsubhajit/psankey). 
