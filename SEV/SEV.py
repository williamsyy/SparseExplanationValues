import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt
from itertools import product,combinations
import time

class SEV:
    """
    This is the overall SEV Calculation and the calculation procedure should be divided to two parts
    fit and search, it uses the DataEncoder and the training model as an input and initialize to get
    the map between the columns information and its features used for adjusting the Xi into its mean
    value.
    :param model: The input trained model that have fit() and predict() methods
    :param data_encoder: The fitted DataEncoder for the training dataset
    """
    def __init__(self, model, data_encoder, X_col, strict=None):
        self.model = model
        self.data_encoder = data_encoder
        self.data_mean = {}
        self.data_map = pd.DataFrame(np.zeros((len(self.data_encoder.original_columns),len(X_col))),columns = X_col, index = self.data_encoder.original_columns)
        # the least unit for the SEV version starting from the least output values
        self.least_unit = None
        self.overall_mean = []
        self.choices = list(np.arange(len(self.data_encoder.original_columns)))
        # use the dataencoder to get the mean/median value for each type of features
        for index,feature in enumerate(self.data_encoder.original_columns):
            # for numerical feature, the mean is just the mean
            if self.data_encoder.columns_types[feature] == "numerical":
                self.data_mean[feature] = self.data_encoder.columns_mean[feature]
                self.overall_mean.append(self.data_encoder.columns_mean[feature])
                self.data_map.loc[feature,feature] = 1
            # for binary feature, the mean is the mode of the feature
            elif self.data_encoder.columns_types[feature] == "binary":
                self.data_mean[feature] = self.data_encoder.columns_mean[feature]
                self.overall_mean.append(self.data_encoder.columns_mean[feature])
                self.data_map.loc[feature, feature] = 1
            # for the categorical features, suppose it is getting from the one-hot feature,
            # then reverse transform back to the one-hot encoded version, suppose it is getting
            # from the categorical features directly, use the OneEncoder to transform the value
            # into one-hot version
            elif self.data_encoder.columns_types[feature] == "category":
                mode_value = self.data_encoder.columns_mean[feature]
                if feature in self.data_encoder.merge_dict.keys():
                    result = self.data_encoder.columns_onehotdecoder[feature].reverse_transform(mode_value)
                    self.data_mean[feature] = result
                else:
                    result = self.data_encoder.columns_labelencoder[feature].transform([[mode_value]]).toarray()[0]
                    self.data_mean[feature] = result
                self.overall_mean += list(result)
                cats = [str(feature) + "=" + str(cat) for cat in self.data_encoder.columns_labelencoder[feature].categories_[0]]
                self.data_map.loc[feature, cats] = 1
            
            if (strict is not None) and (feature in strict):
                self.choices.remove(index)
        # save the data map
        self.data_map = np.array(self.data_map)

        # get the overall mean
        self.overall_mean = np.array(self.overall_mean)

        # save the results
        self.result = {}

    def transform(self, Xi, conditions):
        """
        This function aims to transfer Xi based on its boolean vector
        :param Xi: a DataFrame row for training dataset
        :param conditions: a boolean vector represents which feature should take the mean
        :return: Xi_temp: the transferred Xi
        """
        
        remain_columns = conditions.dot(self.data_map)
        Xi_temp = Xi*remain_columns + self.overall_mean *(1-remain_columns)
        return Xi_temp.reshape(1, -1)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def calculate_all(self,Xi):
        """
        Calculate all the possible combinations for the result
        :param Xi: a numpy array row in the dataset
        :return: The least value Node in the boolean lattice
        """
        # list out all the possible combinations
        combinations = list(product([0,1],repeat = len(self.data_encoder.original_columns)))
        # initialize the least score
        least_score = None
        for combination in combinations:
            # calculate the score of Xi
            score = self.model.predict_proba(self.transform(Xi,np.array(combination)))[0,1]
            # get the least score and the least score population
            if (least_score is None) or (score < least_score):
                self.least_unit = combination
                least_score = score
            # save the combination result in the dictionary
            self.result[combination]= score
        # return the least unit
        return self.least_unit

    def sev_cal(self,Xi, mode = "plus",max_depth=None):
        """
        Calculate the SEV value for Xi
        :param Xi: a DataFrame row for training dataset
        :param mode {'Mean','NegativeMost', 'Counterfactual'} default: 'Mean': the parameter to control
        what kind of SEV that we would like to calculate, 'Mean' represents to search from (0,0,0) and find
        the shortest path from it to first postive term, 'NegativeMost' represents to search the shortest path
        from the least negative outcomes value node to the first postive term and the 'Counterfactual' means to
        search from (1,1,1) to the first negative term value.
        :return: The selected SEV
        """
        if max_depth is None:
            max_depth = len(self.data_encoder.original_columns)
        choices = self.choices

        # BFS process
        for choice in  range(1,len(self.data_encoder.original_columns)+1):
            if choice > max_depth:
                return choice
            combs = combinations(choices,choice)
            for comb in combs:
                if mode == "plus":
                    pointer = np.zeros(len(self.data_encoder.original_columns))
                elif mode == "minus":
                    pointer = np.ones(len(self.data_encoder.original_columns))
                pointer[np.array(comb)] = 1-pointer[np.array(comb)]
                # print(pointer)
                # try to collect the score from the result dictionary if it is already calculated
                try:
                    score = self.result[tuple(pointer)]
                except:
                    score = self.model.predict_proba(self.transform(Xi, pointer))[0, 1]
                # for counterfactual the score should be negative
                if mode == "minus":
                    if score < 0.5:
                        return len(comb)
                else:
                    if score >= 0.5:
                        return len(comb)
    def sev_count(self,Xi, choice, mode = "plus"):
        choices = list(np.arange(len(self.data_encoder.original_columns)))
        result = []
        # BFS process
        combs = combinations(choices,choice)
        for comb in combs:
            if mode == "plus":
                pointer = np.zeros(len(self.data_encoder.original_columns))
            elif mode == "minus":
                pointer = np.ones(len(self.data_encoder.original_columns))
            pointer[np.array(comb)] = 1-pointer[np.array(comb)]
            # print(pointer)
            # try to collect the score from the result dictionary if it is already calculated
            try:
                score = self.result[tuple(pointer)]
            except:
                score = self.model.predict_proba(self.transform(Xi, pointer))[0, 1]
            # for counterfactual the score should be negative
            if mode == "minus":
                if score < 0.5:
                    result += list(np.array(self.data_encoder.original_columns)[np.array(comb)])
            else:
                if score > 0.5:
                    result += list(np.array(self.data_encoder.original_columns)[np.array(comb)])
        return set(result)

def SEVPlot(model,encoder, encoded_data,mode, savefig=None,max_depth = None, strict=[],max_time=14400):
    """
    This function is used to plot the SEV values for encoded
    :param model: The trained model
    :param encoder: The DataEncoder for the training dataset
    :param encoded_data: The fitted one-hot encoded dataset
    :param mode: The hyperparamter in SEV choosing the way of calculating SEV
    :param savefig (default: None): If the savefig is a string, then it would save the figure as
    the string
    :return: The list of SEV values for the encoded data
    """
    y_pred = model.predict(encoded_data)
    print("There are",len(y_pred), "samples in the test dataset,",np.sum(y_pred),"are predicted as positive.")
    sev = SEV(model, encoder, encoded_data.columns,strict=strict)
    sev_lst = []
    count = 0
    start_time = time.time()
    for ind,xi in enumerate(tqdm(np.array(encoded_data))):
        if (max_time is not None) and (time.time() - start_time > max_time):
            print("Time Out!")
            sev_lst.append(0)
            continue
        if y_pred[ind] == 1:
            sev_num = sev.sev_cal(xi, mode=mode,max_depth=max_depth)  # calculate the SEV
            sev_lst.append(sev_num)
            count+=1
        else:
            sev_lst.append(0)
    plt.hist(sev_lst, alpha=0.5)
    if savefig is not None:
        plt.savefig(savefig)
    plt.cla()
    print("The value counts of sev is shown below:")
    print(pd.DataFrame(sev_lst).value_counts())
    print("The average SEV for",mode,"is",np.sum(sev_lst)/count)
    return sev_lst,count

def SEVCount(model,encoder, encoded_data,mode, savefig=None,max_depth = None,strict=[],unique=False,max_time=14400):
    """
    This function is used to count the SEV values for encoded
    :param model: The trained model
    :param encoder: The DataEncoder for the training dataset
    :param encoded_data: The fitted one-hot encoded dataset
    :param mode: The hyperparamter in SEV choosing the way of calculating SEV
    :param savefig (default: None): If the savefig is a string, then it would save the figure as
    the string
    :return: The list of SEV values for the encoded data
    """
    elements_counts = {}
    y_pred = model.predict(encoded_data)
    print("There are",len(y_pred), "samples in the test dataset,",np.sum(y_pred),"are predicted as positive.")
    sev = SEV(model, encoder, encoded_data.columns,strict=strict)
    sev_lst = []
    count = 0
    start_time = time.time()
    for ind,xi in enumerate(tqdm(np.array(encoded_data))):
        if y_pred[ind] == 1:
            sev_num = sev.sev_cal(xi, mode=mode,max_depth=max_depth)  # calculate the SEV
            sev_lst.append(sev_num)
            features = sev.sev_count(xi,mode=mode,choice=sev_num)
            for feature in features:
                # if feature in strict:
                #     continue
                if unique:
                    if len(feature) != 1:
                        continue
                    if feature not in elements_counts.keys():
                        elements_counts[feature] = 1
                    else:
                        elements_counts[feature] += 1
                else:
                    if feature not in elements_counts.keys():
                        elements_counts[feature] = 1
                    else:
                        elements_counts[feature] += 1
            count+=1
        else:
            sev_lst.append(0)
        if time.time() - start_time > max_time:
            print("Time Out!")
            sev_lst = np.zeros(encoded_data.shape[0])
            break
    plt.hist(sev_lst, alpha=0.5)
    if savefig is not None:
        plt.savefig(savefig)
    plt.cla()
    print("The value counts of sev is shown below:")
    print(pd.DataFrame(sev_lst).value_counts())
    print("The average SEV for",mode,"is",np.sum(sev_lst)/count)
    print("The count of each features are")
    print(elements_counts)
    return sev_lst

