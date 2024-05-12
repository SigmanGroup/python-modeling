from __future__ import annotations
import multiprocessing
from hotspot_classes import Threshold, Hotspot
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import copy
import matplotlib.pyplot as plt
from itertools import repeat
import random


def threshold_generation(data_df:pd.DataFrame, class_weight:dict, evaluation_method:str, x_labelname_dict:dict, features:list[str] = ['empty']) -> list[Threshold]:
    """
    Given the master dataframe and some parameters, return the best threshold in each feature.

    :data_df: The master dataframe with x# column names and the first two columns as 'response' and 'y_class'
    :class_weight: Mapping of hits (1) and misses (0) to their respective class weights. Example: {1:10, 0:1}
    :evaluation_method: 'accuracy', 'weighted_accuracy', 'f1', 'weighted_f1'; Primary accuracy metric to be used in threshold comparison
    :x_labelname_dict: Dictionary for converting x# labels to full feature names
    :features: List of x# parameter names to get thresholds for.  Primarily used for manual hotspot selection.
    """

    if(features==['empty']):
        features = list(x_labelname_dict.keys())

    all_thresholds = []
    for f_ind in features:
        x = (data_df.loc[:,f_ind].values).reshape(-1, 1) # pulls the relevant parameter column and formats it in the propper array
        y = data_df.loc[:, 'y_class']
        dt = DecisionTreeClassifier(max_depth=1, class_weight=class_weight).fit(x, y)

        #Turns the dt into a Threshold object
        if(len(dt.tree_.children_left) > 1):            
            # If the amount of hits in the left subtree is greater than hits in the right subtree:
            if(dt.tree_.value[1][0][1] > dt.tree_.value[2][0][1]):
                 operator = '<'
            else:
                operator = '>'
        else:
            operator = '>'
            
        temp_threshold = Threshold(
            f_ind, 
            dt.tree_.threshold[0],
            operator, 
            feature_name = x_labelname_dict[f_ind],
            evaluation_method = evaluation_method
        )

        all_thresholds.append(temp_threshold)
    return all_thresholds


def hs_next_thresholds_fast(hs:Hotspot, all_thresholds:list[Threshold]) -> list[Hotspot]:
    """
    Given a hotspot and a list of thresholds, return a list of hotspots with each threshold added to the hotspot.

    :hs: Hotspot to add additional thresholds to
    :all_thresholds: List of thresholds to add to the hotspot
    """
    
    all_hotspots = []

    for thresh in all_thresholds:
        fresh_thresh = copy.deepcopy(thresh)
        temp_hs = copy.deepcopy(hs)
        temp_hs.add_threshold(fresh_thresh)
        all_hotspots.append(temp_hs)

    return all_hotspots

def hs_next_thresholds(hs:Hotspot, data_df:pd.DataFrame, class_weight:dict, x_labelname_dict:dict, features:list[str] = ['empty']) -> list[Hotspot]:
    """
    Given the master dataframe and some parameters, return the best threshold in each feature.

    :hs: Hotspot to add additional thresholds to
    :data_df: The master dataframe with x# column names and the first two columns as 'response' and 'y_class'
    :class_weight: Mapping of hits (1) and misses (0) to their respective class weights. Example: {1:10, 0:1}
    :x_labelname_dict: Dictionary for converting x# labels to full feature names
    :features: List of x# parameter names to get thresholds for.  Primarily used for manual hotspot selection.
    """

    if(features==['empty']):
        features = list(x_labelname_dict.keys())

    # Makes all possible hotspots by adding one threshold
    all_hotspots = []
    for f_ind in features:
        x = (data_df.loc[:,f_ind].values).reshape(-1, 1) # pulls the relevant parameter column and formats it in the propper array
        y = data_df.loc[:, 'y_class']
        dt = DecisionTreeClassifier(max_depth=1, class_weight=class_weight).fit(x, y)

        #Turns the dt into a Threshold object
        if(len(dt.tree_.children_left)>1):            
            # If the amount of hits in the left subtree is greater than hits in the right subtree:
            if(dt.tree_.value[1][0][1] > dt.tree_.value[2][0][1]):
                 operator = '<'
            else:
                operator = '>'
        else:
            operator = '>'
            
        temp_threshold = Threshold(
            f_ind, 
            dt.tree_.threshold[0],
            operator, 
            feature_name = x_labelname_dict[f_ind],
            evaluation_method = hs.evaluation_method
        )

        temp_hs = copy.deepcopy(hs)
        temp_hs.add_threshold(temp_threshold)
        all_hotspots.append(temp_hs)

    return all_hotspots


def prune_hotspots(hotspots:list[Hotspot], percentage:int, evaluation_method:str) -> list[Hotspot]:
    """
    Given a list of hotspots, returns the top percentage back.

    :hotspots: List of hotspots to be compared
    :percentage: Percentage of hotspots to keep
    :evaluation_method: 'accuracy', 'weighted_accuracy', 'f1', 'weighted_f1'; What metric to use when comparing hotspots
    """
    accuracy_list=[]
    for hs in hotspots:
        accuracy_list.append(hs.accuracy_dict[evaluation_method])

    cut = np.percentile(accuracy_list, 100 - percentage)
    
    hs_out=[]
    for hs in hotspots:
        if(hs.accuracy_dict[evaluation_method]>=cut):
            hs_out.append(hs)
    
    return hs_out

def train_test_splits(temp_data_df:pd.DataFrame, split:str, test_ratio:float, x_labels:list[str], response_label:str, randomstate:int = 0, defined_training_set:list[int] = [], defined_test_set:list[int] = [], subset:list[int] = [], verbose:bool = True) -> tuple[list[int], list[int]]:
    """
    Given the main dataframe and some parameters, return lists of y index values for a training and test set

    :data_df: The master dataframe with x# column names and the first two columns as 'response' and 'y_class'
    :split: 'random', 'ks', 'y_equidist', 'define', 'none'; Type of split to use
    :test_ratio: Ratio of the data to use as a test set
    :x_labels: List of xID# parameter labels corresponding to the parameter column names in the dataframe
    :response_label: The name of the response column in the dataframe
    :randomstate: Seed to use when chosing the random split
    :defined_training_set: Y indexes corresponding to a manual training set. Only used if split == 'define'
    :defined_test_set: Y indexes corresponding to a manual test set. Only used if split == 'define'
    :subset: The subset of y indexes to use for another split method, originally used for MLR after a classification algorithm
    :verbose: Whether to print the extended report
    """

    import kennardstonealgorithm as ks

    
    if (subset == []):
        data_df = temp_data_df.copy()
    else:
        data_df = temp_data_df.loc[subset, :].copy()

    x = data_df[x_labels].to_numpy() # Array of just feature values (X_sel)
    y = data_df[response_label].to_numpy() # Array of response values (y_sel)
    test_size = int(len(data_df.index)*test_ratio) # Number of points in the test set
    train_size = len(data_df.index) - test_size

    if split == "random":
        random.seed(a = randomstate)
        test_set = random.sample(list(data_df.index), k = test_size)
        training_set = [x for x in data_df.index if x not in test_set]

    elif split == "ks":
        # There may be some issues with test_set_index being formatted as an array and training_set_index being a list
        training_set_index, test_set_index = ks.kennardstonealgorithm(x, train_size)
        training_set = list(data_df.index[training_set_index])
        test_set = list(data_df.index[test_set_index])

    elif split == "y_equidist":
        no_extrapolation = True
        # Only difference I can see between extrapolation and no_extrapolation is that no_e cuts off the highest and lowest y values first
        
        if no_extrapolation:
            # Rewritten from above to keep track of which points were removed for being equal to the min or max value
            y_min = np.min(y)
            y_max = np.max(y)
            y_ks = np.array(([i for i in y if i not in [y_min,y_max]]))
            y_ks_indices = [i for i, val in enumerate(y) if val != y_min and val != y_max]
            y_not_ks_indices = [i for i, val in enumerate(y) if val == y_min or val == y_max]

            # indices relative to y_ks:
            y_ks_formatted = y_ks.reshape(np.shape(y_ks)[0], 1)
            VS_ks,TS_ks = ks.kennardstonealgorithm(y_ks_formatted, test_size)

            # indices relative to y_sel:
            TS_ = sorted([y_ks_indices[i] for i in list(TS_ks)]+y_not_ks_indices) # Replaced minmax with y_not_ks_indices
            VS_ = sorted([y_ks_indices[i] for i in VS_ks])

        else:
            VS_,TS_ = ks.kennardstonealgorithm(y.reshape(np.shape(y)[0],1),int((test_ratio)*np.shape(y)[0]))

        training_set = list(data_df.index[TS_])
        test_set = list(data_df.index[VS_])

    elif split == 'define':
        training_set = defined_training_set
        test_set = defined_test_set

    elif split == "none":
        training_set = data_df.index.to_list()
        test_set = []

    else: 
        raise ValueError("split option not recognized")
    
    if(verbose):
        y_train = data_df.loc[training_set, response_label]
        y_test = data_df.loc[test_set, response_label]

        print(f"Training Set: {training_set}")
        print(f"Test Set: {test_set}")
        if (len(training_set) + len(test_set) == len(data_df.index)):
            print('All indices accounted for!')
        else:
            print('Missing indices!')

        print("Training Set mean: {:.3f}".format(np.mean(y_train)))
        print("Test Set mean: {:.3f}".format(np.mean(y_test)))
        # print("Shape X_train: {}".format(X_train.shape))
        # print("Shape X_test:  {}".format(X_test.shape))   
        plt.figure(figsize=(5, 5))
        hist, bins = np.histogram(y,bins="auto")#"auto"
        plt.hist(y_train, bins, alpha=0.5, label='y_train',color="black")
        plt.hist(y_test, bins, alpha=0.5, label='y_test')
        plt.legend(loc='best')
        plt.xlabel("Output",fontsize=20)
        plt.ylabel("N samples",fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.show()

    return training_set, test_set