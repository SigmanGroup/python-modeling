from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import copy
import matplotlib
import matplotlib.pyplot as plt


# Removed correlations entirely
class Threshold:
    def __init__(self, index: str, cut_value: float, operator: str, feature_name:str, evaluation_method:str):
        """
        This is basically a container to keep a handful of variables in one place for easy use.
        Each Threshold object within a Hotspot represents a single cutoff in the dataset.

        :index: The numerical index in the old array system corresponding to the parameter for this threshold - Replaced by x# string
        :cut_value: The parameter value where the threshold divides the dataset
        :operator: a string containing < or > to indicate which side of the threshold is the active side
        :evaluation_method: selected accuracy metric
        """

        self.index = index
        self.cut_value = cut_value
        self.operator = operator
        self.evaluation_method = evaluation_method
        self.feature_label = index
        self.feature_name = feature_name

        self.added_accuracy = 0
    
    def __str__(self):
        return f'{self.feature_label} {self.feature_name} {self.operator} {self.cut_value:.3f} with Added {self.evaluation_method} of {self.added_accuracy:.3f}'
    

class Hotspot:
    def __init__(self, data_df: pd.DataFrame,  thresholds: list[Threshold], y_cut:float,  ts: list[int] = [], vs: list[int] = [], evaluation_method: str = 'weighted_accuracy', class_weight: dict = {1:10, 0:1}):
        """
        This is where most of the actual computations happen
        An object to hold a multiple thresholds and some methods to work with them

        :data_df: the main dataframe containing parameters and responses
        :thresholds: a list of Thresholds that make up the Hotspot
        :y_cut: the cutoff value for the y_class column in data_df
        :ts: data_df index values for the training set
        :vs: data_df index values for the test set
        :evaluation_method: the metric used for comparing Hotspot quality
        :class_weight: dictionary linking classes [0, 1] to relative weights ([10, 1] by default)
        """

        # store some of the variables passed in when initialized
        self.data_df = data_df
        self.evaluation_method = evaluation_method
        self.class_weight = class_weight
        self.y_cut = y_cut
        
        # set up the training and test set indices from the old array system
        # this either needs to be removed or remade
        if(ts == []):
            ts = data_df.index.tolist()
        self.training_set = ts
        self.test_set = vs

        # This reads in and stores any thresholds passed then sets some variables associated with them
        self.thresholds: list[Threshold] = []
        self.__set_accuracy()
        self.initial_accuracy = self.accuracy
        for thresh in thresholds:
            self.add_threshold(thresh)
    
        self.threshold_indexes = self.__get_threshold_indexes()
        # self.threshold_indexes.sort()

        # Removed because I don't want to deal with this right now
        # self.correlated_features = self.__set_correlated_features()
            
    def __str__(self):
        """Calling as a string returns some accuracy metrics and a print out of each threshold"""

        # Set initial_true_accuracy to the ratio of 1 to 0 in the dataset
        initial_true_accuracy = self.data_df['y_class'].sum() / len(self.data_df)

        output = f'Total {self.evaluation_method} with {len(self.thresholds)} thresholds: {self.accuracy:.3f}\n'
        output = output + f'Initial {self.evaluation_method} with no thresholds: {self.initial_accuracy:.3f}\n'
        output = output + f'Total accuracy with {len(self.thresholds)} thresholds: {self.accuracy_dict["accuracy"]:.3f}\n'
        output = output + f'Initial accuracy with no thresholds: {initial_true_accuracy:.3f}\n'

        output = output + 'Thresholds: \n'
        for thresh in self.thresholds:
            output = output + '\t' + str(thresh) + '\n'
        
        return output
    
    def __eq__(self, other: 'Hotspot'):
        """If two hotspots use the same threshold parameters and have the same accuracy, they are considered equal"""
        output = self.threshold_indexes == other.threshold_indexes
        output = output and (self.accuracy == other.accuracy)
        return output
    
    def __deepcopy__(self, memo):
        '''Create a new instance of the class without copying the DataFrame'''
        new_instance = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_instance  # Avoid infinite recursion
        for key, value in self.__dict__.items():
            # Exclude the DataFrame attribute from deep copy
            if key != 'data_df':
                setattr(new_instance, key, copy.deepcopy(value, memo))
        # copy the reference to the data_df without duplicating memory
        setattr(new_instance, 'data_df', self.data_df)
        return new_instance

    def add_threshold(self, threshold: Threshold):
        """
        Adds a threshold to the hotspot and updates all relevant statistics.
        No return value.

        :threshold: The threshold object to be added
        """
        # if(len(self.thresholds) == 0):
        #     temp_accuracy = 0
        # else:
        #     temp_accuracy = self.accuracy
        temp_accuracy = self.accuracy
        
        self.thresholds.append(threshold)
        self.__set_accuracy()
        self.__set_train_test_accuracy()
        added_accuracy = self.accuracy - temp_accuracy
        self.thresholds[-1].added_accuracy = added_accuracy
        
        self.threshold_indexes = self.__get_threshold_indexes()
        # self.threshold_indexes.sort()
        #self.correlated_features = self.__set_correlated_features()
       
    # It might be easier to just call this function instead of storing the result and calling that
    # As it's currently written, it would be more accurate to call this __get_threshold_indexes
    def __get_threshold_indexes(self) -> list[str]:
        """Returns the list of x# parameter labels for all thresholds in the hotspot"""
        indexes = []
        for thresh in self.thresholds:
            indexes.append(thresh.index)
        return indexes

    def __evaluate_threshold(self, value: float, operator: str, cutoff: float) -> bool:
        """Returns the truth of [value operator (> or <) cutoff]
        
        :value: the value to be evaluated in comparison to cutoff
        :operator: '>' or '<', used to compare value and cutoff
        :cutoff: number pulled from a threshold and used as a benchmark for value
        """
        output = False
        if (operator == '<'):
            if(value < cutoff):
                output = True
        elif(operator == '>'):
            if(value > cutoff):
                output = True
        return output
        
    def __is_inside(self, y_index: int, x_space: pd.DataFrame = None) -> list[bool]:
        """
        Currently looks at a molecule identified by index in x_space and sees if it's inside this hotspot.  Returns a list of bools corresponding to each threshold.

        :y_index: the relevant molecule/line in X_space
        :X_space: a parameter dataframe from which you get the parameters for your molecule/line of interest.
            Default is self.data_df, but it can be overwritten if you want to look at expanding the scope.
        """

        # Set the default x_space to the main dataframe
        if x_space is None:
            x_space = self.data_df

        # If there are no thresholds, the whole dataset is considered inside.
        if (len(self.thresholds) == 0):
            return [True]

        bool_list = []
        for thresh in self.thresholds:
            parameter_value = x_space.loc[y_index, thresh.index]
            bool_list.append(self.__evaluate_threshold(parameter_value, thresh.operator, thresh.cut_value))
        return bool_list

    def __set_accuracy(self):
        """Sets self accuracy, f1, weighted accuracy and weighted f1 in the self.accuracy_dict dictionary"""

        tp,tn,fp,fn = 0,0,0,0 # True Positive, True Negative, False Positive, False Negative
        
        for i in self.data_df.index:
            if (self.data_df.loc[i, 'y_class'] == 1):
                if(all(self.__is_inside(i))):
                    tp = tp + 1
                else:
                    fn = fn + 1
            if (self.data_df.loc[i, 'y_class'] == 0):
                if(all(self.__is_inside(i))):
                    fp = fp + 1
                else:
                    tn = tn + 1
        
        try:
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = (2*tp) / (2*tp + fn + fp)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            if(tp + fn == 0):
                print('ERROR: No positive examples in the dataset. Check the y_cut and data_df for errors.')
                raise
            if(tp + fp == 0):
                # This happens if a combination of thresholds predicts no positive examples
                # This is not a problem, but does break the math for precision
                precision = 0
            else:
                print('ERROR: ZeroDivisionError in accuracy calculation.  Check the data_df for errors.')
                raise

        # Weights the confusion matrix to calculate the weighted statistics
        tp = tp * self.class_weight[1]
        tn = tn * self.class_weight[0]
        fp = fp * self.class_weight[0]
        fn = fn * self.class_weight[1]
        
        weighted_accuracy = (tp + tn) / (tp + tn + fp + fn)

        try:
            weighted_f1 = (2*tp) / (2*tp + fn + fp)
        except ZeroDivisionError:
            weighted_f1 = 0

        # Sets self.accuracy to the accuracy statistic in evaluation_method
        self.accuracy_dict = {'accuracy':float(accuracy), 'weighted_accuracy':float(weighted_accuracy), 'f1':float(f1), 'weighted_f1':float(weighted_f1),
                             'precision':float(precision), 'recall':float(recall)}
        self.accuracy = self.accuracy_dict[self.evaluation_method]
    
    def __set_train_test_accuracy(self):
        """Sets training and test accuracy dictionaries with all the accuracy stats"""

        tp,tn,fp,fn = 0,0,0,0
        for i in self.training_set:
            if (self.data_df.loc[i, 'y_class'] == 1):
                if(all(self.__is_inside(i))):
                    tp = tp + 1
                else:
                    fn = fn + 1
            if (self.data_df.loc[i, 'y_class'] == 0):
                if(all(self.__is_inside(i))):
                    fp = fp + 1
                else:
                    tn = tn + 1
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = (2*tp) / (2*tp + fn + fp)

        tp = tp * self.class_weight[1]
        tn = tn * self.class_weight[0]
        fp = fp * self.class_weight[0]
        fn = fn * self.class_weight[1]
        
        weighted_accuracy = (tp + tn) / (tp + tn + fp + fn)
        weighted_f1 = (2*tp) / (2*tp + fn + fp)

        # Sets self.accuracy to the accuracy statistic in evaluation_method
        self.train_accuracy_dict = {'accuracy':float(accuracy), 'weighted_accuracy':float(weighted_accuracy), 'f1':float(f1), 'weighted_f1':float(weighted_f1)}
        
        # If there is a test set, find its accuracy
        if(len(self.test_set) > 0):
            tp,tn,fp,fn = 0,0,0,0
            for i in self.test_set:
                if (self.data_df.loc[i, 'y_class'] == 1):
                    if(all(self.__is_inside(i))):
                        tp = tp + 1
                    else:
                        fn = fn + 1
                if (self.data_df.loc[i, 'y_class'] == 0):
                    if(all(self.__is_inside(i))):
                        fp = fp + 1
                    else:
                        tn = tn + 1

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            f1 = (2*tp) / (2*tp + fn + fp)

            tp = tp * self.class_weight[1]
            tn = tn * self.class_weight[0]
            fp = fp * self.class_weight[0]
            fn = fn * self.class_weight[1]

            weighted_accuracy = (tp + tn) / (tp + tn + fp + fn)
            weighted_f1 = (2*tp) / (2*tp + fn + fp)
        else:
            accuracy = 0
            weighted_accuracy = 0
            f1 = 0
            weighted_f1 = 0
        
        # Sets self.accuracy to the accuracy statistic in evaluation_method
        self.test_accuracy_dict = {'accuracy':float(accuracy), 'weighted_accuracy':float(weighted_accuracy), 'f1':float(f1), 'weighted_f1':float(weighted_f1)}

    def __get_threshold_space(self, threshold: 'Threshold') -> list[int]:
        """
        Returns a list of data_df indices that fall within the given threshold
        :threshold: the threshold you want to compare to
        """
        # Create a list[bool] for whether or not each index is within the threshold
        column_of_interest = self.data_df.loc[:,threshold.index]
        if(threshold.operator == '<'):
            mask = column_of_interest < threshold.cut_value
        elif(threshold.operator == '>'):
            mask = column_of_interest > threshold.cut_value
        else:
            mask = [True for i in self.data_df.index]

        y_indices_inside = self.data_df.index[mask].tolist()
        return y_indices_inside
    
    def get_hotspot_space(self) -> list[int]:
        """Returns a list of y indices that fall within the hotspot"""
        y_index_list = []
        for threshold in self.thresholds:
            y_index_list.append(self.__get_threshold_space(threshold))
        y_index_intersection = set(y_index_list[0]).intersection(*y_index_list) # Gets the common items in the y_index_list
        return list(y_index_intersection)
      
    def expand(self, virtual_data_df:pd.DataFrame) -> pd.DataFrame:
        """
        Given a new parameters dataframe, returns a dataframe showing which lines are inside which thresholds

        :virtual_data_df: an expanded dataframe with rows to be sorted in or out of the hotspot
        """
        bool_list = [self.__is_inside(i, virtual_data_df) for i in virtual_data_df.index]
        threshold_evaluations = pd.DataFrame(bool_list, index=virtual_data_df.index, columns=self.threshold_indexes)
        return threshold_evaluations
    
    def get_external_accuracy(self, virtual_data_df:pd.DataFrame, verbose:bool=False, low_is_good:bool=False) -> tuple[float, float, float]:
        """
        Given a new parameters dataframe with experimental results,
        returns the accuracy, precision, and recall of the hotspot on that dataframe
        
        :virtual_data_df: a dataframe with experimental response in the 'response' column and parameters in the other columns
        """

        tp,tn,fp,fn = 0,0,0,0 # True Positive, True Negative, False Positive, False Negative

        # Sort the ligands from virtual_data_df into the confusion matrix
        for ligand in virtual_data_df.index:
            if (virtual_data_df.loc[ligand, 'response'] >= self.y_cut):
                if(all(self.__is_inside(ligand, virtual_data_df))):
                    tp = tp + 1
                else:
                    fn = fn + 1
            if (virtual_data_df.loc[ligand, 'response'] < self.y_cut):
                if(all(self.__is_inside(ligand, virtual_data_df))):
                    fp = fp + 1
                else:
                    tn = tn + 1

        # If low_is_good, invert the confusion matrix
        if low_is_good:
            tp, tn = tn, tp
            fp, fn = fn, fp
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if verbose:
            print(f'Accuracy: {accuracy:.2f}')
            print(f'Precision: {precision:.2f}')
            print(f'Recall: {recall:.2f}')

        return accuracy, precision, recall

    def print_stats(self):
        """Prints a handful of relevant stats about the hotspot"""
        a_accuracy = self.accuracy_dict['accuracy']
        a_weighted_accuracy = self.accuracy_dict['weighted_accuracy']
        a_f1 = self.accuracy_dict['f1']
        a_weighted_f1 = self.accuracy_dict['weighted_f1']
        a_precision = self.accuracy_dict['precision']
        a_recall = self.accuracy_dict['recall']

        tr_accuracy = self.train_accuracy_dict['accuracy']
        tr_weighted_accuracy = self.train_accuracy_dict['weighted_accuracy']
        tr_f1 = self.train_accuracy_dict['f1']
        tr_weighted_f1 = self.train_accuracy_dict['weighted_f1']
        
        te_accuracy = self.test_accuracy_dict['accuracy']
        te_weighted_accuracy = self.test_accuracy_dict['weighted_accuracy']
        te_f1 = self.test_accuracy_dict['f1']
        te_weighted_f1 = self.test_accuracy_dict['weighted_f1']
        
        print('                    all    train    test')
        print(f'         Accuracy: {a_accuracy:.3f}   {tr_accuracy:.3f}   {te_accuracy:.3f}')
        print(f'Weighted Accuracy: {a_weighted_accuracy:.3f}   {tr_weighted_accuracy:.3f}   {te_weighted_accuracy:.3f}')
        print(f'               F1: {a_f1:.3f}   {tr_f1:.3f}   {te_f1:.3f}')
        print(f'      Weighted F1: {a_weighted_f1:.3f}   {tr_weighted_f1:.3f}   {te_weighted_f1:.3f}\n')
        print(f'        Precision: {a_precision:.3f}')
        print(f'           Recall: {a_recall:.3f}')