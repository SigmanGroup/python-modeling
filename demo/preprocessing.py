import pandas as pd
import numpy as np


class data_prep():
    '''TODO: change param to dataframe and allow option to specify y column'''
    def __init__(self, X = None, Y = None, X_labels = None, Y_labels = None):
        self.except_message = "Error:"
        
        #original data
        self.X = X
        self.Y = Y
        self.X_labels = X_labels
        self.Y_labels = Y_labels
        
        #preselected data
        self.y = self.__copy_data(Y)
        self.x = self.__copy_data(X)
        self.x_labels = self.__copy_data(X_labels)
        self.y_labels = self.__copy_data(Y_labels)
        
    def __data_provided(self, data):
        if data is None:
            print (self.except_message,"Dataset not provided.")
            return False
        return True
    
    def __copy_data(self, data):
        if self.__data_provided(data):
            return data.copy()
        else:
            return None
    
    def reset_data(self):
        self.y = self.Y
        self.x = self.X
        self.y_labels = self.Y_labels
    
    def print_result(self):
        print("Shape X: {}".format(self.x.shape))
        print("Shape y: {}".format(self.y.shape)) 
        print("Shape labels: {}".format(self.y_labels.shape)) 
        
    def get_curr_data(self):
        return self.x.copy(), self.y.copy()
    
    def get_y_exp(self):
        self.y = np.exp(self.y)
    
    def get_y_abs(self):
        self.y = np.abs(self.y)
        
    def get_y_log(self, remove_y_zeros = False):
        if not remove_y_zeros:
            self.y = np.log(self.y+0.0001)
        else:
            print("not yet implemented")
            '''
                todo: handle remove_y_zeros
                    y = np.log(y[y.nonzero()[0]])
                    y_labels_orig,X_orig = y_labels.copy(),X.copy()
                    y_labels = y_labels[y.nonzero()[0]]
                    X = X[y.nonzero()[0]]
            '''
    def drop_cols_names(self, names: list() = None):
        try:
            self.x = self.x.drop(names, axis=1)
        except IOError:
            print(self.except_message, "cannot exclude columns with feature names in list.")
                
    def drop_cols_indices(self, indices: list() = None, cutoff: list() = None):
        try:
            self.x = self.x.drop(self.x.columns[indices], axis=1)
        except IOError:
            print(self.except_message, "cannot exclude columns with indices in list.")
            
    def drop_rows_above_cutoff(self, col_name: str = None, cutoff: int = None):
        if cutoff is None:
            print("Error: cutoff not provided")
            return
        try:
            self.x = self.x[self.x[col_name] <= cutoff]
        except IOError:
            print(self.except_message, "unable to remove rows using the provided parameters.")
              
    #Training/Test set split
    #Feature Scaling
    #Cross-terms/Interaction terms
