import pandas as pd
import numpy as np


class data_prep():
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
        
            
    def preselect(self, exp: bool = False, remove_y_zeros: bool = False, log: bool = False, abs_val: bool = False, exclude_feature = None, cutoff: int = None, exclude_index: int = None):  
        y = self.y
        x = self.x
        y_labels = self.y_labels
        
        if exp and y is not None:
            y = np.exp(y)
        
        if abs_val and y is not None:
            y = np.abs(y)
        
        if log and y is not None and not remove_y_zeros: 
            y = np.log(y+0.0001)
            '''
                todo: handle remove_y_zeros
                    y = np.log(y[y.nonzero()[0]])
                    y_labels_orig,X_orig = y_labels.copy(),X.copy()
                    y_labels = y_labels[y.nonzero()[0]]
                    X = X[y.nonzero()[0]]
            '''
        if exclude_feature is not None and cutoff is not None:
            try:
                mask_prop = x[:,self.x_labels.index(select_feature)]<cutoff
                self.x = x[mask_prop]
                self.y = y[mask_prop]
                self.y_labels = self.y_labels[mask_prop]
            except IOError:
                print(self.except_message, "cannot exclude feature with specified cutoff.")
        
        if exclude_index is not None:
            #todo: exclude = [38] #+[i for i in range(26,37)]?
            try:
                mask_prop = [i for i in range(len(y)) if i not in exclude_index]
                self.x = x[mask_prop]
                self.y = y[mask_prop]
                self.y_labels = self.y_labels[mask_prop]
            except IOError:
                print(self.except_message, "cannot exclude specified index.")
        
    #Training/Test set split
    #Feature Scaling
    #Cross-terms/Interaction terms
