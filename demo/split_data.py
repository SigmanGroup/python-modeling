from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import kennardstonealgorithm as ks 
#https://pypi.org/project/kennard-stone/????

class split_data:
    
    def __init__(self, df: pd.DataFrame = None):
        self.except_message = "ERROR: "
        # the numbers in the variables VS and TS refer to the original 0-indexed sample numbers 
        self.X = None
        self.Y = None
        self.X_train = None 
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def random_split(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state = 1)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def quarter_split(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state = 1)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def define(self, X, y, TS = None, VS = None):
        if VS is None:
            self.VS = [i for i in range(X.shape[0])]       

        if TS is None:
            self.TS = [i-1 for i in VS] 
                       
        self.X_train, self.X_test, self.y_train, self.y_test = X[TS],y[TS],X[VS],y[VS]
        return self.X_train, self.X_test, self.y_train, self.y_test
        
#     def kennardstone(self, X, y, test_ratio = 0.3):
#         TS, VS = ks.kennardstonealgorithm(X,int((1-test_ratio)*np.shape(X_sel)[0]))
#         self.X_train, self.X_test, self.y_train, self.y_test = X_sel[TS], y_sel[TS],X_sel[VS], y_sel[VS]
#         self.TS = [np.argwhere(np.all(X==i,axis=1))[0,0] for i in X_train]
#         self.VS = [np.argwhere(np.all(X==i,axis=1))[0,0] for i in X_test]  
#         return self.X_train, self.X_test, self.y_train, self.y_test
                       
#     def y_equidist(self, X, y, extrapolation = False):
#         if extrapolation:
#             VS_,TS_ = ks.kennardstonealgorithm(y_sel.reshape(np.shape(y)[0],1),int((test_ratio)*np.shape(y)[0]))
#             self.X_train, self.X_test, self.y_train, self.y_test = X[TS_], y[TS_], X[VS_], y[VS_]
#             self.TS = [np.argwhere(np.all(X==i,axis=1))[0,0] for i in X_train]
#             self.VS = [np.argwhere(np.all(X==i,axis=1))[0,0] for i in X_test]
#             return self.X_train, self.X_test, self.y_train, self.y_test

#         else:
#             minmax = [np.argmin(y_sel),np.argmax(y_sel)]
#             y_ks = np.array(([i for i in y if i not in [np.min(y),np.max(y)]]))
#             y_ks_indices = [i for i in range(len(y)) if i not in minmax]
#             # indices relative to y_ks:
#             VS_ks,TS_ks = ks.kennardstonealgorithm(y_ks.reshape(np.shape(y_ks)[0],1),int((test_ratio)*(2+np.shape(y_ks)[0])))
#             # indices relative to y_sel:
#             self.TS = sorted([y_ks_indices[i] for i in list(TS_ks)]+minmax)
#             self.VS = sorted([y_ks_indices[i] for i in VS_ks])
#             print("TS and VS defined.")
#             return
                       
    def none(self, X, y):
        self.TS, self.VS = [i for i in range(X.shape[0]) if i not in exclude],[]
        self.X_train, self.X_test, self.y_train, self.y_test = X[TS],y[TS],X[VS],y[VS]
        return self.X_train, self.X_test, self.y_train, self.y_test
                       
    def view_split_results(self):
        print("y_mean: {:.3f}".format(np.mean(self.y_train)))
        print("y_mean: {:.3f}".format(np.mean(self.y_test)))
        print("Shape X_train: {}".format(self.X_train.shape))
        print("Shape X_test:  {}".format(self.X_test.shape))   