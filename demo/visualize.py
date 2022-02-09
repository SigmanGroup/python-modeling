import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


class visualize:
    
    def __init__(self, df: pd.DataFrame = None):
        self.except_message = "Error:"
        self.df = df.copy()
        self.sub_df = self.df.copy()
        self.y_dict = dict()
        self.x_dict = dict()
        self.rsq_threshold = 0

    def reset_df(self):
        self.sub_df = self.df.copy()

    '''
    This method creates a dictionary that maps feature names to their corresponding indices.
    '''
    def __get_feature_dict(self):
        print("Unfinished")

    '''
    This method sets an r^2 threshold so the only plots within that threshold are included.
    '''
    def set_rsq_threshold(self, threshold: float = 0.0):
         self.rsq_threshold = threshold

    '''
    This method allows a user to plot only a certain features by providing the feature names in a list.This method takes a list of feature names, not indices.
    '''  
    def inlude_features(self, feature_list: list = None):
        try:
            self.sub_df = self.sub_df[self.sub_df[feature_list]]
        except:
            print(self.except_message + "feature in list not in dataframe")

    '''
    This method allows a user to plot only all features except the feature names in a given list. This method takes a list of feature names, not indices.
    '''  
    def exclude_features(self, feature_list: list = None):
        try:
            self.sub_df = self.sub_df.drop(feature_list, axis=1)
        except:
            print(self.except_message + "feature in list not in dataframe")

    '''
    This method allows a user to plot a histogram for one, multiple or all columns in the dataframe. 
    If a column is not specified, all of the columns will be plotted. 
    '''
    def build_hist(self, cols: list = None):
        try:
            if cols is None:
                cols = self.sub_df.columns

            for col in cols:
                plt.hist(self.sub_df[col], bins="auto")
                plt.ylabel("frequency")
                plt.xlabel(col)
                plt.show() 
        except:
            print(self.except_message + "feature in list not in dataframe")

    '''
    This method allows a user to plot a univariate linear regression line for one, multiple or all columns in the dataframe. 
    If a column is not specified, all of the columns will be plotted. 
    If a y column is not specified, then it will be a zero vector.
    '''
    def build_univar(self, cols: list = None, y: list = None):
        try:
            if cols is None:
                cols = self.sub_df.columns
            if y is None:
                y = np.zeros(self.sub_df.shape[0])

            for col in cols:
                slope, intercept, r_value, p_value, std_err = stats.linregress(self.sub_df[col], y)
                #don't plot results with an r_value < 0
                if self.rsq_threshold > 0:
                    continue

                fit_line = intercept+slope*self.sub_df[col]

                plt.scatter(self.sub_df[col], y,color="black",marker="s",alpha=0.5)    
                plt.plot(self.sub_df[col],fit_line,color="black")
                plt.xlabel(col)
                plt.ylabel("y") # "$ΔΔG^{≠}$"  "Yield"
                plt.yticks()        
                plt.tight_layout()
                plt.show()    
        except:
            print(self.except_message + "feature in list not in dataframe")


#                            
#         def show_hist_univar_pairs(self):
#             for f_ind in features:
#                 feature = X_labels[f_ind]
#                 print(feature, X_names[f_ind])
#                 slope, intercept, r_value, p_value, std_err = stats.linregress(X[:,f_ind], y)
#                 fit_line = intercept+slope*X[:,f_ind]

#                 plt.figure(figsize=(9, 4))

#                 plt.subplot(1,2,1)
#                 plt.hist(X[:,f_ind], bins="auto")
#                 plt.ylabel("frequency",fontsize=20)
#                 plt.xlabel(feature + " " + X_names[f_ind],fontsize=20)

#                 plt.subplot(1,2,2)
#                 plt.scatter(X[:,f_ind], y,color="black",marker="s",alpha=0.5)    
#                 plt.plot(X[:,f_ind],fit_line,color="black")
#                 plt.xlabel(feature + " " + X_names[f_ind],fontsize=20)
#                 plt.ylabel("y",fontsize=20) # "$ΔΔG^{≠}$"  "Yield"

#                 plt.yticks(fontsize=15)        
#                 plt.tight_layout()
#                 plt.show()    

#                 if p_value > 0.01:
#                 print("R^2 = {:.2f}; p-value = {:.2f}".format(r_value**2,p_value))
#                 else:
#                 print("R^2 = {:.2f}; p-value = {:.2E}".format(r_value**2,p_value))
#                 print("\n-------------------------------------------------------------------------------\n")
