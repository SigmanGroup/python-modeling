import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


class visualize:
    
    def __init__(self, df: pd.DataFrame = None):
        self.except_message = "ERROR: "
        self.df = df.copy()
        self.sub_df = self.df.copy()
        self.curr_cols = list(self.sub_df.columns)
        self.y_dict = dict()
        self.x_dict = dict()
        self.rsq_threshold = None

    def reset_df(self):
        """
        This method returns the dataframe to it's original state (in case you have some regrets...)
        Consider this your control-z, but...like...everything
        """
        self.sub_df = self.df.copy()

    def __get_feature_dict(self):
        """
        This method creates a dictionary that maps feature names to their corresponding indices.
        """
        print("Unfinished")

    def set_rsq_threshold(self, threshold: float = 0.0):
        """
        This method sets an r^2 threshold so the only plots within that threshold are included.
        """
        self.rsq_threshold = threshold


    def inlude_features(self, feature_list: list = None):
        """
        This method allows a user to plot only a certain features by providing the feature names in a list.
        This method takes a list of feature names, not indices.
        """  
        try:
            self.sub_df = self.sub_df[self.sub_df[feature_list]]
        except:
            print(self.except_message + "feature in list not in dataframe")


    def exclude_features(self, feature_list: list = None):
        """
        This method allows a user to plot only all features except the feature names in a given list. 
        This method takes a list of feature names, not indices.
        """  
        try:
            self.sub_df = self.sub_df.drop(feature_list, axis=1)
            self.curr_cols = list(self.sub_df.columns)
        except:
            print(self.except_message + "feature in list not in dataframe")


    def build_hist(self, cols: list = None, y = None, univar: bool = False):
        """
        This method allows a user to plot a histogram for one, multiple or all columns in the dataframe. 
        If a column is not specified, all of the columns will be plotted. 
        """
        cols = self.__validate_columns(cols)
        if cols is None:
            print(self.except_message+"There are columns in your list that do not exist in the dataframe.")
            return
        
        fig = plt.figure()
        
        for col in cols:
            plt.subplot(1,2,1)
            plt.hist(self.sub_df[col], bins="auto")
            plt.ylabel("frequency")
            plt.xlabel(col)
            
            if univar:
                plt.tight_layout(pad=7.0)
                plt.subplot(1,2,2)
                self.build_univar(cols, y, True)
                
            else:
                plt.show()

    def get_result_label(self, rsq, pval):
        """
        This method gets a label for the regression plots using r^2 and pvalues 
        """
        if pval > 0.01:
            return "R^2 = {:.2f}; p-value = {:.2f}".format(rsq**2,pval)
        return "R^2 = {:.2f}; p-value = {:.2E}".format(rsq**2,pval)
    
    def __validate_columns(self, cols):
        """
        This method checks to make sure that the columns specified for the plot exist.
        """
        if cols is None:
            return self.curr_cols
        elif all(col in self.curr_cols for col in cols):
            return cols
        else:
            return None

    def build_univar(self, cols: list = None, y: list = None, with_hist: bool = False):
        """
        This method allows a user to plot a univariate linear regression line for one, multiple or all columns in the dataframe. 
        If a column is not specified, all of the columns will be plotted. 
        If a y column is not specified, then it will be a zero vector.
        """
        cols = self.__validate_columns(cols)
        if cols is None:
            print(self.except_message+"There are columns in your list that do not exist in the dataframe.")
            return
        
        if y is None:
            y = np.zeros(self.sub_df.shape[0])

        for col in cols:
            slope, intercept, r_value, p_value, std_err = stats.linregress(self.sub_df[col], y)
            result_label = self.get_result_label(r_value, p_value)
            #don't plot results with an r_value < 0 or an r_value below the threshold
            if self.rsq_threshold is not None and self.rsq_threshold > r_value:
                print("r^2 for ", col, "was below the threshold.")
                continue
            fit_line = intercept+slope*self.sub_df[col]
            plt.scatter(self.sub_df[col], y,color="black",marker="s",alpha=0.5)    
            plt.plot(self.sub_df[col],fit_line,color="black")
            plt.xlabel(col)
            plt.ylabel("y") # "$ΔΔG^{≠}$"  "Yield"
            
            if with_hist:
                plt.suptitle("\n x = "+col+"\n"+result_label, y = 1)
            else:
                plt.title(result_label)
            plt.show()            
            
    def show_hist_univar_pairs(self, cols, y):
        for col in cols:
            self.build_hist([col], y, True)
# https://stackoverflow.com/questions/3783217/get-the-list-of-figures-in-matplotlib