import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import itertools
import warnings
warnings.filterwarnings("ignore")
from sklearn import tree
import kennardstonealgorithm as ks

        
'''
This is a class that is used by the my_dataframe class to dynamically store the split X & Y data, as well as train/test splits
'''
class split_df():
    self.X = None
    self.Y = None
    self.X_train = None
    self.Y_train = None
    self.X_test = None
    self.Y_test = None
    
    def __init__(self, X, Y):
        self.except_message = "Error:" 
        self.set_X(X)
        self.set_Y(Y)
    
    def set_X(self, X):
        self.X = X.copy()
        
    def set_Y(self, Y):
        self.Y = Y.copy()
        
    def set_X_train(self, X_train):
        self.X_train = X_train
    
    def set_Y_train(self, Y_train):
        self.Y_train = Y_train
        
    def set_X_test(self, X_test):
        self.X_test = X_test
        
    def set_Y_test(self, Y_test):
        self.Y_test = Y_test
        
    def get_train_set(self):
        return self.X_train, self.Y_train
    
    def get_test_set(self):
        return self.X_test, self.Y_test

'''
    This class can be used to save pandas representations of excel files. 
    
    TODO: Access split_df attributes
'''
class my_dataframe():
    
    self.data = None
    self.processed_dataframe = None
    self.split_dataframe = None
    
    self.except_message = "Error:"
    self.welcome_message = "You just created a new my_dataframe object! \nTo add data to this new object, you can pass in an excel file or a Pandas dataframe you already have. \nTo pass in your own Pandas DataFrame, use the set_my_dataframe method. It takes a single parameter, which is your Pandas dataframe :) \nTo read an excel file, you may use read_excel_file or read_two_excel_files. If you have your X and Y values in different excel files, you will want to use read_two_excel_files. Otherwise, you can use read_excel_file. 
#     self.help_message = "To learn more about how a function works, you can use the help function. Simply pass in the name of the function you're interested in."
#     self.list_files_message = "To view the list of my_dataframe methods, call the method list_functions."

    def __init__(self):
        print(self.welcome_message)


    def reset_data(self):
    '''
        Resets the data within the class to its original form (before preprossesing).
    '''        
        self.processed_dataframe = self.data
        

    def set_my_dataframe(self, pd_df):
    '''
        This function allows the user to pass in their own Pandas dataframe to use the my_dataframe class.
    '''        
        if(type(pd_df) is pd.DataFrame):
            self.data = pd_df.copy()
        else:
            print(self.except_message, "you did not pass in a valid Pandas dataframe.")
    
    def read_excel_file(self, path="", sheet=0, header_row=0, rows_start = None, rows_stop = None, cols_start = None, cols_stop = None):
    '''
    This is a simplified version of the read single file function.
    It includes parameters for the header row, span as index range for the rows to be included, 
    in addition to the file path and sheet name.
    The method also checks for a valid path before trying to read the excel file.
    '''
        #read file and create pandas DataFrame
        try:
            self.data = pd.read_excel(io=path, sheet_name=sheet, header=header_row)
            self.data.dropna(axis = 0, how = 'all', inplace = True)
            self.__get_subset_of_df(rows_start, rows_stop, cols_start, cols_stop)
        #error message if file cannot be found
        except IOError:
            print (self.excepts_message,"Could not read file:", path)
            
    def read_two_excel_files(self, pathX="", pathY="", Y_col: int = None, sheet=0, header_row=0, rows_start = None, rows_stop = None, cols_start = None, cols_stop = None):
    """
    This is a simplified function of the read two files, one for x and one for y.
    """        
        #read file and create pandas DataFrame
        try:
            self.data = pd.read_excel(io=pathX, sheet_name=sheet, header=header_row)
            self.data.dropna(axis = 0, how = 'all', inplace = True)
            self.Y = pd.read_excel(io=pathY, sheet_name=sheet, header=header_row)
            self.Y.dropna(axis = 0, how = 'all', inplace = True)
            self.__get_subset_of_df(rows_start, rows_stop, cols_start, cols_stop)

        #error message if file cannot be found
        except IOError:
            print (self.except_message,"Could not read file:", path)
    

    def get_original_dataframe(self):
    '''
        This returns a copy of the original dataframe so the user can do whatever pandas magic they want with it, wi
    '''
        return self.data.copy()    
    
    """
    Private helper method gets specific subsection of dataframe
    """
    def __get_subset_of_df(self, rows_start=None, rows_stop=None, cols_start=None, cols_stop=None):
            #check parameters
            if (rows_stop) is None:
                rows_stop=self.data.shape[0]-1
                
            if cols_stop is None:
                cols_end=self.data.shape[1]-1
            
            if rows_start is None:
                rows_start = 0
                
            if cols_start is None:
                cols_start = 0
                
            try: 
                self.df = self.data.iloc[rows_start:rows_stop, cols_start:cols_stop]
                
            except IOError:
                print (self.except_message,"Span does not fit dataframe dimensions.")        
                
    '''
    This is a conventience method for removing a column after reading the file.
    You can also do this on your own with lists of column names, etc, using the Pandas library.
    To do this with a dataframe generated with this class, call get_dataframe().
    '''           
    def drop_column(self, col = None):
        if col is not None:
            try:
                self.data.drop(col_name, axis=1, inplace=True)
                self.data.reset_index(inplace = True)
                print("Column successfully removed from the dataframe.")
                return
            except:
                print (self.except_message,"The column you specified is not in the dataframe")
        else:
            print("You did not specify a column, so nothing in the dataframe was changed.")
            
                
    '''
    This is a conventience method for removing a row after reading the file.
    '''   
    def drop_row(self, row = None):
        if row is not None:
            try:
                self.data.drop(row, axis=0, inplace=True)
                self.data.reset_index(inplace = True)
                print("Row successfully removed from the dataframe.")
                return
            except:
                print (self.except_message,"The column you specified is not in the dataframe")
        else:
             print("You did not specify a column, so nothing in the dataframe was changed.")
                
    '''
        This is a a conventience method that drops column(s) from the data using the name of the column
    '''
    def drop_cols_names(self, names: list() = None):
        try:
            self.x = self.x.drop(names, axis=1)
        except IOError:
            print(self.except_message, "cannot exclude columns with feature names in list.")
            
    '''
        This is a a conventience method that drops column(s) from the data using the index number of the column
    '''                
    def drop_cols_indices(self, indices: list() = None, cutoff: list() = None):
        try:
            self.x = self.x.drop(self.x.columns[indices], axis=1)
        except IOError:
            print(self.except_message, "cannot exclude columns with indices in list.")
            
    '''
        This is a a conventience method that drops rows(s) from the data using a numeric cutoff (int)
    '''     
    def drop_rows_above_cutoff(self, col_name: str = None, cutoff: int = None):
        if cutoff is None:
            print("Error: cutoff not provided")
            return
        try:
            self.x = self.x[self.x[col_name] <= cutoff]
        except IOError:
            print(self.except_message, "unable to remove rows using the provided parameters.")
                
    '''
        Old method to be changed into two methods.
        TODO: CHANGE TO SET_X & SET_Y (takes col names as params)
    '''
    def data_prep(self, X_labels = None, Y_labels = None):
        #original data
        self.X = self.data[self.data[X_labels]]
        self.Y = self.data[self.data[Y_labels]]

        #preselected data
        self.y = self.__copy_data(Y)
        self.x = self.__copy_data(X)
    
    '''
        Private helper method that checks to see if there is data available in the class
    '''
    def __data_provided(self, data):
        if data is None and self.data is None:
            print (self.except_message,"Dataset not provided.")
            return False
        return True
    
    '''
        Private helper method. Creates a copy of the class data, if possible. 
    '''
    def __copy_data(self, data):
        if self.__data_provided(data):
            return data.copy()
        else:
            return None
    
    '''
        This method prints the shape of the data in the class
    '''
    def print_dataframe_shape(self):
        print("Shape X: {}".format(self.x.shape))
        print("Shape y: {}".format(self.y.shape)) 
        print("Shape labels: {}".format(self.y_labels.shape)) 
        
    '''
        This method scales X using skitlern
    '''
    def scaleX(self, X):
        return StandardScaler().fit_transform(X)
    
    '''
        Set Y to exp(Y)
    '''
    def get_y_exp(self):
        self.y = np.exp(self.y)
    
    '''
        Set Y to abs(Y)
    '''
    def get_y_abs(self):
        self.y = np.abs(self.y)
        
    '''
        Set Y to log(Y). Default is to add a small value to 0-valued Y variables. 
        True can be passed in as the first parameter, which is the same as remove_y_zeros = True. This option removes instances of zero. 
    '''
    def get_y_log(self, remove_y_zeros = False):
        if not remove_y_zeros:
            self.y = np.log(self.y+0.0001)
        else:
            self.y = self.y.loc[(self.y!=0).any(axis=1)]

    '''
        Create training subsets for X and Y via random split
        TODO: create reset methods for training/test subsets in class
    '''
    def random_split(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state = 1)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    '''
        Create training subsets for X and Y via quarter split.
        TODO: create reset methods for training/test subsets in class
    '''
    def quarter_split(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state = 1)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    '''
        Create training subsets for X and Y via user-provided indices.
        TODO: create reset methods for training/test subsets in class
    '''
    def define(self, X, y, TS = None, VS = None):
        if VS is None:
            self.VS = [i for i in range(X.shape[0])]       

        if TS is None:
            self.TS = [i-1 for i in VS] 
                       
        self.X_train, self.X_test, self.y_train, self.y_test = X[TS],y[TS],X[VS],y[VS]
        return self.X_train, self.X_test, self.y_train, self.y_test
        
'''
TODO
'''
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
            
    '''
        Create "split", but really just two copies of the data.
        TODO: REFACTOR
    '''
    def none(self, X, y):
        self.TS, self.VS = [i for i in range(X.shape[0]) if i not in exclude],[]
        self.X_train, self.X_test, self.y_train, self.y_test = X[TS],y[TS],X[VS],y[VS]
        return self.X_train, self.X_test, self.y_train, self.y_test
         
    '''
        Prints the result of the split. 
    '''
    def view_split_results(self):
        print("y_mean: {:.3f}".format(np.mean(self.y_train)))
        print("y_mean: {:.3f}".format(np.mean(self.y_test)))
        print("Shape X_train: {}".format(self.X_train.shape))
        print("Shape X_test:  {}".format(self.X_test.shape))   
        
        
class visualize:
    
    def __init__(self, df: pd.DataFrame = None):
        self.except_message = "ERROR: "
        self.df = df.copy()
        self.sub_df = self.df.copy()
        self.curr_cols = list(self.sub_df.columns)
        self.rsq_threshold = None

    def set_rsq_threshold(self, threshold: float = 0.0):
        """
        This method sets an r^2 threshold so the only plots within that threshold are included.
        """
        self.rsq_threshold = threshold
        
    def reset_rsq_threshold(self):
        self.rsq_threshold = None
        
    def show_hist_univar_pairs(self, cols, y):
        '''
        This method allows the hist and univar plots to be viewed side-by-side (like the original script)
        '''
        for col in cols:
            self.build_hist([col], y, True)            


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
            try:
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
            except:
                print("Failed to plot for col:" + str(col))
                continue


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
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(self.sub_df[col], y)
                result_label = self.__get_result_label(r_value, p_value)

                #don't plot results with an r_value < 0 or an r_value below the threshold
                if self.rsq_threshold is not None and self.rsq_threshold > r_value:
                    print("r^2 for ", col, "was below the threshold.")
                    continue

                #plot 
                fit_line = intercept+slope*self.sub_df[col]
                plt.scatter(self.sub_df[col], y,color="black",marker="s",alpha=0.5)    
                plt.plot(self.sub_df[col],fit_line,color="black")
                plt.xlabel(col)
                plt.ylabel("y") # "$ΔΔG^{≠}$"  "Yield"

                #handle titles
                if with_hist:
                    plt.suptitle("\n x = "+col+"\n"+result_label, y = 1)
                else:
                    plt.title(result_label)
                plt.show()
            except:
                print("Couldn't plot for col:" + str(col))
                continue
                

    def build_bubble_plot(self, features: list = None, bp_x: list = None, bp_y: list = None, bp_size: list = None, layer: bool = False):
        '''
        Bubble plot, more than one feature can be plotted
        '''  
        #validate features
        features = self.__validate_columns(features)
        if features is None:
            print(self.except_message+"There are columns in your list that do not exist in the dataframe.")
            return
        
        if bp_x is None:
            bp_x = np.zeros(self.sub_df.shape[0])        
        
        if bp_y is None:
            bp_y = np.zeros(self.sub_df.shape[0])   
            
        #plot
        plt.figure()
        palette = itertools.cycle(sns.color_palette())
        for feature in features:
            sns.scatterplot(data=self.sub_df[feature], x=bp_x, y=bp_y, size=bp_size, sizes=(50, 300), alpha = 0.5, color=next(palette), marker="D")            
            plt.tight_layout()
            plt.legend(bbox_to_anchor=(1.01,1),loc=2,borderaxespad=0)
            plt.title(feature)
            plt.show()
            
       
    def build_bubble_plot_virtual(self, features: list = None, bp_x: list = None, bp_y: list = None, bp_size: list = None, layer: bool = False):
        '''
        Bubble plot that uses "Virtual" feature
        '''         
        #validate features
        features = self.__validate_columns(features)
        if features is None:
            print(self.except_message+"There are columns in your list that do not exist in the dataframe.")
            return

        if bp_x is None:
            bp_x = np.zeros(self.sub_df.shape[0])        

        if bp_y is None:
            bp_y = np.zeros(self.sub_df.shape[0]) 
            
        #plot
        plt.figure()
        palette = itertools.cycle(sns.color_palette())
        for feature in features:
            sns.scatterplot(data=self.sub_df[feature], x=bp_x, y=bp_y, size=bp_size, sizes=(50, 300), alpha = 0.5, hue=self.sub_df['virtual'], style=self.sub_df['virtual'], color=next(palette), marker="D")            
            plt.tight_layout()
            plt.legend(bbox_to_anchor=(1.01,1),loc=2,borderaxespad=0)
            plt.title(feature + "\n parm v parm sized by y")
            plt.show()
            
    def correlation_map(self, features: list = None):
        #validate features
        features = self.__validate_columns(features)
        
        if features is None:
            print(self.except_message+"There are columns in your list that do not exist in the dataframe.")
            return        
        
        X = self.sub_df[features].to_numpy()
        corrmap = np.corrcoef(X.T)
        sns.heatmap(corrmap,center=0, annot=False, cmap="coolwarm", cbar=True)
        plt.xticks(range(len(features)),features, fontsize=10, rotation=90)
        plt.yticks(range(len(features)),features, fontsize=10)
        plt.show()
            
    def include_features(self, feature_list: list = None):
        """
        This method allows a user to plot only a certain features by providing the feature names in a list.
        This method takes a list of feature names, not indices.
        """  
        try:
            self.sub_df = self.sub_df[feature_list]
            self.curr_cols = list(self.sub_df.columns)
            print("columns in"+ str(feature_list) +" will be the only accessible features for visualizations.")
        except:
            print(self.except_message + "a feature in the list is not in your dataframe")


    def exclude_features(self, feature_list: list = None):
        """
        This method allows a user to plot only all features except the feature names in a given list. 
        This method takes a list of feature names, not indices.
        """  
        try:
            self.sub_df = self.sub_df.drop(feature_list, axis=1)
            self.curr_cols = list(self.sub_df.columns)
            print("columns "+ str(feature_list) +" were successfully deleted from the dataframe.")
        except:
            print(self.except_message + "feature in list will not be included in visualizations.")
            
    def show_feature_index(self):
        """
        This method shows the feature names and their corresponding indices.
        """
        for (i, item) in enumerate(self.cur_cols):
            print(i, item)   
            
    def reset_df(self):
        """
        This method returns the dataframe to it's original state (in case you have some regrets...)
        Consider this your control-z, but...like...everything
        """
        self.sub_df = self.df.copy()
        self.curr_cols = list(self.sub_df.columns)
        print("All features in your original dataframe can be included in visualizations.")

    def __get_result_label(self, rsq, pval):
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