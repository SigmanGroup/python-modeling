import pandas as pd
import numpy as np


class single_file:
    
    def __init__(self):
        self.except_message = "Error:"
        self.df = None #to be data frame
    
    """
    This is a simplified version of the read single file function (see the original implemented as a method below)
    It includes parameters for the header row, span as index range for the rows to be included, 
    in addition to the file path and sheet name.
    The method also checks for a valid path before trying to read the excel file.
    """
    def read_excel_file(self, path="", sheet=0, header_row=0, rows_start = None, rows_stop = None, cols_start = None, cols_stop = None):
        #read file and create pandas DataFrame
        try:
            self.df = pd.read_excel(io=path, sheet_name=sheet, header=header_row)
            self.df.dropna(axis = 0, how = 'all', inplace = True)
            self.__get_subset_of_df(rows_start, rows_stop, cols_start, cols_stop)
        #error message if file cannot be found
        except IOError:
            print (self.except_message,"Could not read file:", path)

    """
    This is the single file code from the original Matlab_modeling notebook as a method. 
    The default settings are also matched with the original file, given in as method parameters.
    """
    def read_single_file(self, excel_file="../model_oct_bothsubs", excel_sheet="bubbleplot", header_row=4, num_par=2, par_start_col=9, num_samples=24, response_col=5, y_label_col = 0, apply_mask=True, verbose=True, xlabelrow=True):
        inp = pd.read_excel(excel_file + ".xlsx", excel_sheet, header=header_row, index_col=y_label_col, nrows=num_samples + int(xlabelrow), usecols=list(range(0, (num_par + par_start_col))))
        print(inp.head())
        print()

        if xlabelrow:
            X_names = list(inp.iloc[0, par_start_col - 1:num_par + par_start_col - 1])
            X_labels = list(inp.columns)[par_start_col - 1:num_par + par_start_col - 1]
            resp_label = list(inp.columns)[response_col - 1]
            inp.drop(index=inp.index[0], inplace=True)
        else:
            X_labels = list(inp.columns)[par_start_col - 1:num_par + par_start_col - 1]
            X_names = X_labels
            resp_label = list(inp.columns)[response_col - 1]

        X_labelname = [" ".join(i) for i in zip(X_labels, X_names)]
        X_labelname_dict = dict(zip(X_labels, X_names))
        y = np.asarray(inp[resp_label], dtype=float)
        X = np.asarray(inp[X_labels], dtype=float)
        y_labels = np.asarray(list(inp.index), dtype=str)
        y_labels_comp = y_labels

        if apply_mask:
            mask = y.nonzero()[0]
            mask = ~np.isnan(y)
            print("n_samples before removing empty cells: {}".format(len(y)))
            print("Removing {} samples.".format(len(y) - sum(mask)))
            X = X[np.array(mask)]
            y = y[np.array(mask)]
            y_labels = y_labels[np.array(mask)]
        X_all = X
        if verbose:
            print("Shape X: {}".format(X.shape))
            print("Shape y: {}".format(y.shape))
            print("Shape labels: {}".format(y_labels.shape))
            print("First X cell: {}".format(X[0, 0]))
            print("Last X cell:  {}".format(X[-1, -1]))
            print("First y: {}".format(y[0]))
            print("Last y:  {}".format(y[-1]))
            print("Last label: {}".format(y_labels[-1]))
            
    """
    Private helper method gets specific subsection of dataframe
    """
    def __get_subset_of_df(self, rows_start=None, rows_stop=None, cols_start=None, cols_stop=None):
            #check parameters
            if (rows_stop) is None:
                rows_stop=self.df.shape[0]-1
                
            if cols_stop is None:
                cols_end=self.df.shape[1]-1
            
            if rows_start is None:
                rows_start = 0
                
            if cols_start is None:
                cols_start = 0
                
            try: 
                self.df = self.df.iloc[rows_start:rows_stop, cols_start:cols_stop]
                
            except IOError:
                print (self.except_message,"Span does not fit dataframe dimensions.")

#todo
class multiple_files():
    def __init__(self):
        self.except_message = "Error:"
        self.df = None #to be data frame
        
        