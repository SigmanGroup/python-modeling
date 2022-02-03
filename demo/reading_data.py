import pandas as pd
import numpy as np


class read_excel_file:
    
    def __init__(self):
        self.except_message = "Error:"
    
    """
    This is a simplified version of the read single file function.
    It includes parameters for the header row, span as index range for the rows to be included, 
    in addition to the file path and sheet name.
    The method also checks for a valid path before trying to read the excel file.
    """
    def read_excel_file(self, path="", sheet=0, header_row=0, rows_start = None, rows_stop = None, cols_start = None, cols_stop = None):
        #read file and create pandas DataFrame
        try:
            self.data = pd.read_excel(io=path, sheet_name=sheet, header=header_row)
            self.data.dropna(axis = 0, how = 'all', inplace = True)
            self.__get_subset_of_df(rows_start, rows_stop, cols_start, cols_stop)
        #error message if file cannot be found
        except IOError:
            print (self.except_message,"Could not read file:", path)
            
        """
    This is a simplified version of the read single file function.
    It includes parameters for the header row, span as index range for the rows to be included, 
    in addition to the file path and sheet name.
    The method also checks for a valid path before trying to read the excel file.
    """
    def read_two_excel_files(self, pathX="", pathY="", Y_col: int = None, sheet=0, header_row=0, rows_start = None, rows_stop = None, cols_start = None, cols_stop = None):
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
        