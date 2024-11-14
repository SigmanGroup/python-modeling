from __future__ import annotations
import copy
import random
from typing import Optional
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from hotspot_classes import Threshold, Hotspot

# This supresses warnings that arise in the plotting functions
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def threshold_generation(data_df:pd.DataFrame, class_weight:dict, evaluation_method:str, features:list[str]) -> list[Threshold]:
    """
    Given the master dataframe and some parameters, return the best threshold in each feature.

    :data_df: The master dataframe with x# column names and the first two columns as 'response' and 'y_class'
    :class_weight: Mapping of hits (1) and misses (0) to their respective class weights. Example: {1:10, 0:1}
    :evaluation_method: 'accuracy', 'weighted_accuracy', 'f1', 'weighted_f1'; Primary accuracy metric to be used in threshold comparison
    :features: List of x# parameter names to get thresholds for.  Primarily used for manual hotspot selection.
    """

    all_thresholds = []
    for feature in features:
        x = (data_df.loc[:,feature].values).reshape(-1, 1) # pulls the relevant parameter column and formats it in the propper array
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
            dt.tree_.threshold[0],
            operator, 
            feature_name = feature,
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

def hs_next_thresholds(hs:Hotspot, data_df:pd.DataFrame, class_weight:dict, features:list[str]) -> list[Hotspot]:
    """
    Given the master dataframe and some parameters, return the best threshold in each feature.

    :hs: Hotspot to add additional thresholds to
    :data_df: The master dataframe with x# column names and the first two columns as 'response' and 'y_class'
    :class_weight: Mapping of hits (1) and misses (0) to their respective class weights. Example: {1:10, 0:1}
    :x_labelname_dict: Dictionary for converting x# labels to full feature names
    :features: List of x# parameter names to get thresholds for.  Primarily used for manual hotspot selection.
    """

    # Makes all possible hotspots by adding one threshold
    all_hotspots = []
    for feature in features:
        x = (data_df.loc[:,feature].values).reshape(-1, 1) # pulls the relevant parameter column and formats it in the propper array
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
            dt.tree_.threshold[0],
            operator, 
            feature_name = feature,
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
        accuracy_list.append(hs.train_accuracy_dict[evaluation_method])

    cut = np.percentile(accuracy_list, 100 - percentage)
    
    hs_out=[]
    for hs in hotspots:
        if(hs.accuracy_dict[evaluation_method]>=cut):
            hs_out.append(hs)
    
    return hs_out

def plot_hotspot(hs:Hotspot,
                 test_response_data:Optional[pd.DataFrame] = None, vs_parameters:Optional[pd.DataFrame] = None,
                 subset:str = 'all', hide_training:bool = False,
                 coloring:str = 'scaled', gradient_color:str = 'Oranges', output_label:str = 'Yield (%)'):
    """
    Plot a single, double, or triple threshold by calling the relevant function.
    Plotting style (normal, test, or virtual screen) is determined by the presence of test_response_data and vs_parameters.

    :hs: Hotspot object to plot
    :test_response_data: DataFrame of test set response data (optional)
    :vs_parameters: DataFrame of virtual screening / test set parameters (optional)
    :subset: 'all', 'train', or 'test'; indicates which subset to show on the plot
    :hide_training: True or False; indicates if the training set should be hidden
    :coloring: 'scaled' or 'binary'; indicates if points should be colored based on actual output values or by output category
    :gradient_color: the color scheme applied to the heatmap, default 'Oranges'
    :output_label: default 'Yield (%)'
    """
    if(len(hs.thresholds)==1):
        plot_single_threshold(hs, test_response_data, vs_parameters, subset, hide_training, coloring, gradient_color, output_label)
    elif(len(hs.thresholds)==2):
        plot_double_threshold(hs, test_response_data, vs_parameters, subset, hide_training, coloring, gradient_color, output_label)
    elif(len(hs.thresholds)==3):
        plot_triple_threshold(hs, test_response_data, vs_parameters, subset, hide_training, coloring, gradient_color, output_label)
    else:
        print(f'Unable to plot {len(hs.thresholds)} thresholds')

def plot_single_threshold(hs: Hotspot,
                          test_response_data: Optional[pd.DataFrame] = None, vs_parameters: Optional[pd.DataFrame] = None,
                          subset: str = 'all', hide_training: bool = False,
                          coloring: str = 'scaled', gradient_color: str = 'Oranges', output_label: str = 'Yield (%)'):
    """
    Plot a single threshold in 2 dimensions

    :hs: Hotspot object to plot
    :test_response_data: DataFrame of test set response data (optional)
    :vs_parameters: DataFrame of virtual screening / test set parameters (optional)
    :subset: 'all', 'train', or 'validation'; indicates which subset to show on the plot
    :hide_training: True or False; indicates if the training set should be hidden
    :coloring: 'scaled' or 'binary'; indicates if points should be colored based on actual output values or by output category
    :gradient_color: the color scheme applied to the heatmap, default 'Oranges'
    :output_label: default 'Yield (%)'
    """

    # Set up flags for what kind of plotting is requested
    plot_test = test_response_data is not None and vs_parameters is not None
    plot_virtual_screening = test_response_data is None and vs_parameters is not None 

    x_col = hs.thresholds[0].feature_name
    fig, ax = plt.subplots(figsize=(10, 8))  # Create a figure and an axes

    # This section auto-scales the plot
    if plot_virtual_screening:
        x_values = list(chain(*[hs.data_df.loc[:, x_col], vs_parameters.loc[:, x_col]]))
    elif plot_test:
        x_values = list(chain(*[hs.data_df.loc[:, x_col], vs_parameters.loc[test_response_data.index, x_col]]))
    else:
        x_values = hs.data_df.loc[:, x_col]

    if plot_test:
        y_values = list(chain(*[hs.data_df.loc[:, output_label], test_response_data.iloc[:, 0]]))
    else:
        y_values = hs.data_df.loc[:, output_label]

    x_min = float(min(x_values))
    x_max = float(max(x_values))
    y_min = float(min(y_values))
    y_max = float(max(y_values))

    dx = abs(x_min - x_max)
    dy = abs(y_min - y_max)

    x_min = x_min - abs(dx * 0.05)
    x_max = x_max + abs(dx * 0.05)
    y_min = y_min - abs(dy * 0.05)
    y_max = y_max + abs(dy * 0.05)
    
    # Set which points to plot based on the subset parameter
    if(subset == 'all'):
        points_to_plot = hs.data_df.index
    elif(subset == 'train'):
        points_to_plot = hs.training_set
    elif(subset == 'validation'):
        points_to_plot = hs.validation_set
    else:
        raise ValueError('Subset must be "all", "train", or "validation"')
    
    # Change how the points are colored, controlled by the coloring parameter
    if(coloring=='scaled'):
        mapping_cl = hs.data_df.loc[points_to_plot, output_label]
        if(plot_test):
            test_mapping_cl = test_response_data.iloc[:, 0]
    elif(coloring=='binary'):
        mapping_cl = hs.data_df.loc[points_to_plot, 'y_class']
        if(plot_test):
            test_mapping_cl = [1 if i >= hs.y_cut else 0 for i in test_response_data.iloc[:, 0]]
    else:
        raise ValueError('Coloring must be either "scaled" or "binary"')

    # Plot the main dataset if not hiding it
    if not hide_training:
        x = hs.data_df.loc[points_to_plot, x_col]
        y = hs.data_df.loc[points_to_plot, output_label]
        if plot_test: alpha = 0.5
        else: alpha=1
        ax.scatter(x, y, c = mapping_cl, cmap = gradient_color, edgecolor ='black', alpha=alpha, s = 100, marker = 'o')

    # Plot the virtual screening set if only given parameters
    if(plot_virtual_screening):
        vs_x = vs_parameters.loc[:, x_col]
        ax.scatter(vs_x, [0 for i in range(len(vs_x))], c='grey', edgecolor='black', alpha=0.5, linewidth=2, s=100, marker='x')

    # Plot the test data set if given parameters and response
    if(plot_test):
        test_x = vs_parameters.loc[test_response_data.index, x_col]
        test_y = test_response_data.iloc[:, 0]
        ax.scatter(test_x, test_y, c = test_mapping_cl, cmap = gradient_color, edgecolor = 'black', linewidth=2, s = 100, marker = 's')
    
    # Set the gradient bar or binary legend
    if(coloring == 'scaled'):
        norm = Normalize(vmin=min(mapping_cl), vmax=max(mapping_cl))
        mappable = ScalarMappable(cmap=gradient_color, norm=norm)
        mappable.set_array([])
        
        cbar = plt.colorbar(mappable, ax=ax, shrink=1)
        cbar.set_label(output_label, rotation=90, size=25)

        # Define the legend symbols
        training_symbol = Line2D([0], [0], marker='o', color='w', label='Training', markerfacecolor='white', markersize=10, markeredgecolor='black')
        test_symbol = Line2D([0], [0], marker='s', color='w', label='Test', markerfacecolor='white', markersize=10, markeredgecolor='black')
        virtual_screen_symbol = Line2D([0], [0], marker='x', color='w', label='Virtual Screen', markerfacecolor='white', markersize=10, markeredgecolor='black')

        # Decide which symbols to include in the legend
        legend_symbols = []
        if ((plot_test or plot_virtual_screening) and not hide_training):
            legend_symbols.extend([training_symbol])
        if plot_test:
            legend_symbols.extend([test_symbol])
        if plot_virtual_screening:
            legend_symbols.append([virtual_screen_symbol])

        if plot_virtual_screening or plot_test:
            ax.legend(handles=legend_symbols, fontsize=15, loc='upper right', edgecolor='black')

    elif(coloring == 'binary'):
        # Define the legend colors
        colormap = plt.get_cmap(gradient_color)
        active_color = mcolors.to_hex(colormap(1.0))
        inactive_color = mcolors.to_hex(colormap(0.0))
        virtual_screen_color = mcolors.to_hex('grey')
 
        # Define the legend symbols
        active_symbol = Line2D([0], [0], marker='o', color='w', label='Active', markerfacecolor=active_color, markersize=10, markeredgecolor='black')
        inactive_symbol = Line2D([0], [0], marker='o', color='w', label='Inactive', markerfacecolor=inactive_color, markersize=10, markeredgecolor='black')
        active_test_symbol = Line2D([0], [0], marker='s', color='w', label='Active Test', markerfacecolor=active_color, markersize=10, markeredgecolor='black')
        inactive_test_symbol = Line2D([0], [0], marker='s', color='w', label='Inactive Test', markerfacecolor=inactive_color, markersize=10, markeredgecolor='black')
        virtual_screen_symbol = Line2D([0], [0], marker='x', color='w', label='Virtual Screen', markerfacecolor=virtual_screen_color, markersize=10, markeredgecolor='black')

        # Decide which symbols to include in the legend
        legend_symbols = []
        if not hide_training:
            legend_symbols.extend([active_symbol, inactive_symbol])
        if plot_test:
            legend_symbols.extend([active_test_symbol, inactive_test_symbol])
        if plot_virtual_screening:
            legend_symbols.append(virtual_screen_symbol)

        ax.legend(handles=legend_symbols, fontsize=15, loc='upper right', edgecolor='black')
    
    # Draw the threshold line
    ax.axvline(x=hs.thresholds[0].cut_value, color='black', linestyle='--')
    # Draw y_cut line
    ax.axhline(y=hs.y_cut, color='r', linestyle='--')

    # Axis setup
    ax.set_xlabel(hs.thresholds[0].feature_name, fontsize=25)
    ax.set_ylabel(output_label, fontsize=25)
    ax.tick_params(axis='x', labelsize=18)
    ax.set_xlim(x_min, x_max)
    ax.locator_params(axis='x', nbins=5)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(y_min, y_max)
    ax.locator_params(axis='y', nbins=4)

    ax.set_title(f'{hs.thresholds[0].feature_name} Threshold', fontsize=25, pad=10)
    
    plt.show()

def plot_double_threshold(hs:Hotspot, 
                          test_response_data:Optional[pd.DataFrame] = None, vs_parameters:Optional[pd.DataFrame] = None,
                          subset:str = 'all', hide_training:bool = False,
                          coloring:str = 'scaled', gradient_color:str = 'Oranges', output_label:str = 'Yield (%)'):
    """
    Plot a double threshold in 2 dimensions

    :hs: Hotspot object to plot
    :test_response_data: DataFrame of test set response data (optional)
    :vs_parameters: DataFrame of virtual screening / test set parameters (optional)
    :subset: 'all', 'train', or 'validation'; indicates which subset to show on the plot
    :hide_training: True or False; indicates if the training set should be hidden
    :coloring: 'scaled' or 'binary'; indicates if points should be colored based on actual output values or by output category
    :gradient_color: the color scheme applied to the heatmap, default 'Oranges'
    :output_label: default 'Yield (%)'
    """

    # Set up flags for what kind of plotting is requested
    plot_test = test_response_data is not None and vs_parameters is not None
    plot_virtual_screening = test_response_data is None and vs_parameters is not None 

    x_col,y_col = hs.thresholds[0].feature_name, hs.thresholds[1].feature_name
    fig, ax = plt.subplots(figsize=(10, 8))

    # This section auto-scales the plot
    if plot_virtual_screening:
        x_values = list(chain(*[hs.data_df.loc[:, x_col], vs_parameters.loc[:, x_col]]))
        y_values = list(chain(*[hs.data_df.loc[:, y_col], vs_parameters.loc[:, y_col]]))
    elif plot_test:
        x_values = list(chain(*[hs.data_df.loc[:, x_col], vs_parameters.loc[test_response_data.index, x_col]]))
        y_values = list(chain(*[hs.data_df.loc[:, y_col], vs_parameters.loc[test_response_data.index, y_col]]))
    else:
        x_values = hs.data_df.loc[:, x_col]
        y_values = hs.data_df.loc[:, y_col]

    x_min = float(min(x_values))
    x_max = float(max(x_values))
    y_min = float(min(y_values))
    y_max = float(max(y_values))
    
    dx = abs(x_min - x_max)
    dy = abs(y_min - y_max)

    x_min = x_min - abs(dx * 0.05)
    x_max = x_max + abs(dx * 0.05)
    y_min = y_min - abs(dy * 0.05)
    y_max = y_max + abs(dy * 0.05)
    
    # Set which points to plot based on the subset parameter
    if(subset == 'all'):
        points_to_plot = hs.data_df.index
    elif(subset == 'train'):
        points_to_plot = hs.training_set
    elif(subset == 'validation'):
        points_to_plot = hs.validation_set
    else:
        raise ValueError('Subset must be "all", "train", or "validation"')
    
    # Change how the points are colored, controlled by the coloring parameter
    if(coloring=='scaled'):
        mapping_cl = hs.data_df.loc[points_to_plot, output_label]
        if(plot_test):
            test_mapping_cl = test_response_data.iloc[:, 0]
    elif(coloring=='binary'):
        mapping_cl = hs.data_df.loc[points_to_plot, 'y_class']
        if(plot_test):
            test_mapping_cl = [1 if i >= hs.y_cut else 0 for i in test_response_data.iloc[:, 0]]
    else:
        raise ValueError('coloring must be either "scaled" or "binary"')

    # Plot the main dataset if not hiding it
    if not hide_training:
        x = hs.data_df.loc[points_to_plot,x_col]
        y = hs.data_df.loc[points_to_plot,y_col]
        if plot_test: alpha = 0.5
        else: alpha=1
        ax.scatter(x, y, c=mapping_cl,cmap=gradient_color, edgecolor='black', alpha=alpha, s=100, marker='o')  

    # Plot the virtual screening set if only given parameters
    if(plot_virtual_screening):
        vs_x = vs_parameters.loc[:, x_col]
        vs_y = vs_parameters.loc[:, y_col]
        ax.scatter(vs_x, vs_y, c='grey', edgecolor='black', alpha=0.5, linewidth=2, s=100, marker='x')

    # Plot the test data set if given parameters and response
    if(plot_test):
        test_x = vs_parameters.loc[test_response_data.index, x_col]
        test_y = vs_parameters.loc[test_response_data.index, y_col]
        ax.scatter(test_x, test_y, c=test_mapping_cl, cmap=gradient_color, edgecolor='black', linewidth=2, s=100, marker='s')

    # Draw threshold lines
    ax.axhline(y=hs.thresholds[1].cut_value, color='black', linestyle='--')
    ax.axvline(x=hs.thresholds[0].cut_value, color='black', linestyle='--')
    
    # Set the gradient bar or binary legend
    if(coloring == 'scaled'):
        norm = Normalize(vmin=min(mapping_cl), vmax=max(mapping_cl))
        mappable = ScalarMappable(cmap=gradient_color, norm=norm)
        mappable.set_array([])
        
        cbar = plt.colorbar(mappable, ax=ax, shrink=1)
        cbar.set_label(output_label, rotation=90, size=25)

        # Define the legend symbols
        training_symbol = Line2D([0], [0], marker='o', color='w', label='Training', markerfacecolor='white', markersize=10, markeredgecolor='black')
        test_symbol = Line2D([0], [0], marker='s', color='w', label='Test', markerfacecolor='white', markersize=10, markeredgecolor='black')
        virtual_screen_symbol = Line2D([0], [0], marker='x', color='w', label='Virtual Screen', markerfacecolor='white', markersize=10, markeredgecolor='black')

        # Decide which symbols to include in the legend
        legend_symbols = []
        if ((plot_test or plot_virtual_screening) and not hide_training):
            legend_symbols.extend([training_symbol])
        if plot_test:
            legend_symbols.extend([test_symbol])
        if plot_virtual_screening:
            legend_symbols.append([virtual_screen_symbol])

        if plot_virtual_screening or plot_test:
            ax.legend(handles=legend_symbols, fontsize=15, loc='upper right', edgecolor='black')

    elif(coloring == 'binary'):
        # Define the legend colors
        colormap = plt.get_cmap(gradient_color)
        active_color = mcolors.to_hex(colormap(1.0))
        inactive_color = mcolors.to_hex(colormap(0.0))
        virtual_screen_color = mcolors.to_hex('grey')
 
        # Define the legend symbols
        active_symbol = Line2D([0], [0], marker='o', color='w', label='Active', markerfacecolor=active_color, markersize=10, markeredgecolor='black')
        inactive_symbol = Line2D([0], [0], marker='o', color='w', label='Inactive', markerfacecolor=inactive_color, markersize=10, markeredgecolor='black')
        active_test_symbol = Line2D([0], [0], marker='s', color='w', label='Active Test', markerfacecolor=active_color, markersize=10, markeredgecolor='black')
        inactive_test_symbol = Line2D([0], [0], marker='s', color='w', label='Inactive Test', markerfacecolor=inactive_color, markersize=10, markeredgecolor='black')
        virtual_screen_symbol = Line2D([0], [0], marker='x', color='w', label='Virtual Screen', markerfacecolor=virtual_screen_color, markersize=10, markeredgecolor='black')

        # Decide which symbols to include in the legend
        legend_symbols = []
        if not hide_training:
            legend_symbols.extend([active_symbol, inactive_symbol])
        if plot_test:
            legend_symbols.extend([active_test_symbol, inactive_test_symbol])
        if plot_virtual_screening:
            legend_symbols.append(virtual_screen_symbol)

        ax.legend(handles=legend_symbols, fontsize=15, loc='upper right', edgecolor='black')

    # Axis setup
    ax.set_xlabel(hs.thresholds[0].feature_name, fontsize=25)
    ax.set_ylabel(hs.thresholds[1].feature_name, fontsize=25)
    ax.tick_params(axis='x', labelsize=18)
    ax.set_xlim(x_min, x_max)
    ax.locator_params(axis='x', nbins=5)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(y_min, y_max)
    ax.locator_params(axis='y', nbins=4)


    # Print the title of the plot
    ax.set_title(f'{hs.thresholds[0].feature_name} x {hs.thresholds[1].feature_name}', fontsize = 25)

    plt.show()

def plot_triple_threshold(hs:Hotspot,
                          test_response_data:Optional[pd.DataFrame] = None, vs_parameters:Optional[pd.DataFrame] = None,
                          subset:str ='all', hide_training:bool = False,
                          coloring:str = 'scaled', gradient_color:str = 'Oranges', output_label:str = 'Yield (%)'):
    """
    Plot a triple threshold in 3 dimensions

    :hs: Hotspot object to plot
    :test_response_data: DataFrame of test set response data (optional)
    :vs_parameters: DataFrame of virtual screening / test set parameters (optional)
    :subset: 'all', 'train', or 'validation'; indicates which subset to show on the plot
    :hide_training: True or False; indicates if the training set should be hidden
    :coloring: 'scaled' or 'binary'; indicates if points should be colored based on actual output values or by output category
    :gradient_color: the color scheme applied to the heatmap, default 'Oranges'
    :output_label: default 'Yield (%)'
    """

    # Set up flags for what kind of plotting is requested
    plot_test = test_response_data is not None and vs_parameters is not None
    plot_virtual_screening = test_response_data is None and vs_parameters is not None  

    x_col,y_col,z_col = hs.thresholds[0].feature_name, hs.thresholds[1].feature_name, hs.thresholds[2].feature_name
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection = '3d')

    # This section auto-scales the plot
    if plot_virtual_screening:
        x_values = list(chain(*[hs.data_df.loc[:, x_col], vs_parameters.loc[:, x_col]]))
        y_values = list(chain(*[hs.data_df.loc[:, y_col], vs_parameters.loc[:, y_col]]))
        z_values = list(chain(*[hs.data_df.loc[:, z_col], vs_parameters.loc[:, z_col]]))
    elif plot_test:
        x_values = list(chain(*[hs.data_df.loc[:, x_col], vs_parameters.loc[test_response_data.index, x_col]]))
        y_values = list(chain(*[hs.data_df.loc[:, y_col], vs_parameters.loc[test_response_data.index, y_col]]))
        z_values = list(chain(*[hs.data_df.loc[:, z_col], vs_parameters.loc[test_response_data.index, z_col]]))
    else:
        x_values = hs.data_df.loc[:, x_col]
        y_values = hs.data_df.loc[:, y_col]
        z_values = hs.data_df.loc[:, z_col]

    x_min = float(min(x_values))
    x_max = float(max(x_values))
    y_min = float(min(y_values))
    y_max = float(max(y_values))
    z_min = float(min(z_values))
    z_max = float(max(z_values))

    dx = abs(x_min - x_max)
    dy = abs(y_min - y_max)
    dz = abs(z_min - z_max)

    x_min = x_min - abs(dx * 0.05)
    x_max = x_max + abs(dx * 0.05)
    y_min = y_min - abs(dy * 0.05)
    y_max = y_max + abs(dy * 0.05)
    z_min = z_min - abs(dz * 0.05)
    z_max = z_max + abs(dz * 0.05)

    # Set which points to plot based on the subset parameter
    if(subset == 'all'):
        points_to_plot = hs.data_df.index
    elif(subset == 'train'):
        points_to_plot = hs.training_set
    elif(subset == 'validation'):
        points_to_plot = hs.validation_set
    else:
        raise ValueError('Subset must be "all", "train", or "validation"')
        
    # Change how the points are colored, controlled by the coloring parameter
    if(coloring=='scaled'):
        mapping_cl = hs.data_df.loc[points_to_plot, output_label]
        if(plot_test):
            test_mapping_cl = test_response_data.iloc[:, 0]
    elif(coloring=='binary'):
        mapping_cl = hs.data_df.loc[points_to_plot, 'y_class']
        if(plot_test):
            test_mapping_cl = [1 if i >= hs.y_cut else 0 for i in test_response_data.iloc[:, 0]]
    else:
        raise ValueError('coloring must be either "scaled" or "binary"')

    # Plot the virtual screening set if only given parameters
    if(plot_virtual_screening):
        vs_x = vs_parameters.loc[:, x_col]
        vs_y = vs_parameters.loc[:, y_col]
        vs_z = vs_parameters.loc[:, z_col]
        ax.scatter(vs_x, vs_y, vs_z, c='grey', linewidth=2, alpha=0.5, marker="x", s=50, edgecolors='k')
    
    # Plot the main dataset if not hiding it
    if not hide_training:
        x = hs.data_df.loc[points_to_plot,x_col]
        y = hs.data_df.loc[points_to_plot,y_col]
        z = hs.data_df.loc[points_to_plot,z_col]
        if plot_test: alpha = 0.5
        else: alpha=0.95
        ax.scatter(x, y, z, c=mapping_cl, cmap=gradient_color, alpha=alpha, marker="o", s=50, edgecolors='k')

    # Plot the test data set if given parameters and response
    if(plot_test):
        test_x = vs_parameters.loc[test_response_data.index, x_col]
        test_y = vs_parameters.loc[test_response_data.index, y_col]
        test_z = vs_parameters.loc[test_response_data.index, z_col]
        ax.scatter(test_x, test_y, test_z, c=test_mapping_cl, cmap=gradient_color, linewidth=2, alpha=0.95, marker="s", s=50, edgecolors='k')
        
    # Plot the z-axis threshold
    temp_x = np.linspace(x_min, x_max, num=10)
    temp_y = np.linspace(y_min, y_max, num=10)
    temp_x, temp_y = np.meshgrid(temp_x, temp_y)
    temp_z = hs.thresholds[2].cut_value + 0 * temp_x + 0 * temp_y
    ax.plot_surface(temp_x, temp_y, temp_z, alpha=0.15, color='gray')
    
    # Plot the x-axis threshold
    temp_y = np.linspace(y_min, y_max, num=10)
    temp_z = np.linspace(z_min, z_max, num=10)
    temp_z, temp_y = np.meshgrid(temp_z, temp_y)
    temp_x = hs.thresholds[0].cut_value + 0 * temp_z + 0 * temp_y
    ax.plot_surface(temp_x, temp_y, temp_z, alpha=0.15, color='gray') 
    
    # Plot the y-axis threshold
    temp_x = np.linspace(x_min, x_max, num = 10)
    temp_z = np.linspace(z_min, z_max, num = 10)
    temp_x, temp_z = np.meshgrid(temp_x, temp_z)
    temp_y = hs.thresholds[1].cut_value + 0 * temp_x + 0 * temp_z
    ax.plot_surface(temp_x, temp_y, temp_z, alpha=0.15, color='gray')
    
    plt.xticks(fontsize = 10) 
    plt.yticks(fontsize = 10)

    # Set axes labels
    ax.set_xlabel(hs.thresholds[0].feature_name,fontsize=12.5)
    ax.set_ylabel(hs.thresholds[1].feature_name,fontsize=12.5)
    ax.set_zlabel(hs.thresholds[2].feature_name,fontsize=12.5)
    plt.locator_params(axis='y', nbins=8)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Set the gradient bar on the side
    if(coloring == 'scaled'):
        norm = Normalize(vmin=min(mapping_cl), vmax=max(mapping_cl))
        mappable = ScalarMappable(cmap=gradient_color, norm=norm)
        mappable.set_array([])
        
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5)
        cbar.set_label(output_label, rotation=90, size=18)

       # Define the legend symbols
        training_symbol = Line2D([0], [0], marker='o', color='w', label='Training', markerfacecolor='white', markersize=10, markeredgecolor='black')
        test_symbol = Line2D([0], [0], marker='s', color='w', label='Test', markerfacecolor='white', markersize=10, markeredgecolor='black')
        virtual_screen_symbol = Line2D([0], [0], marker='x', color='w', label='Virtual Screen', markerfacecolor='white', markersize=10, markeredgecolor='black')

        # Decide which symbols to include in the legend
        legend_symbols = []
        if ((plot_test or plot_virtual_screening) and not hide_training):
            legend_symbols.extend([training_symbol])
        if plot_test:
            legend_symbols.extend([test_symbol])
        if plot_virtual_screening:
            legend_symbols.append([virtual_screen_symbol])

        if plot_virtual_screening or plot_test:
            ax.legend(handles=legend_symbols, fontsize=15, loc='upper right', edgecolor='black')

    elif(coloring == 'binary'):
        # Define the legend colors
        colormap = plt.get_cmap(gradient_color)
        active_color = mcolors.to_hex(colormap(1.0))
        inactive_color = mcolors.to_hex(colormap(0.0))
        virtual_screen_color = mcolors.to_hex('grey')
 
        # Define the legend symbols
        active_symbol = Line2D([0], [0], marker='o', color='w', label='Active', markerfacecolor=active_color, markersize=10, markeredgecolor='black')
        inactive_symbol = Line2D([0], [0], marker='o', color='w', label='Inactive', markerfacecolor=inactive_color, markersize=10, markeredgecolor='black')
        active_test_symbol = Line2D([0], [0], marker='s', color='w', label='Active Test', markerfacecolor=active_color, markersize=10, markeredgecolor='black')
        inactive_test_symbol = Line2D([0], [0], marker='s', color='w', label='Inactive Test', markerfacecolor=inactive_color, markersize=10, markeredgecolor='black')
        virtual_screen_symbol = Line2D([0], [0], marker='x', color='w', label='Virtual Screen', markerfacecolor=virtual_screen_color, markersize=10, markeredgecolor='black')

        # Decide which symbols to include in the legend
        legend_symbols = []
        if not hide_training:
            legend_symbols.extend([active_symbol, inactive_symbol])
        if plot_test:
            legend_symbols.extend([active_test_symbol, inactive_test_symbol])
        if plot_virtual_screening:
            legend_symbols.append(virtual_screen_symbol)

        plt.legend(handles=legend_symbols, fontsize=15, loc='upper right', edgecolor='black')

    plt.show()

def train_test_splits(temp_data_df:pd.DataFrame, split:str, validation_ratio:float, test_ratio:float, feature_names:list[str], response_label:str, use_test=False,
                      randomstate:int = 0, subset:list[int] = [], stratified_quantiles:int = 10, verbose:bool = True,
                      defined_training_set:list=[], defined_validation_set:list=[], defined_test_set:list=[]) -> tuple[list[str], list[str], list[str]]:
    """
    Given the main dataframe and some parameters, return lists of y index values for training, validation, and potentially test sets.
    Training ratio is 1 - validation_ratio - test_ratio.
    Function updated with correct train/validation/test labeling

    :data_df: The master dataframe with x# column names and the first two columns as 'response' and 'y_class'
    :split: 'random', 'ks', 'y_equidistant', 'stratified', 'define', 'none'; Type of split to use
    :validation_ratio: Ratio of the data to use as a validation set
    :test_ratio: Ratio of the data to use as a test set
    :feature_names: List of parameter labels corresponding to the parameter column names in the dataframe
    :response_label: The name of the response column in the dataframe
    :use_test: Whether to return a test set in addition to the training and validation sets
    :randomstate: Seed to use when chosing the random split
    :defined_training_set: Y indexes corresponding to a manual training set. Only used if split == 'define'
    :defined_validation_set: Y indexes corresponding to a manual validation set. Only used if split == 'define'
    :defined_test_set: Y indexes corresponding to a manual test set. Only used if split == 'define'
    :subset: The subset of y indexes to use for another split method, originally used for MLR after a classification algorithm
    :verbose: Whether to print the extended report
    """
    
    # Trim the data_df to only the subset if given
    if (subset == []):
        data_df = temp_data_df.copy()
    else:
        data_df = temp_data_df.loc[subset, :].copy()

    x = data_df[feature_names].to_numpy() # Array of just feature values (X_sel)
    y = data_df[response_label].to_numpy() # Array of response values (y_sel)
    
    # Calculate the sizes of the training, validation, and test sets
    validation_size = int(len(data_df.index)*validation_ratio) # Number of points in the validation set
    test_size = 0
    if use_test:
        test_size = int(len(data_df.index)*test_ratio) # Number of points in the test set

    train_size = len(data_df.index) - validation_size - test_size
    test_set = []

    if split == "random":
        # Purely random split
        random.seed(a = randomstate)
        validation_set = random.sample(list(data_df.index), k = validation_size)
        if use_test:
            test_set = random.sample([x for x in data_df.index if x not in validation_set], k = test_size)
        training_set = [x for x in data_df.index if x not in test_set and x not in validation_set]

    elif split == "stratified":
        # Stratified split based on the response variable, gives sets distributed over {stratified_quantiles} bins
        y_binned = pd.qcut(y, q=stratified_quantiles, labels=False, duplicates='drop')
        training_set, validation_set = train_test_split(range(len(data_df.index)), test_size=validation_size + test_size, stratify=y_binned, random_state=randomstate)

        if use_test:
            validation_set, test_set = train_test_split(validation_set, test_size=test_size, stratify=y_binned[validation_set], random_state=randomstate)

        training_set = list(data_df.index[training_set])
        validation_set = list(data_df.index[validation_set])
        if use_test:
            test_set = list(data_df.index[test_set])

    elif split == "ks":
        # Kennard-Stone algorithm split
        validation_set_index, training_set_index = kennardstonealgorithm(x, validation_size + test_size, randomstate)

        if use_test:
            validation_set_temp_index, test_set_temp_index = kennardstonealgorithm(x[validation_set_index], validation_size, randomstate)

            test_set_index = [validation_set_index[i] for i in test_set_temp_index]
            validation_set_index = [validation_set_index[i] for i in validation_set_temp_index]

            test_set = list(data_df.index[test_set_index])   

        training_set = list(data_df.index[training_set_index])
        validation_set = list(data_df.index[validation_set_index])

    elif split == "y_equidistant":
        # Splitting that maximizes the spread of y values in the test set
        no_extrapolation = True # If True, the min and max y values are removed from the dataset before splitting
        
        if no_extrapolation:
            # Identify the min and max y values and remove them from the dataset for the KS algorithm
            y_min = np.min(y)
            y_max = np.max(y)
            y_internal = np.array(([i for i in y if i not in [y_min,y_max]])) # y values without the min and max
            y_internal_indices = [i for i, val in enumerate(y) if val != y_min and val != y_max] # indices of y that are in y_internal
            y_extrema_indices = [i for i, val in enumerate(y) if val == y_min or val == y_max] # indices of y that are not in y_internal

            # Run the KS algorithm on the internal y values
            y_internal_formatted = y_internal.reshape(np.shape(y_internal)[0], 1)
            # training_set_index, validation_set_index = kennardstonealgorithm(y_internal_formatted, train_size, randomstate)
            validation_set_index, training_set_index = kennardstonealgorithm(y_internal_formatted, test_size+validation_size, randomstate)

            if use_test:
                validation_set_temp_index, test_set_temp_index = kennardstonealgorithm(y_internal_formatted[validation_set_index], validation_size, randomstate)
            
            # Convert indices relative to y_internal
            if use_test:
                test_set_index = [validation_set_index[i] for i in test_set_temp_index]
                validation_set_index = [validation_set_index[i] for i in validation_set_temp_index]

            # Convert indices relative to y
            training_set_index = sorted([y_internal_indices[i] for i in list(training_set_index)] + y_extrema_indices)
            validation_set_index = sorted([y_internal_indices[i] for i in validation_set_index])
            if use_test:
                test_set_index = sorted([y_internal_indices[i] for i in test_set_index])

        else:
            y_formatted = y.reshape(np.shape(y)[0], 1)
            training_set_index, validation_set_index = kennardstonealgorithm(y_formatted, train_size, randomstate)
            if use_test:
                validation_set_temp_index, test_set_temp_index = kennardstonealgorithm(y_formatted[validation_set_index], validation_size, randomstate)
                test_set_index = [validation_set_index[i] for i in test_set_temp_index]
                validation_set_index = [validation_set_index[i] for i in validation_set_temp_index]

        # Convert indices to row names
        training_set = list(data_df.index[training_set_index])
        validation_set = list(data_df.index[validation_set_index])
        if use_test:
            test_set = list(data_df.index[test_set_index])           

    elif split == 'define':
        # Manually defined training and test sets
        training_set = defined_training_set
        test_set = defined_test_set
        validation_set = defined_validation_set

    elif split == "none":
        # No split, just use the entire dataset as the training set
        training_set = data_df.index.to_list()
        test_set = []
        validation_set = []

    else: 
        raise ValueError("split option not recognized")
    
    if(verbose):
        y_train = data_df.loc[training_set, response_label]
        y_validate = data_df.loc[validation_set, response_label]
        y_test = data_df.loc[test_set, response_label]

        print(f"Training Set: {training_set}")
        print(f'Validation Set: {validation_set}')
        if use_test: print(f"Test Set: {test_set}")
        if set(training_set).union(set(test_set)).union(set(validation_set)) != set(data_df.index):
            print('Missing indices!')

        print("\nTraining Set size: {}".format(len(training_set)))
        print('Validation Set size: {}'.format(len(validation_set)))
        if use_test: print("Test Set size: {}".format(len(test_set)))
        print("\nTraining Set mean: {:.3f}".format(np.mean(y_train)))
        print("Validation Set mean: {:.3f}".format(np.mean(y_validate)))
        if use_test: print("Test Set mean: {:.3f}".format(np.mean(y_test)))

        # Plot the distribution of the sets
        plt.figure(figsize=(5, 5))
        hist, bins = np.histogram(y,bins="auto")

        hist_data = [y_train, y_validate, y_test] if use_test else [y_train, y_validate]
        hist_labels = ['y_train', 'y_validate', 'y_test'] if use_test else ['y_train', 'y_validate']
        hist_colors = ['black', '#BE0000', '#008090'] if use_test else ['black', '#BE0000']
        plt.hist(hist_data, bins, alpha=0.5, stacked=True, label=hist_labels, color=hist_colors)

        # plt.hist(y_train, bins, stacked=True, alpha=0.5, label='y_train',color="black")
        # if use_test: plt.hist(y_validate, bins, stacked=True, alpha=0.5, label='y_validate',color="red")
        # plt.hist(y_test, bins, stacked=True, alpha=0.5, label='y_test')

        plt.legend(loc='best')
        plt.xlabel(response_label,fontsize=20)
        plt.ylabel("N samples",fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.show()

    return training_set, validation_set, test_set

def kennardstonealgorithm(X:np.ndarray, k:int, randomseed:int = 0) -> tuple[list[int], list[int]]:
    X = np.array( X )
    originalX = X
    np.random.seed(randomseed)

    # Find the average value vector of the dataset and calculate the distance of each sample to the average value
    distancetoaverage = ( (X - np.tile(X.mean(axis=0), (X.shape[0], 1) ) )**2 ).sum(axis=1)

    # Find the sample with the maximum distance to the average value
    maxdistancesamplenumber = np.where( distancetoaverage == np.max(distancetoaverage) )
    # maxdistancesamplenumber = maxdistancesamplenumber[0][0] # This line selects the first occurance of the maximum distance with the second index
    maxdistancesamplenumber = np.random.choice(maxdistancesamplenumber[0]) # This line randomly selects one of the maximum distance samples
    selectedsamplenumbers = list()
    selectedsamplenumbers.append(maxdistancesamplenumber)

    # Remove the sample with the maximum distance to the average value from the dataset
    remainingsamplenumbers = np.arange( 0, X.shape[0], 1)
    X = np.delete( X, selectedsamplenumbers, 0)
    remainingsamplenumbers = np.delete( remainingsamplenumbers, selectedsamplenumbers, 0)

    for iteration in range(1, k):
        selectedsamples = originalX[selectedsamplenumbers,:]
        mindistancetoselectedsamples = list()
        for mindistancecalculationnumber in range( 0, X.shape[0]):
            distancetoselectedsamples = ( (selectedsamples - np.tile(X[mindistancecalculationnumber,:], (selectedsamples.shape[0], 1)) )**2 ).sum(axis=1)
            mindistancetoselectedsamples.append( np.min(distancetoselectedsamples) )
        maxdistancesamplenumber = np.where( mindistancetoselectedsamples == np.max(mindistancetoselectedsamples) )
        # maxdistancesamplenumber = maxdistancesamplenumber[0][0] # This line selects the first occurance of the maximum distance with the second index
        maxdistancesamplenumber = np.random.choice(maxdistancesamplenumber[0]) # This line randomly selects one of the maximum distance samples
        selectedsamplenumbers.append(remainingsamplenumbers[maxdistancesamplenumber])
        X = np.delete( X, maxdistancesamplenumber, 0)
        remainingsamplenumbers = np.delete( remainingsamplenumbers, maxdistancesamplenumber, 0).tolist()

    return(selectedsamplenumbers, remainingsamplenumbers)

