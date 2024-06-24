from __future__ import annotations
import multiprocessing
from hotspot_classes import Threshold, Hotspot
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from itertools import repeat
import random


def threshold_generation(data_df:pd.DataFrame, class_weight:dict, evaluation_method:str, x_labelname_dict:dict, features:list[str] = ['empty']) -> list[Threshold]:
    """
    Given the master dataframe and some parameters, return the best threshold in each feature.

    :data_df: The master dataframe with x# column names and the first two columns as 'response' and 'y_class'
    :class_weight: Mapping of hits (1) and misses (0) to their respective class weights. Example: {1:10, 0:1}
    :evaluation_method: 'accuracy', 'weighted_accuracy', 'f1', 'weighted_f1'; Primary accuracy metric to be used in threshold comparison
    :x_labelname_dict: Dictionary for converting x# labels to full feature names
    :features: List of x# parameter names to get thresholds for.  Primarily used for manual hotspot selection.
    """

    if(features==['empty']):
        features = list(x_labelname_dict.keys())

    all_thresholds = []
    for f_ind in features:
        x = (data_df.loc[:,f_ind].values).reshape(-1, 1) # pulls the relevant parameter column and formats it in the propper array
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
            f_ind, 
            dt.tree_.threshold[0],
            operator, 
            feature_name = x_labelname_dict[f_ind],
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

def hs_next_thresholds(hs:Hotspot, data_df:pd.DataFrame, class_weight:dict, x_labelname_dict:dict, features:list[str] = ['empty']) -> list[Hotspot]:
    """
    Given the master dataframe and some parameters, return the best threshold in each feature.

    :hs: Hotspot to add additional thresholds to
    :data_df: The master dataframe with x# column names and the first two columns as 'response' and 'y_class'
    :class_weight: Mapping of hits (1) and misses (0) to their respective class weights. Example: {1:10, 0:1}
    :x_labelname_dict: Dictionary for converting x# labels to full feature names
    :features: List of x# parameter names to get thresholds for.  Primarily used for manual hotspot selection.
    """

    if(features==['empty']):
        features = list(x_labelname_dict.keys())

    # Makes all possible hotspots by adding one threshold
    all_hotspots = []
    for f_ind in features:
        x = (data_df.loc[:,f_ind].values).reshape(-1, 1) # pulls the relevant parameter column and formats it in the propper array
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
            f_ind, 
            dt.tree_.threshold[0],
            operator, 
            feature_name = x_labelname_dict[f_ind],
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
        accuracy_list.append(hs.accuracy_dict[evaluation_method])

    cut = np.percentile(accuracy_list, 100 - percentage)
    
    hs_out=[]
    for hs in hotspots:
        if(hs.accuracy_dict[evaluation_method]>=cut):
            hs_out.append(hs)
    
    return hs_out

def plot_threshold(hs:Hotspot, subset:str = 'all', coloring:str = 'scaled', gradient_color:str = 'Oranges', output_label:str = '% Yield', active_label:str = 'Active', inactive_label:str = 'Inactive'):
    """
    Plot a single, double, or triple threshold by calling the relevant function

    :hs: Hotspot object to plot
    :subset: 'all', 'train', or 'test'; indicates which subset to show on the plot
    :coloring: 'scaled' or 'binary'; indicates if points should be colored based on actual output values or by output category
    :gradient_color: the color scheme applied to the heatmap, default 'Oranges'
    :output_label: default '% Yield'
    """
    if(len(hs.thresholds)==1):
        plot_single_threshold(hs, subset, coloring, gradient_color, output_label, active_label, inactive_label)
    elif(len(hs.thresholds)==2):
        plot_double_threshold(hs, subset, coloring, gradient_color, output_label, active_label, inactive_label)
    elif(len(hs.thresholds)==3):
        plot_triple_threshold(hs, subset, coloring, gradient_color, output_label, active_label, inactive_label)
    else:
        print(f'Unable to plot {len(hs.thresholds)} thresholds')

def plot_single_threshold(hs:Hotspot, subset:str = 'all', coloring:str = 'scaled', gradient_color:str = 'Oranges', output_label:str = '% Yield', active_label:str = 'Active', inactive_label:str = 'Inactive'):
    """
    Plot a single threshold in 2 dimensions

    :hs: Hotspot object to plot
    :subset: 'all', 'train', or 'test'; indicates which subset to show on the plot
    :coloring: 'scaled' or 'binary'; indicates if points should be colored based on actual output values or by output category
    :gradient_color: the color scheme applied to the heatmap, default 'Oranges'
    :output_label: default '% Yield'
    """

    x_col = hs.thresholds[0].index
    plt.figure(figsize=(12, 8))

    # This section auto-scales the plot
    x_min = float(min(hs.data_df.loc[:, x_col]))
    x_max = float(max(hs.data_df.loc[:, x_col]))
    y_min = float(min(hs.data_df.loc[:, 'response']))
    y_max = float(max(hs.data_df.loc[:, 'response']))

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
    elif(subset == 'test'):
        points_to_plot = hs.test_set
    else:
        points_to_plot = []
    
    # Change how the points are colored, controlled by the coloring parameter
    if(coloring=='scaled'):
        mapping_cl = hs.data_df.loc[points_to_plot, 'response']
    elif(coloring=='binary'):
        mapping_cl = hs.data_df.loc[points_to_plot, 'y_class']
    else:
        mapping_cl = hs.data_df.loc[points_to_plot, 'response']

    # This is where it selects which descriptors to plot
    x = hs.data_df.loc[points_to_plot, x_col]
    y = hs.data_df.loc[points_to_plot, 'response']
    plt.scatter(x, y, c = mapping_cl, cmap = gradient_color, edgecolor ='black', s = 100)
    
    # Set the gradient bar or binary legend
    if(coloring == 'scaled'):
        cbar = plt.colorbar(plt.gca().collections[0], shrink=1)
        cbar.set_label(output_label, rotation=90, size=25)     
        cbar.ax.tick_params(labelsize=18)   
    elif(coloring == 'binary'):
        colormap = plt.get_cmap(gradient_color)
        active_color = mcolors.to_hex(colormap(1.0))
        inactive_color = mcolors.to_hex(colormap(0.0))
        active_patch = mpatches.Patch(color=active_color, label=active_label, edgecolor='black')
        inactive_patch = mpatches.Patch(color=inactive_color, label=inactive_label, edgecolor='black')
        plt.legend(handles=[active_patch, inactive_patch], fontsize=15, loc='upper right', edgecolor='black')
 
    
    # Draw the threshold line
    plt.axvline(x=hs.thresholds[0].cut_value, color='black', linestyle='--')
    # Draw y_cut line
    plt.axhline(y=hs.y_cut, color='r', linestyle='--')

    # Axis setup
    plt.xlabel(f'{hs.thresholds[0].feature_label} {hs.thresholds[0].feature_name}',fontsize=25)
    plt.ylabel(output_label,fontsize=25)
    plt.xticks(fontsize=18)
    plt.xlim(x_min, x_max)
    plt.locator_params(axis='x', nbins=5)
    plt.yticks(fontsize=18)
    plt.ylim(y_min, y_max)
    plt.locator_params(axis='y', nbins=4)

    plt.title(f'{hs.thresholds[0].feature_name} Threshold', fontsize = 25)
    
    plt.show()

def plot_double_threshold(hs:Hotspot, subset:str = 'all', coloring:str = 'scaled', gradient_color:str = 'Oranges', output_label:str = 'Yield (%)', active_label:str = 'Active', inactive_label:str = 'Inactive'):
    """
    Plot a double threshold in 2 dimensions

    :subset: 'all', 'train', or 'test'; indicates which subset to show on the plot
    :coloring: 'scaled' or 'binary'; indicates if points should be colored based on actual output values or by output category
    :gradient_color: the color scheme applied to the heatmap, default 'Oranges'
    :output_label: default 'Yield (%)'
    """

    x_col,y_col = hs.thresholds[0].index, hs.thresholds[1].index
    plt.figure(figsize=(12, 8))

    # This section auto-scales the plot
    x_min = float(min(hs.data_df.loc[:, x_col]))
    x_max = float(max(hs.data_df.loc[:, x_col]))
    y_min = float(min(hs.data_df.loc[:, y_col]))
    y_max = float(max(hs.data_df.loc[:, y_col]))

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
    elif(subset == 'test'):
        points_to_plot = hs.test_set
    else:
        points_to_plot = []
    
    # Change how the points are colored, controlled by the coloring parameter
    if(coloring=='scaled'):
        mapping_cl = hs.data_df.loc[points_to_plot, 'response']
    elif(coloring=='binary'):
        mapping_cl = hs.data_df.loc[points_to_plot, 'y_class']
    else:
        mapping_cl = hs.data_df.loc[points_to_plot, 'response']

    # This is where it selects which descriptors to plot
    x = hs.data_df.loc[points_to_plot,x_col]
    y = hs.data_df.loc[points_to_plot,y_col]
    plt.scatter(x, y, c=mapping_cl,cmap=gradient_color, edgecolor='black', s=100)  

    # Draw threshold lines
    plt.axhline(y=hs.thresholds[1].cut_value, color='black', linestyle='--')
    plt.axvline(x=hs.thresholds[0].cut_value, color='black', linestyle='--')
    
    # Set the gradient bar or binary legend
    if(coloring == 'scaled'):
        cbar = plt.colorbar(plt.gca().collections[0], shrink=1)
        cbar.set_label(output_label, rotation=90, size=18)
    elif(coloring == 'binary'):
        colormap = plt.get_cmap(gradient_color)
        active_color = mcolors.to_hex(colormap(1.0))
        inactive_color = mcolors.to_hex(colormap(0.0))
        active_patch = mpatches.Patch(color=active_color, label=active_label, edgecolor='black')
        inactive_patch = mpatches.Patch(color=inactive_color, label=inactive_label, edgecolor='black')
        plt.legend(handles=[active_patch, inactive_patch], fontsize=15, loc='upper right', edgecolor='black')

    # Axis setup
    plt.xlabel(f'{hs.thresholds[0].feature_label} {hs.thresholds[0].feature_name}', fontsize = 15)
    plt.ylabel(f'{hs.thresholds[1].feature_label} {hs.thresholds[1].feature_name}', fontsize = 15)
    plt.xticks(fontsize = 18)
    plt.xlim(x_min, x_max)
    plt.locator_params(axis = 'x', nbins = 5)
    plt.yticks(fontsize = 18)
    plt.ylim(y_min, y_max)
    plt.locator_params(axis = 'y', nbins = 4)

    # Print the title of the plot
    plt.title(f'{hs.thresholds[0].feature_name} x {hs.thresholds[1].feature_name}', fontsize = 20)

    plt.show()

def plot_triple_threshold(hs:Hotspot, subset:str ='all', coloring:str = 'scaled', gradient_color:str = 'Oranges', output_label:str = '% Yield', active_label:str = 'Active', inactive_label:str = 'Inactive'):
    """
    Plot a triple threshold in 3 dimensions

    :subset: 'all', 'train', or 'test'; indicates which subset to show on the plot
    :coloring: 'scaled' or 'binary'; indicates if points should be colored based on actual output values or by output category
    :gradient_color: the color scheme applied to the heatmap, default 'Oranges'
    :output_label: default '% Yield'
    """

    x_col,y_col,z_col = hs.thresholds[0].index, hs.thresholds[1].index, hs.thresholds[2].index
    
    # Auto-scale the plot to your data
    x_min = float(min(hs.data_df.loc[:, x_col]))
    x_max = float(max(hs.data_df.loc[:, x_col]))
    y_min = float(min(hs.data_df.loc[:, y_col]))
    y_max = float(max(hs.data_df.loc[:, y_col]))
    z_min = float(min(hs.data_df.loc[:, z_col]))
    z_max = float(max(hs.data_df.loc[:, z_col]))

    dx = abs(x_min - x_max)
    dy = abs(y_min - y_max)
    dz = abs(z_min - z_max)

    x_min = x_min - abs(dx * 0.05)
    x_max = x_max + abs(dx * 0.05)
    y_min = y_min - abs(dy * 0.05)
    y_max = y_max + abs(dy * 0.05)
    z_min = z_min - abs(dz * 0.05)
    z_max = z_max + abs(dz * 0.05)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection = '3d')

    # Set which points to plot based on the subset parameter
    if(subset == 'all'):
        points_to_plot = hs.data_df.index
    elif(subset == 'train'):
        points_to_plot = hs.training_set
    elif(subset == 'test'):
        points_to_plot = hs.test_set
    else:
        points_to_plot = []
        
    # Change how the points are colored, controlled by the coloring parameter
    if(coloring=='scaled'):
        mapping_cl = hs.data_df.loc[points_to_plot, 'response']
    elif(coloring=='binary'):
        mapping_cl = hs.data_df.loc[points_to_plot, 'y_class']
    else:
        mapping_cl = hs.data_df.loc[points_to_plot, 'response']

    # Actually plot the data
    x = hs.data_df.loc[points_to_plot,x_col]
    y = hs.data_df.loc[points_to_plot,y_col]
    z = hs.data_df.loc[points_to_plot,z_col]
    ax.scatter(x, y, z, c=mapping_cl, cmap=gradient_color, alpha=0.95, marker="s", s=50, edgecolors='k') 
        
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
    ax.set_xlabel(f'{hs.thresholds[0].feature_label} {hs.thresholds[0].feature_name}',fontsize=12.5)
    ax.set_ylabel(f'{hs.thresholds[1].feature_label} {hs.thresholds[1].feature_name}',fontsize=12.5)
    ax.set_zlabel(f'{hs.thresholds[2].feature_label} {hs.thresholds[2].feature_name}',fontsize=12.5)
    plt.locator_params(axis='y', nbins=8)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Set the gradient bar on the side
    if(coloring == 'scaled'):
        cbar = plt.colorbar(plt.gca().collections[0], shrink=0.5)
        cbar.set_label(output_label, rotation=90, size=18)

    plt.show()

def train_test_splits(temp_data_df:pd.DataFrame, split:str, test_ratio:float, x_labels:list[str], response_label:str, randomstate:int = 0, defined_training_set:list[int] = [], defined_test_set:list[int] = [], subset:list[int] = [], verbose:bool = True) -> tuple[list[str], list[str]]:
    """
    Given the main dataframe and some parameters, return lists of y index values for a training and test set

    :data_df: The master dataframe with x# column names and the first two columns as 'response' and 'y_class'
    :split: 'random', 'ks', 'y_equidistant', 'define', 'none'; Type of split to use
    :test_ratio: Ratio of the data to use as a test set
    :x_labels: List of xID# parameter labels corresponding to the parameter column names in the dataframe
    :response_label: The name of the response column in the dataframe
    :randomstate: Seed to use when chosing the random split
    :defined_training_set: Y indexes corresponding to a manual training set. Only used if split == 'define'
    :defined_test_set: Y indexes corresponding to a manual test set. Only used if split == 'define'
    :subset: The subset of y indexes to use for another split method, originally used for MLR after a classification algorithm
    :verbose: Whether to print the extended report
    """
    
    if (subset == []):
        data_df = temp_data_df.copy()
    else:
        data_df = temp_data_df.loc[subset, :].copy()

    x = data_df[x_labels].to_numpy() # Array of just feature values (X_sel)
    y = data_df[response_label].to_numpy() # Array of response values (y_sel)
    test_size = int(len(data_df.index)*test_ratio) # Number of points in the test set
    train_size = len(data_df.index) - test_size

    if split == "random":
        random.seed(a = randomstate)
        test_set = random.sample(list(data_df.index), k = test_size)
        training_set = [x for x in data_df.index if x not in test_set]

    elif split == "ks":
        # There may be some issues with test_set_index being formatted as an array and training_set_index being a list
        training_set_index, test_set_index = kennardstonealgorithm(x, train_size)
        training_set = list(data_df.index[training_set_index])
        test_set = list(data_df.index[test_set_index])

    elif split == "y_equidistant":
        no_extrapolation = True
        # Only difference I can see between extrapolation and no_extrapolation is that no_e cuts off the highest and lowest y values first
        
        if no_extrapolation:
            # Rewritten from above to keep track of which points were removed for being equal to the min or max value
            y_min = np.min(y)
            y_max = np.max(y)
            y_ks = np.array(([i for i in y if i not in [y_min,y_max]]))
            y_ks_indices = [i for i, val in enumerate(y) if val != y_min and val != y_max]
            y_not_ks_indices = [i for i, val in enumerate(y) if val == y_min or val == y_max]

            # indices relative to y_ks:
            y_ks_formatted = y_ks.reshape(np.shape(y_ks)[0], 1)
            VS_ks,TS_ks = kennardstonealgorithm(y_ks_formatted, test_size)

            # indices relative to y_sel:
            TS_ = sorted([y_ks_indices[i] for i in list(TS_ks)]+y_not_ks_indices) # Replaced minmax with y_not_ks_indices
            VS_ = sorted([y_ks_indices[i] for i in VS_ks])

        else:
            VS_,TS_ = kennardstonealgorithm(y.reshape(np.shape(y)[0],1),int((test_ratio)*np.shape(y)[0]))

        training_set = list(data_df.index[TS_])
        test_set = list(data_df.index[VS_])

    elif split == 'define':
        training_set = defined_training_set
        test_set = defined_test_set

    elif split == "none":
        training_set = data_df.index.to_list()
        test_set = []

    else: 
        raise ValueError("split option not recognized")
    
    if(verbose):
        y_train = data_df.loc[training_set, response_label]
        y_test = data_df.loc[test_set, response_label]

        print(f"Training Set: {training_set}")
        print(f"Test Set: {test_set}")
        if (len(training_set) + len(test_set) == len(data_df.index)):
            print('All indices accounted for!')
        else:
            print('Missing indices!')

        print("Training Set mean: {:.3f}".format(np.mean(y_train)))
        print("Test Set mean: {:.3f}".format(np.mean(y_test)))
        # print("Shape X_train: {}".format(X_train.shape))
        # print("Shape X_test:  {}".format(X_test.shape))   
        plt.figure(figsize=(5, 5))
        hist, bins = np.histogram(y,bins="auto")#"auto"
        plt.hist(y_train, bins, alpha=0.5, label='y_train',color="black")
        plt.hist(y_test, bins, alpha=0.5, label='y_test')
        plt.legend(loc='best')
        plt.xlabel("Output",fontsize=20)
        plt.ylabel("N samples",fontsize=20)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.show()

    return training_set, test_set

# Still need to clean this one up a bit
def kennardstonealgorithm( X, k ):
    X = np.array( X )
    originalX = X
    distancetoaverage = ( (X - np.tile(X.mean(axis=0), (X.shape[0], 1) ) )**2 ).sum(axis=1)
    maxdistancesamplenumber = np.where( distancetoaverage == np.max(distancetoaverage) )
    maxdistancesamplenumber = maxdistancesamplenumber[0][0]
    selectedsamplenumbers = list()
    selectedsamplenumbers.append(maxdistancesamplenumber)
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
        maxdistancesamplenumber = maxdistancesamplenumber[0][0]
        selectedsamplenumbers.append(remainingsamplenumbers[maxdistancesamplenumber])
        X = np.delete( X, maxdistancesamplenumber, 0)
        remainingsamplenumbers = np.delete( remainingsamplenumbers, maxdistancesamplenumber, 0)

    return(selectedsamplenumbers, remainingsamplenumbers)