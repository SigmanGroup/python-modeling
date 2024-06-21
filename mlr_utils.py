from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn import metrics
import loo_q2 as loo
import itertools
import time
from sklearn.model_selection import RepeatedKFold, LeaveOneOut

import multiprocessing
nproc = max([1,multiprocessing.cpu_count()-2]) # Set the number of CPUs to use in parallel computation
from joblib import Parallel,delayed

# Determine if usescore is ever relevant and then get rid of it
class Model:
    def __init__(self, terms:tuple, X:pd.DataFrame, y:pd.DataFrame, regression_type, usescore:str = 'q2'):
        """
        An object for storing a single MLR model.  When initialized, fits a regression model with the data given.

        :terms: Tuple of x# defining the parameters used in this model
        :X: Subset of the training dataframe containing the columns corresponding to the terms
        :y: Response column of the training dataframe
        :regression_type: The type of regressor to use
        :usescore: Not relevant. Set as anything except 'q2' if you don't want to use q^2 in your models.
        """

        self.terms = terms
        self.n_terms = len(terms)
        self.formula = '1 + ' + ' + '.join(terms)
        self.model = regression_type().fit(X,y)
        self.r2 = self.model.score(X,y)

        # q2 is only calculated when requested as it is computationally expensive
        if usescore=='q2':
            self.q2 = loo.q2_df(X,y,regression_type())[0]

def filter_unique(models:dict, step:int, comparison_score:str = 'r2'):
    """
    Sort through the models in the scores dictionary and return a list of terms-tuples that have less than n parameters in common.
    N here is determined by the step number: 1 for up to 4-term-models; 2 for 5 to 7 terms; 3 for 8+ terms.
    When two models have more than n parameters in common, the one with the lower score is removed.
    """
    # models_sorted = sorted(scores, key=scores.get, reverse=True)
    if(comparison_score == 'r2'):
        models_sorted = sorted(models, key=lambda model: models[model].r2, reverse=True)
    elif(comparison_score == 'q2'):
        models_with_q2 = [model for model in models if hasattr(models[model], 'q2')]
        models_sorted = sorted(models_with_q2, key=lambda model: models[model].q2, reverse=True)
    else:
        raise ValueError('Invalid comparison score. Please use "r2" or "q2"')

    # Iterate through the sorted models list until you find one where the number of parameters is equal to the step number
    reference_model_index = 0
    while len(models_sorted[reference_model_index]) != step: # use the best model from the current step as reference
        reference_model_index += 1
    
    repeat_parameter_cutoff = min([max([round(step/3), 1]), 3]) # 1 for up to 4-term-models; 2 for 5 to 7 terms; 3 for 8+ terms

    unique_models = [models_sorted[reference_model_index]]
    for model in models_sorted:
        add = True

        # Ignore models with less than or equal to 2 or (step - 2) parameters
        if len(model) <= max([2, step-2]):
            add = False
        else:
            # If the curent model has more than {repeat_parameter_cutoff} parameters in common with a model already in {unique_models}, skip it
            for unique_model in unique_models:
                common_parameters = set(unique_model) & set(model)
                if len(common_parameters) > repeat_parameter_cutoff:
                    add = False
                    break

        # Otherwise, add it to the uniquemods list
        if add:      
            unique_models.append(model)
            
    return(unique_models)

def create_model(terms:tuple, data:pd.DataFrame, response:str, regression_type:type, usescore:str = 'r2'):
    """
    Creates a Model object from the data passed to it, then returns a bunch of stuff that's probably not necessary.
    Predominately used for parallel processing in the bidirectional_stepwise_regression function.

    :terms: Tuple of x# terms
    :data: Dataframe containing all training parameters and responses with x# parameter labels
    :response: Name of the response column in data
    :regression_type: The type of regressor to use
    :usescore: 'q2', 'r2'; What statistic to use in comparing models
    """
    # terms = tuple(terms)
    model = Model(terms, data.loc[:,terms], data[response], regression_type, usescore) 
    if usescore == 'q2':
        score = model.q2    
    elif usescore == 'r2':
        score = model.r2
    
    return(terms,model,score,response)

def calculate_q2(X:pd.DataFrame, y:pd.DataFrame, model:type=LinearRegression()):
    """
    Calculates the q^2 score for a given data set.
    Returns the score and the predictions.

    :X: Dataframe containing only parameters for the model
    :y: Dataframe containing the response variable
    :model: The type of regressor to use
    """
    loo = LeaveOneOut()
    ytests = []
    ypreds = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] #requires arrays
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train,y_train) 
        y_pred = model.predict(X_test)
            
        ytests += list(y_test)
        ypreds += list(y_pred)
            
    rr = metrics.r2_score(ytests, ypreds)
    return(rr,ypreds)

def q2_parallel(terms:tuple, X:pd.DataFrame, y:pd.DataFrame, regression_type:type) -> dict:
    """This is just a call to calculate_q2 set up for parallel processing with the returns needed in bidirectional_stepwise_regression."""
    # q2_score = loo.q2_df(X,y,regression_type())[0]
    q2_score = calculate_q2(X,y,regression_type())[0]
    return_dict = {
        'terms':terms,
        'q2_score':q2_score,
        }
    return(return_dict)

def bidirectional_stepwise_regression(data:pd.DataFrame, response_label:str, n_steps:int = 3, n_candidates:int = 30 , regression_type:type=LinearRegression, collinearity_cutoff:float = 0.5):
    """
    Does a bunch of stuff to put models together

    :data: Dataframe containing all training parameters and responses with x# parameter labels
    :response_label: Name of the response column in data
    :n_steps: Number of parameters in the largest models desired
    :n_candidates: Number used in determining how many models to carry through each step
    :regression_type: The type of regressor to use
    :collinearity_cutoff: parameters with an R^2 greater than this are considered collinear (and won't appear in the same model?)
    """
    start_time = time.time() # Set start to report how long the process takes
    pool = Parallel(n_jobs=nproc,verbose=0)

    # Pull the list of x# features and take out the response column
    features = list(data.columns)
    features.remove(response_label)
    
    # Set up the correlation_map and collinearity cutoff in a comparable way
    correlation_map = data.corr() # pearson correlation coefficient R: -1 ... 1
    collinearity_cutoff = np.sqrt(collinearity_cutoff) # convert from R2 to R

    # {models} is a dictionary of term tuples and their associated Model objects
    models = {}

    for step in [1,2]:
        print(f'Starting {step} parameter models. Total time taken (sec): %0.2f' %((time.time()-start_time)))

        # Create a list of tuples with all the parameter combos to be modeled in steps one and two
        if step == 1:
            todo = [(feature,) for feature in features] # todo is a list of single-element tuples, basically reformatting the features (x#) list
        if step == 2:
            # The itertools bit makes all possible 2-parameter tuples
            # The if statement filters out correlated 2-parameter tuples
            all_pairs = itertools.combinations(features,step)
            todo = sorted([(t1,t2) for (t1,t2) in all_pairs if abs(correlation_map.loc[t1,t2]) < collinearity_cutoff])   

        # Create a queue of calls to the create_model function to create models from terms, then run them in parallel
        # The {modeling_results} variable is a list of tuples containing the terms, the model object, the score, and the response for each feature combination
        modeling_results = pool(delayed(create_model)(terms, data, response_label, regression_type) for terms in todo)

        # Store information about the created models 
        for result in modeling_results:
            if len(result) == 0: # In what situation would this happen?
                continue
            models[result[0]] = result[1] # Expand the models dictionary with terms:Model

    # Calculate q^2 scores for the best models so far
    n_models = min([2*(len(features) + n_candidates), len(models)]) # Some math to determine how many models to calculate q^2 for
    best_r2_candidates = sorted(models, key=lambda terms: models[terms].r2, reverse=True)
    best_r2_candidates = best_r2_candidates[:n_models] # Trim the list of terms tuples down to just the best n_models
    best_r2_candidates = tuple(best_r2_candidates) # Convert the list of terms tuples to a big tuple of terms tuples

    modeling_results = pool(delayed(q2_parallel)(terms, data.loc[:,terms], data[response_label], regression_type) for terms in best_r2_candidates)

    for result in modeling_results:
        models[result['terms']].q2 = result['q2_score'] # Attach the q2 score to the associate Model object in the models dictionary

    # keep n best scoring models based on q^2
    n_models = n_candidates * step # Number of models to carry forward to the next step
    sorted_candidates = sorted(best_r2_candidates, key=lambda terms: models[terms].q2, reverse=True) # Sort the models dictionary by q^2 and return a list of terms tuples in order
    candidates = tuple(sorted_candidates[:n_models]) # Convert the top n models in the list into a tuple of best model terms tuples
    
    while step < n_steps:
        step += 1
        print(f'Starting {step} parameter models. Total time taken (sec): %0.2f' %((time.time()-start_time)))
        time_step = time.time()

        # Cycle through all parameter tuples that add one term to the existing list
        all_combinations = itertools.product(candidates,features) # Create a list of all possible term combinations of current models (cantidates) and one additiona feature
        todo = set([tuple(sorted(set(candidate_model+(additional_term,)))) for (candidate_model,additional_term) in all_combinations]) # Using set() makes it so that no model gets duplicated terms and we don't get duplicated models
        todo = [i for i in todo if i not in models.keys()] # Remove any term combinations that have already been modeled

        # Remove candidate term combinations from todo if any pair of terms {t1, t2} in it exceeds the collinearity cutoff
        todo = [candidate for candidate in todo if max([correlation_map.loc[t1,t2] for (t1, t2) in itertools.combinations(candidate,2)]) <= collinearity_cutoff]
        todo.sort()
        
        # Create a queue of calls to the create_model function to create models from terms, then run them in parallel
        modeling_results = pool(delayed(create_model)(terms,data,response_label,regression_type) for terms in todo)

        for result in modeling_results:
            if len(result) == 0:
                continue            
            models[result[0]] = result[1] # Expand the models dictionary with terms:Model     
 
        print(f'\tFinished running all {step} parameter models. Time taken (sec): %0.2f' %((time.time()-time_step)))
        time_step = time.time()

        #######################################################################
        # This bit seems like an overly complicated way to get the best models
        #######################################################################

        n_models = min([step * (len(features) + n_candidates), len(models)]) # Some math to determine how many models to bring forward
        best_r2_candidates = sorted(models, key=lambda model: models[model].r2, reverse=True)# Sort the models dictionary by r^2 and return a list of terms tuples in order
        best_r2_candidates = [terms for terms in best_r2_candidates if not hasattr(models[terms], 'q2')] # Limit best_r2_candidates to only models that q2 has not been calculated for
        best_r2_candidates = best_r2_candidates[:n_models] # Trim the list of terms-tuples down to just the best n_models

        # Get a second list of terms-tuples representing all unique terms combinations
        unique_candidates = filter_unique(models, step, comparison_score='r2')

        # Combine the best and unique lists into a single list of terms-tuples
        candidates_r2 = tuple(set(best_r2_candidates + unique_candidates))

        # Calculate q^2 in parallel for all models in candidates_r2
        parall = pool(delayed(q2_parallel)(terms, data.loc[:,terms], data[response_label], regression_type) for terms in candidates_r2)
        for results in parall:
            models[results['terms']].q2 = results['q2_score'] # Attach the q2 score to the associate Model object in the models dictionary

        # Run through the same logic, selecting by best q^2
        models_with_q2 = [model for model in models if hasattr(models[model], 'q2')]
        best_q2_candidates = sorted(models_with_q2, key=lambda terms: models[terms].q2, reverse=True)[:n_candidates*step]

        unique_candidates = filter_unique(models, step, comparison_score='q2')

        candidates = tuple(set(best_q2_candidates + unique_candidates))

        print(f'\tFinished identifying best and unique models. Time taken (sec): %0.2f' %((time.time()-time_step)))
        time_step = time.time()

        #######################################################################
        # From here to the end of the loop takes the majority (%75) of the run time
        #######################################################################

        # Iterate through all candidates and all terms combinations with one removed
        for candidate in candidates:
            for terms in itertools.combinations(candidate,len(candidate)-1):

                if terms == (): # Skip if empty
                    continue
                elif terms not in models.keys(): # If the model hasn't been seen yet, calculate it
                    models[terms] = Model(terms, data.loc[:,terms], data[response_label], regression_type)
                elif terms in models.keys() and not hasattr(models[terms], 'q2'): # If the model has already been seen but q^2 hasn't been calculated, do so
                    models[terms].q2 = loo.q2_df(data.loc[:,terms],data[response_label],regression_type())[0]

        # Select best models from this batch based on q^2
        models_with_q2 = [model for model in models if hasattr(models[model], 'q2')]
        best_q2_candidates = sorted(models_with_q2, key=lambda terms: models[terms].q2, reverse=True)[:n_candidates*step]

        unique_candidates = filter_unique(models, step, comparison_score='q2')

        candidates = tuple(set(best_q2_candidates + unique_candidates))

        print(f'\tFinished backwards step and filtering. Time taken (sec): %0.2f' %((time.time()-time_step)))
        time_step = time.time()

    models_with_q2 = [model for model in models if hasattr(models[model], 'q2')]
    sorted_models = sorted(models_with_q2, key=lambda terms: models[terms].q2, reverse=True)

    results_dict = {
        'Model': sorted_models,
        'n_terms': [models[terms].n_terms for terms in sorted_models],
        'R^2': [models[terms].r2 for terms in sorted_models],
        'Q^2': [models[terms].q2 for terms in sorted_models],
    }
    results = pd.DataFrame(results_dict)        
    print('Done. Time taken (minutes): %0.2f' %((time.time()-start_time)/60))
    return(results,models,sorted_models,candidates)        
            
# Still need to clean this one up
def repeated_k_fold(X_train,y_train,reg = LinearRegression(), k=3, n=100):
    """Reapeated k-fold cross-validation. 
    For each of n repeats, the (training)data is split into k folds. 
    For each fold, this part of the data is predicted using the rest. 
    Once this is done for all k folds, the coefficient of determination (R^2) of the predictions of all folds combined (= the complete data set) is evaluated
    This is repeated n times and all n R^2 are returned for averaging/further analysis
    """
    
    rkf = RepeatedKFold(n_splits=k, n_repeats=n)
    r2_scores = []
    y_validations,y_predictions = np.zeros((np.shape(X_train)[0],n)),np.zeros((np.shape(X_train)[0],n))
    foldcount = 0
    for i,foldsplit in enumerate(rkf.split(X_train)):
        fold, rep = i%k, int(i/k) # Which of k folds. Which of n repeats
        model = reg.fit(X_train[foldsplit[0]],y_train[foldsplit[0]]) # foldsplit[0]: k-1 training folds
        y_validations[foldcount:foldcount+len(foldsplit[1]),rep] = y_train[foldsplit[1]] # foldsplit[1]: validation fold
        y_predictions[foldcount:foldcount+len(foldsplit[1]),rep]  = model.predict(X_train[foldsplit[1]])
        foldcount += len(foldsplit[1])
        if fold+1==k:
            foldcount = 0
    r2_scores = np.asarray([metrics.r2_score(y_validations[:,rep],y_predictions[:,rep]) for rep in range(n)])
    return(r2_scores)

def external_r2(y_test_measured,y_test_predicted,y_train):
    """Calculates the external R2 pred as described:
    https://pdfs.semanticscholar.org/4eb2/5ff5a87f2fd6789c5b9954eddddfd1c59dab.pdf"""
    
    y_residual = y_test_predicted - y_test_measured
    SS_residual = np.sum(y_residual**2)
    y_varience = y_test_measured - np.mean(y_train)
    SS_total = np.sum(y_varience**2)
    r2_validation = 1-SS_residual/SS_total
    return(r2_validation)
