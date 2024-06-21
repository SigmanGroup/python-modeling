from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn import metrics
import loo_q2 as loo
import itertools
import time

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
        if usescore=='q2':
            self.q2 = loo.q2_df(X,y,regression_type())[0]


def filter_unique(scores:dict, step:int):
    """
    Sort through the models in the scores dictionary and return a list of terms-tuples that have less than cutpar parameters in common
    """
    models_sorted = sorted(scores, key=scores.get, reverse=True)
    
    # Iterate through the sorted models list until you find one where the number of parameters is equal to the step number
    refmodel = 0
    while len(models_sorted[refmodel]) != step: # use the best model from the current step as reference
        refmodel += 1
        # print(f'Refmodel: {refmodel}')
    
    cutpar = min([max([round(step/3), 1]), 3]) # 1 for up to 4-term-models; 2 for 5 to 7 terms; 3 for 8+ terms
    # print("cutpar: ",cutpar)

    uniquemods = [models_sorted[refmodel]]
    for selmod in models_sorted:

        # Ignore models with less than 2 or step - 2 parameters
        if len(selmod) <= max([2,step-2]):
            continue

        # If the curent model (selmod) has more than cutpar parameters in common with a model already in uniquemods, skip it
        add = True
        for mod in uniquemods:
            temp = [i for i in mod if i in selmod]
            if len(temp) > cutpar:
                add = False
                break

        # Otherwise, add it to the uniquemods list
        if add:      
            uniquemods.append(selmod)
    return(uniquemods)

def step_par(terms:tuple, data:pd.DataFrame, response:str, regression_type, usescore:str = 'r2'):
    """
    Creates a Model object from the data passed to it, then returns a bunch of stuff that's probably not necessary.

    :terms: Tuple of x# terms
    :data: Dataframe containing all training parameters and responses with x# parameter labels
    :response: Name of the response column in data
    :regression_type: The type of regressor to use
    :usescore: 'q2', 'r2'; What statistic to use in comparing models
    """
    #todo: implement checks for p-value of added term
    terms = tuple(terms)
    model = Model(terms, data.loc[:,terms], data[response], regression_type,usescore) 
    if usescore == 'q2':
        score = model.q2    
    elif usescore == 'r2':
        score = model.r2
    # implement weighted average of several scores per model    
    return(terms,model,score,response)

def q2_par(terms,X,y,regression_type):
    """This is just a call to loo.q2_df. Why even have this function?"""
    cand_q2 = loo.q2_df(X,y,regression_type())[0]
    return(terms,cand_q2)

# Lots of variable should be renamed
# It looks like when a model is created it calculates q^2, so why do we need to set up additional q^2 calculations?
def ForwardStep_py(data:pd.DataFrame, response:str, n_steps:int = 3, n_candidates:int = 30 , regression_type=LinearRegression, collin_criteria:float = 0.5):
    """
    Does a bunch of stuff to put models together

    :data: Dataframe containing all training parameters and responses with x# parameter labels
    :response: Name of the response column in data
    :n_steps: Number of parameters in the largest models desired
    :n_candidates: Number of models to carry through each step
    :regression_type: The type of regressor to use
    :collin_criteria: parameters with an R^2 greater than this are considered collinear (and won't appear in the same model?)
    """
    start_time = time.time() # Set start to report how long the process takes
    pool = Parallel(n_jobs=nproc,verbose=0)

    # Pull the list of x# features and take out the response column label
    features = list(data.columns)
    features.remove(response)
    
    # Set up the corrmap and collin critera in a comparable way
    corrmap = data.corr() # pearson correlation coefficient R: -1 ... 1
    collin_criteria = np.sqrt(collin_criteria) # convert from R2 to R

#     univars = corrmap.drop("y",axis=0)["y"]**2
#     models = univars.to_dict()
#     scores_r2 = univars.to_dict()

    # Initialize some empty dictionaries
    models,scores_r2,scores_q2 = {},{},{}

    for step in [1,2]:
        print(f"Step {step}")

        # Create a list of tuples with all the parameter combos to be modeled in steps one and two
        if step == 1:
            todo = [(feature,) for feature in features] # todo is a list of single-element tuples, basically reformatting the features (x#) list
        if step == 2:
            # The itertools bit makes all possible 2-parameter tuples
            # The if statement filters out correlated 2-parameter tuples
            all_pairs = itertools.combinations(features,step)

            todo = sorted([(t1,t2) for (t1,t2) in all_pairs if abs(corrmap.loc[t1,t2]) < collin_criteria])   

        # Create a queue of calls to the step_par function to create models from terms, then run them in parallel
        parall = pool(delayed(step_par)(terms, data, response, regression_type) for terms in todo)

        # Store information about the created models 
        for results in parall:
            if len(results) == 0:
                continue
            models[results[0]] = results[1] # Expand the models dictionary with terms:Model
            scores_r2[results[0]] = results[2] # Expand the scores_r2 dictionary with terms:score

    # Calculate q^2 scores for the best models so far
    #candidates_r2 = tuple(sorted(scores_r2,key=scores_r2.get,reverse=True)[:min([2*(len(features)+n_candidates),len(scores_r2)])])
    n_models = min([2*(len(features) + n_candidates), len(scores_r2)]) # Some math to determine how many models to calculate q^2 for
    candidates_r2 = sorted(scores_r2, key=scores_r2.get, reverse=True) # Sort the scores_r2 dictionary by r^2 and return a list of terms tuples in order
    candidates_r2 = candidates_r2[:n_models] # Trim the list of terms tuples down to just the best n_models
    candidates_r2 = tuple(candidates_r2) # Convert the list of terms tuples to a big tuple of terms tuples

    parall = pool(delayed(q2_par)(terms, data.loc[:,terms], data[response], regression_type) for terms in candidates_r2)
    for results in parall:
        models[results[0]].q2 = results[1] # Attach the q2 score to the associate Model object in the models dictionary
        scores_q2[results[0]] = results[1] # Expand the scores_q2 dictionary with terms:score

    print('Finished 1 and 2 parameter models. Total time taken (sec): %0.4f' %((time.time()-start_time)))

    # keep n best scoring models based on q^2
    #candidates = tuple(sorted(scores_q2, key=scores_q2.get, reverse=True)[:n_candidates*step])
    n_models = n_candidates * step # Number of models to carry forward to the next step
    sorted_candidates = sorted(scores_q2, key=scores_q2.get, reverse=True) # Sort the scores_q2 dictionary by q^2 score
    candidates = tuple(sorted_candidates[:n_models]) # Convert the top n model list into a tuple of best model terms tuples
    
    while step < n_steps:
        step += 1
        # print("Step " +str(step))
        print(f'Starting {step} parameter models. Total time taken (sec): %0.4f' %((time.time()-start_time)))
        time_step = time.time()

        # Cycle through all parameter tuples that add one term to the existing list
        todo_rem = []
        todo = set([tuple(sorted(set(candidate+(term,)))) for (candidate,term) in itertools.product(candidates,features)]) # Using set() makes it so that no model gets duplicated terms
        todo = [i for i in todo if i not in models.keys()] # Remove any term combinations that have already come through

        # Remove candidate term combinations from todo if any of the terms exceed the collinearity cap
        for newcandidate in todo:
            collin = max([corrmap.loc[t1,t2] for (t1, t2) in itertools.combinations(newcandidate,2)])
            if collin > collin_criteria:
                todo_rem.append(newcandidate)
        todo = sorted([i for i in todo if i not in todo_rem])
        
        # Create a queue of calls to the step_par function to create models from terms, then run them in parallel
        parall = pool(delayed(step_par)(terms,data,response,regression_type) for terms in todo)

        print(f'\tFinished running all {step} parameter models. Time taken (sec): %0.4f' %((time.time()-time_step)))
        time_step = time.time()

        for results in parall:
            if len(results) == 0:
                continue            
            models[results[0]] = results[1] # Expand the models dictionary with terms:Model
            scores_r2[results[0]] = results[2] # Expand the scores_r2 dictionary with terms:score     
        
            #implement checks for p-value of added term

        n_models = min([step * (len(features) + n_candidates), len(scores_r2)]) # Some math to determine how many models to bring forward
        sorted_candidates = sorted(scores_r2, key=scores_r2.get, reverse=True) # Sort the scores_r2 dictionary by r^2 and return a list of terms tuples in order
        cands_a = [i for i in sorted_candidates if i not in scores_q2.keys()] # Limit cands_a to only models that did not appear in the last round
        cands_a = cands_a[:n_models] # Trim the list of terms-tuples down to just the best n_models

        # Get a second list of terms-tuples representing all unique terms combinations
        cands_b = filter_unique(scores_r2, step)

        candidates_r2 = tuple(set(cands_a + cands_b))

        # Calculate q^2 in parallel for all models in candidates_r2
        parall = pool(delayed(q2_par)(terms, data.loc[:,terms], data[response], regression_type) for terms in candidates_r2)
        for results in parall:
            models[results[0]].q2 = results[1] # Attach the q2 score to the associate Model object in the models dictionary
            scores_q2[results[0]] = results[1] # Expand the scores_q2 dictionary with terms:score

        # Run through the same logic, selecting by best q^2
        cands_a = sorted(scores_q2, key=scores_q2.get, reverse=True)[:n_candidates*step]
        cands_b = filter_unique(scores_q2,step)
        candidates = tuple(set(cands_a+cands_b))

        print(f'\tFinished identifying best and unique models. Time taken (sec): %0.4f' %((time.time()-time_step)))
        time_step = time.time()

        # remove 1 term
        for candidate in candidates:

            # Iterate through all candidates and all terms combinations with one removed
            for test in itertools.combinations(candidate,len(candidate)-1):
                if test == (): # Skip if empty
                    continue
                terms = test # Decide you don't like what you called a variable, but you also don't want to change it in the for loop line
                if terms in scores_q2.keys(): # Skip if the new combination is already in the best models list
                    continue
                elif terms in models.keys(): # If the model has already been seen but q^2 hasn't been calculated, do so
                    cand_q2 = loo.q2_df(data.loc[:,terms],data[response],regression_type())[0]
                    models[terms].q2 = cand_q2
                    scores_q2[terms] = cand_q2

                # Calculate model
                models[terms] = Model(terms, data.loc[:,terms], data[response], regression_type) 
                scores_r2[terms] = models[terms].r2      
                scores_q2[terms] = models[terms].q2   

        cands_a = sorted(scores_q2, key=scores_q2.get, reverse=True)[:n_candidates*step]
        cands_b = filter_unique(scores_q2,step)
        candidates = tuple(set(cands_a + cands_b))

        print(f'\tFinished backwards step and filtering. Time taken (sec): %0.4f' %((time.time()-time_step)))
        time_step = time.time()

    sortedmodels = sorted(scores_q2,key=scores_q2.get,reverse=True)
    results_d = {
        'Model': sortedmodels,
        'n_terms': [models[terms].n_terms for terms in sortedmodels],
        'R^2': [models[terms].r2 for terms in sortedmodels],
        'Q^2': [models[terms].q2 for terms in sortedmodels],
    }
    results = pd.DataFrame(results_d)        
    print('Done. Time taken (minutes): %0.2f' %((time.time()-start_time)/60))
    return(results,models,scores_q2,sortedmodels,candidates)        
            
