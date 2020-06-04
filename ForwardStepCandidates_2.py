# 19-07-17: include all "unique" models at each candidate selection step, as defined by having at most one common parameter with other unique models
#           fixed a bug in the collinearity criteria

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 
from sklearn import metrics
import loo_q2 as loo
import itertools
import time

import multiprocessing
nproc = max([1,multiprocessing.cpu_count()-2])
from joblib import Parallel,delayed

class Model:
    def __init__(self,terms,X,y,reg,usescore='q2'):
        self.terms = terms
        self.n_terms = len(terms)
        self.formula = '1 + ' + ' + '.join(terms)
        self.model = reg.fit(X,y)
        self.r2 = self.model.score(X,y)
        if usescore=='q2':
            self.q2 = loo.q2_df(X,y,reg)[0]
        
                
reg = LinearRegression()        
n_steps = 3        
n_candidates = 30

def filter_unique(scores,step):
    models_sorted = sorted(scores,key=scores.get,reverse=True)
    refmodel = 0
    while len(models_sorted[refmodel]) != step: # use the best model from the current step as reference
        refmodel += 1
    cutpar = min([max([round(step/3),1]),3]) # 1 for up to 4-term-models; 2 for 5 to 7 terms; 3 for 8+ terms
    print("cutpar: ",cutpar)
    uniquemods = [models_sorted[refmodel]]
    for selmod in models_sorted:
        if len(selmod) <= max([2,step-2]):
            continue
        add = True
        for mod in uniquemods:
            if len([i for i in mod if i in selmod]) > cutpar:
                add = False
                break
        if add:      
            uniquemods.append(selmod)
    return(uniquemods)

def corrmap_par(t1,t2,data):
    i,j = data[t1].values.reshape(-1,1),data[t2].values.reshape(-1,1)
    t1t2corr = LinearRegression().fit(i,j)
    score = t1t2corr.score(i,j)
    return(t1,t2,score)

def step_par(terms,data,response,reg,usescore='r2'):
    #todo: implement checks for p-value of added term
    terms = tuple(terms)
    model = Model(terms,data.loc[:,terms],data[response],reg,usescore) 
    if usescore == 'q2':
        score = model.q2    
    elif usescore == 'r2':
        score = model.r2
    # implement weighted average of several scores per model    
    return(terms,model,score,response)

def q2_par(terms,X,y,reg):
    cand_q2 = loo.q2_df(X,y,reg)[0]
    return(terms,cand_q2)

def ForwardStep_py(data,response,n_steps=3,n_candidates=30,reg=LinearRegression(),collin_criteria=0.5):
    start_time = time.time()
    pool = Parallel(n_jobs=nproc,verbose=0)
        
    # data: pd.dataframe with all features and a response column
    # it is advised to have the column titles as x1...xn
    # response: string of the response column name in data
    features = list(data.columns)
    features.remove(response)
    
    corrmap = data.corr() # pearson correlation coefficient R: -1 ... 1
    collin_criteria = np.sqrt(collin_criteria) # convert from R2 to R
#     univars = corrmap.drop("y",axis=0)["y"]**2
#     models = univars.to_dict()
#     scores_r2 = univars.to_dict()
    models,scores_r2,scores_q2 = {},{},{}

    for step in [1,2]:
        print("Step " + str(step))
        if step == 1:
            todo = [(feature,) for feature in features]
        if step == 2:
            todo = sorted([(t1,t2) for (t1,t2) in itertools.combinations(features,step) if abs(corrmap.loc[t1,t2]) < collin_criteria])   

        parall = pool(delayed(step_par)(terms,data,response,reg) for terms in todo)
        for results in parall:
            if len(results) == 0:
                continue
            models[results[0]] = results[1]
            scores_r2[results[0]] = results[2]

    # calculate Q^2 only for the highest-correlating models to save time
    candidates_r2 = tuple(sorted(scores_r2,key=scores_r2.get,reverse=True)[:min([2*(len(features)+n_candidates),len(scores_r2)])])
    parall = pool(delayed(q2_par)(terms,data.loc[:,terms],data[response],reg) for terms in candidates_r2)
    for results in parall:
        models[results[0]].q2 = results[1]
        scores_q2[results[0]] = results[1]
    ## non-parallel:
    # for cand_r2 in candidates_r2:
    #     cand_q2 = loo.q2_df(data.loc[:,cand_r2],data[response],reg)[0]
    #     models[cand_r2].q2 = cand_q2
    #     scores_q2[cand_r2] = cand_q2

    print('Finished 1 and 2 parameter models. Time taken (sec): %0.4f' %((time.time()-start_time)))

    # keep n best scoring models
    candidates = tuple(sorted(scores_q2,key=scores_q2.get,reverse=True)[:n_candidates*step])
    # print(len(models.keys()))
        
    while step < n_steps:
        step += 1
        print("Step " +str(step))
        # add 1 term
        todo_rem = []
        todo = set([tuple(sorted(set(candidate+(term,)))) for (candidate,term) in itertools.product(candidates,features)])
        todo = [i for i in todo if i not in models.keys()]
        for newcandidate in todo:
            collin = max([corrmap.loc[t1,t2] for (t1, t2) in itertools.combinations(newcandidate,2)])
            if collin > collin_criteria:
                todo_rem.append(newcandidate)
        todo = sorted([i for i in todo if i not in todo_rem])
        
        # print(len(todo))
        parall = pool(delayed(step_par)(terms,data,response,reg) for terms in todo)

        for results in parall:
            if len(results) == 0:
                continue            
            models[results[0]] = results[1]
            scores_r2[results[0]] = results[2]            
        
            #implement checks for p-value of added term

        cands_a = [i for i in sorted(scores_r2,key=scores_r2.get,reverse=True) if i not in scores_q2.keys()][:min([step*(len(features)+n_candidates),len(scores_r2)])]
        cands_b = filter_unique(scores_r2,step)
        candidates_r2 = tuple(set(cands_a+cands_b))
        # print(len(cands_a),len(cands_b),len(candidates_r2))
        parall = pool(delayed(q2_par)(terms,data.loc[:,terms],data[response],reg) for terms in candidates_r2)
        for results in parall:
            models[results[0]].q2 = results[1]
            scores_q2[results[0]] = results[1]
        cands_a = sorted(scores_q2,key=scores_q2.get,reverse=True)[:n_candidates*step]
        cands_b = filter_unique(scores_q2,step)
        candidates = tuple(set(cands_a+cands_b))
        # print(len(cands_a),len(cands_b),len(candidates))

        # print(len(models.keys()))    
        # remove 1 term
        for candidate in candidates:
            for test in itertools.combinations(candidate,len(candidate)-1):
                if test == ():
                    continue
                terms = test
                if terms in scores_q2.keys():
                    continue
                elif terms in models.keys():
                    cand_q2 = loo.q2_df(data.loc[:,terms],data[response],reg)[0]
                    models[terms].q2 = cand_q2
                    scores_q2[terms] = cand_q2  
                models[terms] = Model(terms,data.loc[:,terms],data[response],reg) 
                scores_r2[terms] = models[terms].r2      
                scores_q2[terms] = models[terms].q2      
        cands_a = sorted(scores_q2,key=scores_q2.get,reverse=True)[:n_candidates*step]
        cands_b = filter_unique(scores_q2,step)
        candidates = tuple(set(cands_a+cands_b))
        # print(len(cands_a),len(cands_b),len(candidates))

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
            
