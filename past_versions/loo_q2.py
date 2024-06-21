import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut  
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut,cross_val_predict

def q2(X,y,model=LinearRegression()):

    loo = LeaveOneOut()
    ytests = []
    ypreds = []
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx] #requires arrays
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train,y_train) 
        y_pred = model.predict(X_test)
            
        ytests += list(y_test)
        ypreds += list(y_pred)
            
    rr = metrics.r2_score(ytests, ypreds)
    return(rr,ypreds)

def q2_df(X,y,model=LinearRegression()):

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

def q2_cv(X,y,model=LinearRegression()):
    ypreds = cross_val_predict(model,X,y,cv=LeaveOneOut())
    rr = metrics.r2_score(y,ypreds)
    return(rr,ypreds)

