from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
from dtreeviz.trees import dtreeviz # pip install dtreeviz

class dt():
    def __init__(self):
        self.except_message = "Error:"
        self.reset_tree_attributes()
        
    def reset_tree_attributes(self):
        self.tree = None
        self.X = None
        self.Y = None
        self.train_X = None 
        self.test_X = None
        self.train_y = None
        self.test_y = None
        self.model = None
        self.mae = None
        self.y_predict = None
        
    def delete_tree(self):
        self.reset_tree_attributes()
        print("tree deleted")
        
    def plot_tree(self):
        if self.tree.node_count > 20:
            plt.figure(figsize=(20,20))
        else:
            plt.figure()
        tree.plot_tree(self.model, filled=True) 
        
    def plot_thresholds(self):
        plot_colors = "rg"
        plot_step = 0.02
        x_min, x_max = self.X.min(), self.X.max()
        y_min, y_max = self.y.min(), self.y.max()
        dx,dy = x_max-x_min,y_max-y_min
        xx, yy = np.meshgrid(np.arange(x_min-0.04*dx, x_max+0.04*dx, plot_step),
                             np.arange(y_min-0.04*dy, y_max+0.04*dy, plot_step))

        plt.figure(figsize=(4, 4))    
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = self.model.predict(xx.ravel().reshape(-1, 1))
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, cmap="Pastel1")

        plt.ylabel("y")  # "$ΔΔG^{≠}$"  "Yield"
        plt.scatter(self.X, self.y, c='b',cmap=plt.cm.RdYlBu, s=15) 
        plt.show()


class decision_tree_regressor(dt):
    def __init__(self):
        dt.__init__(self)
        
    '''
    Uses sklearn to split the data, fits a single node decision tree regressor, then calculates MAE
    '''
    def auto_decision_tree(self, X, y, plot = True):
        self.reset_tree_attributes()
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, random_state = 1)
        self.model = DecisionTreeRegressor(random_state=0)
        self.model.fit(self.train_X, self.train_y)
        self.y_predict = self.model.predict(self.test_X)
        self.mae = mean_absolute_error(self.test_y, self.y_predict)
        self.tree = self.model.tree_
        print("Accuracy: {:.2f}".format(self.model.score(X, y)))
        if plot:
            self.plot_tree()
        return self.mae
     
    '''
    Uses split the data passed in by the user, fits num_nodes decision tree regressor, then calculates MAE
    '''
    def decision_tree(self, X, y, num_nodes, plot = True):
        self.reset_tree_attributes()
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, random_state = 1)
        self.model = DecisionTreeRegressor(max_leaf_nodes=num_nodes, random_state=0)
        self.model.fit(self.train_X, self.train_y)
        self.y_predict = self.model.predict(self.test_X)
        self.mae = mean_absolute_error(self.test_y, self.y_predict)
        self.tree = self.model.tree_        
        print("Accuracy: {:.2f}".format(self.model.score(X, y)))
        if plot:
            self.plot_tree()        
        return self.mae
    
    '''
    Uses split the data passed in by the user, fits a single node decision tree regressor, then calculates MAE
    '''
    def single_node_decision_tree(self, X, y, plot=True):
        self.reset_tree_attributes()
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, random_state = 1)
        self.model = DecisionTreeRegressor(max_depth=1, random_state=0)
        self.model.fit(self.train_X, self.train_y)
        self.y_predict = self.model.predict(self.test_X)
        self.mae = mean_absolute_error(self.test_y, self.y_predict)
        self.tree = self.model.tree_        
        print("Accuracy: {:.2f}".format(self.model.score(X, y)))
        if plot:
            self.plot_tree()
        return self.mae    
    
'''
Decision tree parent class saves the most recent tree generated 
'''    
class decision_tree_classifier(dt):
    def __init__(self):
        dt.__init__(self)  
        
    def __split_y(self):
        y_hist,y_bin_edges = np.histogram(self.y,bins="auto")
        y0 = np.asarray([0 if i < y_bin_edges[1] else 1 for i in self.y])
        y1 = np.asarray([1 if i > y_bin_edges[-2] else 0 for i in self.y])
        return [i for binary in [y0, y1] for i in binary]

    def __cut_y(self, cut):
        return np.array([0 if i < cut else 1 for i in self.y])


    '''
    Splits y into min and max values, for classification, if its values are not binary 
    '''
    def __getY(self, y, split, cut):
        if split:
            self.y = self.__split_y()
        elif cut is not None:
            self.y = self.__cut_y(cut)
        else:
            self.y = np.array(y)
        
    '''
    Uses sklearn to split the data, fits a single node decision tree regressor, then calculates MAE
    '''
    def auto_decision_tree(self, X, y, split = False, cut: int = None, plot = True):
        self.reset_tree_attributes()
        self.X = np.asarray(X)
        self.__getY(y, split, cut)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, random_state = 1)
        self.model = DecisionTreeClassifier(random_state=0)
        self.model.fit(self.train_X, self.train_y)
        self.y_predict = self.model.predict(self.test_X)
        self.mae = mean_absolute_error(self.test_y, self.y_predict)
        self.tree = self.model.tree_
        print("Accuracy: {:.2f}".format(self.model.score(self.X, self.y)))
        if plot:
            self.plot_tree()        
        return self.mae
     
    '''
    Uses split the data passed in by the user, fits num_nodes decision tree regressor, then calculates MAE
    '''
    def decision_tree(self, X, y, num_nodes, plot = True):
        self.reset_tree_attributes()
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, random_state = 1)
        self.model = DecisionTreeClassifier(max_leaf_nodes=num_nodes, random_state=0)
        self.model.fit(self.train_X, self.train_y)
        self.y_predict = self.model.predict(self.test_X)
        self.mae = mean_absolute_error(self.test_y, self.y_predict)
        self.tree = self.model.tree_    
        print("Accuracy: {:.2f}".format(self.model.score(self.X, self.y)))     
        if plot:
            self.plot_tree()        
        return self.mae
    
    


        
    
    

