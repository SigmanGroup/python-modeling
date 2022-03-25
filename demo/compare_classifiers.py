import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
#our classes
from preprocessing import data_prep
from split_data import split_data

class classifiers:
    def __init__(self, X, y):
        self.except_message = "Error:"
        self.X = StandardScaler().fit_transform(X)
        self.Y = Y
        self.plot_inx = 0
        
    def split(self):
        split_methods = split_data()
        X_train, X_test, y_train, y_test = split_methods.quarter_split(self.X, self.Y)
        return X_train, X_test, y_train, y_test
        
    def decision_tree(self):
        X_train, X_test, y_train, y_test = self.split()
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        score = dt.score(X_test, y_test)
        
    def plot_data(self, X_train, X_test, y_train, y_test, plot_classifiers = False):
        cmap = plt.cm.RdBu
        ax = plt.subplot(1, num_classifiers + 1, self.plot_inx)
        ax.set_title("Input data")
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        self.plot_inx += 1        