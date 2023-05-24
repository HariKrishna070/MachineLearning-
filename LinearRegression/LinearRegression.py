"""
    Linear Regression
"""

import numpy as np

class LinearRegression:
    def __init__(self,l_rate=0.01,n_iter = 300):
        self.l_rate = l_rate
        self.n_iter = n_iter
        self.slope = None
        self.intercept = None
        
    def fit(self,X_train,y_train):
        n_samples,n_features = X_train.shape
        self.slope = np.zeros(n_features)
        self.intercept = 0
        for i in range(self.n_iter):
            y_cap = np.dot(X_train,self.slope) + self.intercept
        
            w = 1/(n_samples) * np.dot(X_train.T,(y_cap - y_train))
            b = 1/(len(X_train)) * np.sum(y_cap - y_train)
            
            self.slope = self.slope - self.l_rate * w  
            self.intercept = self.intercept - self.l_rate * b

    def predict(self,X_test):
        return [x*self.slope + self.intercept for x in X_test]
    
    def values(self):
        return f"slope : {self.slope}\nintercept : {self.intercept}"
