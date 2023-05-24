"""
    Linear Regression
"""
# importing important modules
import numpy as np 

# linear regression from scratch
class LinearRegression:
    def __init__(self,l_rate=0.01,n_iter = 300):  # initializing hyperameters i.e, learning rate as l_rate and number of iterations should be done as n_iter
        self.l_rate = l_rate
        self.n_iter = n_iter
        
    # training model and finding slope and intercept of the linear line which fits the given data
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
    
    # pridicting the values of required data
    def predict(self,X_test):
        return [x*self.slope + self.intercept for x in X_test]
    
    # method to know the values of slope and intercept of the linear line
    def values(self):
        return f"slope : {self.slope}\nintercept : {self.intercept}"
    
    
