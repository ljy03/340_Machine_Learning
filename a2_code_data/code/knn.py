"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        num_test = X_hat.shape[0]
        y_pred = np.zeros(num_test)
        
        distance = euclidean_dist_squared(self.X,X_hat)

        for i in range(num_test):
            k_nearest_indices = np.argsort(distance[:,i])[:self.k]
            k_nearest_labels = self.y[k_nearest_indices]
            y_pred[i] = np.bincount(k_nearest_labels).argmax()
            
        return y_pred