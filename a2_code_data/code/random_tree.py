from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    """
    YOUR CODE HERE FOR Q4
    Hint: start with the constructor __init__(), which takes the hyperparameters.
    Hint: you can instantiate objects inside fit().
    Make sure predict() is able to handle multiple examples.
    """

    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []


    def fit(self, X, y):
        n, d = X.shape
        
        for i in range(self.num_trees):
            cur_tree = RandomTree(self.max_depth)
            cur_tree.fit(X, y)  # RandomTree takes its bootstrap sample internally
            self.trees.append(cur_tree)



    def predict(self, X_pred):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X_pred))

        # transpose the prediction array to look at prediction for each example
        predictions_T = list(zip(*predictions))

        final_prediction = []
        for sample_prediction in predictions_T:
            final_prediction.append(utils.mode(np.array(sample_prediction)))
        return np.array(final_prediction)

