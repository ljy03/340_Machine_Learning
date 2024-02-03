import numpy as np


class NaiveBayes:
    """
    Naive Bayes implementation.
    Assumes the feature are binary.
    Also assumes the labels go from 0,1,...k-1
    """

    p_y = None
    p_xy = None

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n
        
        p_xy = np.zeros((d, k))

        for c in range(k):
            class_indices = np.where(y == c)[0]
            X_class = X[class_indices]
            feature_sums = np.sum(X_class, axis=0)
            p_xy[:, c] = feature_sums / len(class_indices)


        self.p_y = p_y
        self.p_xy = p_xy

    def predict(self, X):
        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(n)
        for i in range(n):

            probs = p_y.copy()  # initialize with the p(y) terms
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= 1 - p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred


class NaiveBayesLaplace(NaiveBayes):
    def __init__(self, num_classes, beta=0):
        super().__init__(num_classes)
        self.beta = beta

    def fit(self, X, y):
        n, d = X.shape

    # Compute the number of class labels
        k = self.num_classes
        counts = np.bincount(y, minlength=k)
        p_y = counts / n

        # Compute the conditional probabilities with Laplace smoothing
        p_xy = np.zeros((d, k))

        for c in range(k):
            class_indices = np.where(y == c)[0]
            X_class = X[class_indices]
            for j in range(d):
                feature_sum = np.sum(X_class[:, j])
                p_xy[j, c] = (feature_sum + self.beta) / (len(class_indices) + self.beta * 2)

        self.p_y = p_y
        self.p_xy = p_xy