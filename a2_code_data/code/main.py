#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree

# our code
from utils import handle, load_dataset, main, plot_classifier, run


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    for k in [1, 3, 10]:
        knn = KNN(k)
        knn.fit(X, y)
        train_preds = knn.predict(X)
        test_preds = knn.predict(X_test)
        train_error = np.mean(train_preds != y)
        test_error = np.mean(test_preds != y_test)
        if k == 1:
            plot_classifier(knn, X, y)
            fname = Path("..", "figs", "q1_k=1.pdf")
            plt.savefig(fname)
            print("\nFigure saved as '%s'" % fname)

        print(
            f"For k = {k}, Training error: {train_error:.4f}, Test error: {test_error:.4f}"
        )
        sk_knn = KNeighborsClassifier(n_neighbors=3)
        sk_knn.fit(X, y)
        train_error = np.mean(sk_knn.predict(X) != y)
        test_error = np.mean(sk_knn.predict(X_test) != y_test)
        print(f"Training error: {train_error:.4f}, Test error: {test_error:.4f}")


@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))
    """YOUR CODE HERE FOR Q2"""
    num_folds = 10
    fold_size = len(X) // num_folds
    cv_accs = []
    test_accs = []
    train_accs = []

    for k in ks:
        accuracies = []
        for fold in range(num_folds):
            mask = np.ones(len(X), dtype=bool)
            mask[fold * fold_size : (fold + 1) * fold_size] = False

            X_train, y_train = X[mask], y[mask]
            X_val, y_val = X[~mask], y[~mask]

            knn = KNN(k)
            knn.fit(X_train, y_train)
            cv_preds = knn.predict(X_val)

            accuracy = np.mean(cv_preds == y_val)
            accuracies.append(accuracy)

        cv_accs.append(np.mean(accuracies))

        test_preds = knn.predict(X_test)
        test_ac = np.mean(y_test == test_preds)

        train_preds = knn.predict(X_train)
        train_ac = np.mean(y_train == train_preds)

        test_accs.append(test_ac)
        train_accs.append(train_ac)

    plt.figure(figsize=(9, 5))
    plt.plot(ks, cv_accs, "-o", label="Cross Validation Accuracy")
    plt.plot(ks, test_accs, "-x", label="Test Accuracy")
    plt.plot(ks, train_accs, "-o", label="Training Accuracy")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. k for kNN")
    plt.legend()
    plt.grid(True)
    fname = Path("..", "figs", "q2_plot.pdf")
    plt.savefig(fname)
    plt.show()


@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    """YOUR CODE HERE FOR Q3.2"""
    print(wordlist[72])
    indices = np.where(X[802] != False)[0]
    result = wordlist[indices]
    print(result)
    print(groupnames[y[802]])


@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    """CODE FOR Q3.4: Modify naive_bayes.py/NaiveBayesLaplace"""

    model = NaiveBayes(num_classes=2)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    """YOUR CODE HERE FOR Q3.4. Also modify naive_bayes.py/NaiveBayesLaplace"""

    model_laplace = NaiveBayesLaplace(
        num_classes=4, beta=1
    )  # Using beta=1 for Laplace smoothing
    model_laplace.fit(X, y)
    y_pred_laplace = model_laplace.predict(X_valid)
    accuracy_laplace = np.mean(y_pred_laplace == y_valid)
    print(f"Validation accuracy (Laplace-smoothed Naive Bayes): {accuracy_laplace:.4f}")


@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    """
    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
    """
    """YOUR CODE FOR Q4. Also modify random_tree.py/RandomForest"""
    evaluate_model(RandomTree(np.inf))
    evaluate_model(RandomForest(num_trees=50, max_depth=np.inf))


@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.1. Also modify kmeans.py/Kmeans"""
    raise NotImplementedError()


@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    """YOUR CODE HERE FOR Q5.2"""
    raise NotImplementedError()


if __name__ == "__main__":
    main()
