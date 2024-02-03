#!/usr/bin/env python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from encoders import PCAEncoder
from kernels import GaussianRBFKernel, LinearKernel, PolynomialKernel
from linear_models import (
    LinearModel,
    LinearClassifier,
    KernelClassifier,
)
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    StochasticGradient,
)
from fun_obj import (
    LeastSquaresLoss,
    LogisticRegressionLossL2,
    KernelLogisticRegressionLossL2,
)
from learning_rate_getters import (
    ConstantLR,
    InverseLR,
    InverseSqrtLR,
    InverseSquaredLR,
)
from utils import (
    load_dataset,
    load_trainval,
    load_and_split,
    plot_classifier,
    savefig,
    standardize_cols,
    handle,
    run,
    main,
)


@handle("1")
def q1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # Standard (regularized) logistic regression
    loss_fn = LogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    lr_model = LinearClassifier(loss_fn, optimizer)
    lr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(lr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(lr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(lr_model, X_train, y_train)
    savefig("logRegPlain.png", fig)

    # kernel logistic regression with a linear kernel
    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    kernel = LinearKernel()
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegLinear.png", fig)


@handle("1.1")
def q1_1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # Kernel logistic regression with a Polynomial Kernel
    loss_fn = KernelLogisticRegressionLossL2(0.01)  # Regularization strength lambda = 0.01
    optimizer = GradientDescentLineSearch()
    poly_kernel = PolynomialKernel(p=2)  # Polynomial kernel with degree p=2
    poly_klr_model = KernelClassifier(loss_fn, optimizer, poly_kernel)
    poly_klr_model.fit(X_train, y_train)

    print(f"Polynomial Kernel - Training error: {np.mean(poly_klr_model.predict(X_train) != y_train):.1%}")
    print(f"Polynomial Kernel - Validation error: {np.mean(poly_klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(poly_klr_model, X_train, y_train)
    savefig("logRegPolynomial.png", fig)

    # Kernel logistic regression with a Gaussian RBF Kernel
    loss_fn = KernelLogisticRegressionLossL2(0.01)  # Regularization strength lambda = 0.01
    optimizer = GradientDescentLineSearch()
    rbf_kernel = GaussianRBFKernel(sigma=0.5)  # Gaussian RBF kernel with sigma=0.5
    rbf_klr_model = KernelClassifier(loss_fn, optimizer, rbf_kernel)
    rbf_klr_model.fit(X_train, y_train)

    print(f"Gaussian RBF Kernel - Training error: {np.mean(rbf_klr_model.predict(X_train) != y_train):.1%}")
    print(f"Gaussian RBF Kernel - Validation error: {np.mean(rbf_klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(rbf_klr_model, X_train, y_train)
    savefig("logRegGaussianRBF.png", fig)


@handle("1.2")
def q1_2():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    sigmas = 10.0 ** np.array([-2, -1, 0, 1, 2])
    lammys = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])

    # train_errs[i, j] should be the train error for sigmas[i], lammys[j]
    train_errs = np.full((len(sigmas), len(lammys)), 100.0)
    val_errs = np.full((len(sigmas), len(lammys)), 100.0)  # same for val

    best_train_err = np.inf
    best_sigma_train = sigmas[0]
    best_sigma_val = sigmas[0]
    best_validation_err = np.inf
    best_lambda_train = lammys[0]
    best_lambda_val = lammys[0]
    for i in range(len(sigmas)):
        for j in range(len(lammys)):
            loss_fn = KernelLogisticRegressionLossL2(lammys[j])
            optimizer = GradientDescentLineSearch()
            kernel = GaussianRBFKernel(sigmas[i])
            model = KernelClassifier(loss_fn, optimizer, kernel)
            model.fit(X_train, y_train)

            E_train = np.mean(model.predict(X_train) != y_train)
            E_val = np.mean(model.predict(X_val) != y_val)
            train_errs[i, j] = E_train
            val_errs[i, j] = E_val

            if E_train <= best_train_err:
                best_train_err = E_train
                best_sigma_train = sigmas[i]
                best_lambda_train = lammys[j]
            if E_val <= best_validation_err:
                best_validation_err = E_val
                best_sigma_val = sigmas[i]
                best_lambda_val = lammys[j]

    print(f"Best sigma for training err: {best_sigma_train}")
    print(f"Best lambda for training err: {best_lambda_train}")
    print(f"Best training err: {best_train_err}")
    print(f"Best sigma for validation err: {best_sigma_val}")
    print(f"Best lambda for validation err: {best_lambda_val}")
    print(f"Best validation err: {best_validation_err}")

    # plot the best classifier on training error:
    loss_fn = KernelLogisticRegressionLossL2(best_lambda_train)
    optimizer = GradientDescentLineSearch()
    kernel = GaussianRBFKernel(best_sigma_train)
    model = KernelClassifier(loss_fn, optimizer, kernel)
    model.fit(X_train, y_train)

    fig = plot_classifier(model, X_train, y_train)
    savefig("q1.2BestTrain.png", fig)

    # plot the best classifier on validation error:
    loss_fn = KernelLogisticRegressionLossL2(best_lambda_val)
    optimizer = GradientDescentLineSearch()
    kernel = GaussianRBFKernel(best_sigma_val)
    model = KernelClassifier(loss_fn, optimizer, kernel)
    model.fit(X_train, y_train)

    fig = plot_classifier(model, X_train, y_train)
    savefig("q1.2BestValidation.png", fig)
    # Make a picture with the two error arrays. No need to worry about details here.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))
    for (name, errs), ax in zip([("training", train_errs), ("val", val_errs)], axes):
        cax = ax.matshow(errs, norm=norm)

        ax.set_title(f"{name} errors")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(sigma) for sigma in sigmas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(range(len(lammys)))
        ax.set_xticklabels([str(lammy) for lammy in lammys])
        ax.xaxis.set_ticks_position("bottom")
    fig.colorbar(cax)
    savefig("logRegRBF_grids.png", fig)


@handle("3.2")
def q3_2():
    data = load_dataset("animals.pkl")
    X_train = data["X"]
    animal_names = data["animals"]
    trait_names = data["traits"]

    # Standardize features
    X_train_standardized, mu, sigma = standardize_cols(X_train)
    n, d = X_train_standardized.shape

    # Matrix plot
    fig, ax = plt.subplots()
    ax.imshow(X_train_standardized)
    savefig("animals_matrix.png", fig)
    plt.close(fig)

    # 2D visualization
    np.random.seed(3164)  # make sure you keep this seed
    j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    random_is = np.random.choice(n, 15, replace=False)  # choose random examples

    fig, ax = plt.subplots()
    ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
    for i in random_is:
        xy = X_train_standardized[i, [j1, j2]]
        ax.annotate(animal_names[i], xy=xy)
    savefig("animals_random.png", fig)
    plt.close(fig)

    """YOUR CODE HERE FOR Q3"""
    encoder = PCAEncoder(k=2)
    fig, ax = plt.subplots()
    encoder.fit(X_train)
    Z = encoder.encode(X_train)
    ax.scatter(Z[:,0], Z[:,1])
    W = encoder.W
    for i in random_is:
        xy = Z[i]
        plt.annotate(animal_names[i],xy=xy)
    print(f"Maximum Trait on first PC is: {trait_names[np.argmax(np.abs(W[0]))]}")
    print(f"Maximum Trait on first PC is: {trait_names[np.argmax(np.abs(W[1]))]}")
    savefig("animals_PCA_scatter", fig)
    plt.close(fig)


@handle("4")
def q4():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    # Train ordinary regularized least squares
    loss_fn = LeastSquaresLoss()
    optimizer = GradientDescentLineSearch()
    model = LinearModel(loss_fn, optimizer, check_correctness=False)
    model.fit(X_train, y_train)
    print(model.fs)  # ~700 seems to be the global minimum.

    print(f"Training MSE: {((model.predict(X_train) - y_train) ** 2).mean():.3f}")
    print(f"Validation MSE: {((model.predict(X_val) - y_val) ** 2).mean():.3f}")

    # Plot the learning curve!
    fig, ax = plt.subplots()
    ax.plot(model.fs, marker="o")
    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    savefig("gd_line_search_curve.png", fig)


@handle("4.1")
def q4_1():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    """YOUR CODE HERE FOR Q4.1"""
    base = GradientDescent()
    loss_fn = LeastSquaresLoss()
    lr_getter = ConstantLR(0.0003)
    b_sizes = [1,10,100]
    for size in b_sizes:
        optimizer = StochasticGradient(base_optimizer=base,learning_rate_getter=lr_getter,batch_size = size,max_evals=10)
        model = LinearModel(loss_fn, optimizer, check_correctness=False)
        model.fit(X_train,y_train)
        print(f"T MSE after 10 epoch : {((model.predict(X_train)-y_train)**2).mean()} with a batch size of {size}")
        print(f"V MSE after 10 epoch : {((model.predict(X_val)-y_val)**2).mean()} with a batch size of {size}")
        


@handle("4.3")
def q4_3():
    X_train_orig, y_train, X_val_orig, y_val = load_trainval("dynamics.pkl")
    X_train, mu, sigma = standardize_cols(X_train_orig)
    X_val, _, _ = standardize_cols(X_val_orig, mu, sigma)

    """YOUR CODE HERE FOR Q4.3"""
    base = GradientDescent()
    loss_fn = LeastSquaresLoss()
    lr_getter = [ConstantLR(0.1), InverseLR(0.1), InverseSquaredLR(0.1), InverseSqrtLR(0.1)]
    
    fig, ax = plt.subplots()
    colors = ['b', 'g', 'r', 'c']  # Different colors for each learning rate

    for lr, color in zip(lr_getter, colors):
        optimizer = StochasticGradient(base_optimizer=base, learning_rate_getter=lr, batch_size=10, max_evals=50)
        model = LinearModel(loss_fn, optimizer, check_correctness=False)
        model.fit(X_train, y_train)

        ax.plot(model.fs, color=color)

    ax.set_xlabel("Gradient descent iterations")
    ax.set_ylabel("Objective function f value")
    ax.legend(title='Learning Rate')
    savefig("lr_curve.png", fig)


if __name__ == "__main__":
    main()
