import numpy as np


def example(x):
    return np.sum(x ** 2)


def example_grad(x):
    return 2 * x


def foo(x):
    result = 1
    λ = 4  # this is here to make sure you're using Python 3
    # ...but in general, it's probably better practice to stick to plaintext
    # names. (Can you distinguish each of λ𝛌𝜆𝝀𝝺𝞴 at a glance?)
    for x_i in x:
        result += x_i ** λ
    return result

def foo_grad(x):
    # Your implementation here...
    return 4*x**3

def bar(x):
    return np.prod(x)

def bar_grad(x):
    #Used ChatGPT
    if 0 in x:
        return np.zeros_like(x)
    total_product = np.prod(x)
    grad = np.where( x != 0, total_product/x, 0)
    zero_indices = np.where(x == 0)[0]
    for idx in zero_indices:
        grad[idx] = np.prod(np.delete(x, idx))

    return grad



