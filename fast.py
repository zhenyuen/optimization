from numba import jit, config
import numpy as np
import timeit
import cProfile

config.DISABLE_JIT = True

@jit(nopython=True)
def steepest_descent(start, precision, max_iter):
    f = lambda x:\
        x.T @ np.array([[25., 0.], [0., 20.]]) @ x\
        + np.array([[-1., -1.]]) @ x
    df = lambda x: np.array([[50., 0.], [0., 40.]]) @ x\
        - np.array([[-1., -1.]]).T
    ddf = lambda x: np.array([[50., 0.], [0., 40.]])

    x = start.copy()
    for i in range(max_iter):
        d = -df(x)
        lr = d.T @ d / (d.T @ ddf(x) @ d)
        delta = lr * d
        if np.all(np.abs(delta) < precision): break
        x += delta
    for i in range(1000000000): pass # Added this line to experiment with numba speed up
    return x

start = np.array([[3., 1.]]).T
precision = 1E-17
max_iter = 1000

cProfile.run('steepest_descent(start, precision, max_iter)') # Profiling

# print(steepest_descent(start, precision, max_iter))
