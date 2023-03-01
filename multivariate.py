import numpy as np

def newton_rhapson_univariate(f, df, ddf, start, precision, max_iter):
    x = start
    for i in range(max_iter):
        d = -df(x) / ddf(x)

        if (abs(d) < precision): break
        
        x += d 
    print(f"Iterations: {i}")
    return x


def steepest_descent(f, df, ddf, start, precision, max_iter):
    x = start
    for i in range(max_iter):
        d = -df(x)
        lr = d.T @ d / (d.T @ ddf(x) @ d)
        delta = lr * d
        if np.all(np.abs(delta) < precision): break
        x += delta
    print(f"Iterations: {i}")
    return x


def newton_rhapson_multivariate(f, df, ddf, start, precision, max_iter):
    x = start
    for i in range(max_iter):
        d = - np.linalg.inv(ddf(x)) @ df(x) 

        if np.all(np.abs(d) < precision): break
        x += d
    print(f"Iterations: {i}")
    return x


def barzilai_borwein():
    pass


def conjugate_gradient(f, df, ddf, start, precision, max_iter):
    x = start
    d = -df(x)
    lr = d.T @ d / (d.T @ ddf(x) @ d)
    delta = lr * d
    

    for i in range(max_iter):
        x += delta

        dfx = df(x)
        beta = (dfx.T @ dfx) / (d.T @ d)

        d = -dfx + beta * d
        lr = -(d.T @ dfx) / (d.T @ ddf(x) @ d)

        delta = lr * d
        if np.all(np.abs(delta) < precision): break

    print(f"Iterations: {i}")
    return x


def newton_gauss(r, j, start, x, max_iter):
    # Cannot figure out an example application yet
    x = start
    for i in range(max_iter):
        J = j(x)
        d = - (J.T @ J).I @ J.T @ r(x)
        d = - []
    pass

# Manual unit testing, t
# f = lambda x: 1 / ((x ** 0.148) * (3 * np.pi - 16 * x))
# df = lambda x: -0.148 * (x ** -1.148) / (3 * np.pi - 16 * x) + 16 * x ** -0.148 / (3 * np.pi - 16 * x) ** 2
# ddf = lambda x: 0.170 * (x ** -2.148) / (3 * np.pi - 16 * x) - 4.736 * x ** -1.148 / (3 * np.pi - 16 * x) ** 2 + 512 * x ** -0.148 * (3 * np.pi - 16 * x) ** -3

# start = (f(0.15) + f(0.1)) / 2
# precision = 1E-10
# max_iter = 1000

# print(newton_rhapson_univariate(f, df, ddf, start, precision, max_iter))
# print(f(0.07576556030681753), f(0.07593962581838594))


f = lambda x: x.T @ np.array([[25, 0], [0, 20]]) @ x + np.array([[-1, -1]]) @ x
df = lambda x: np.array([[50, 0], [0, 40]]) @ x - np.array([[-1, -1]]).T
ddf = lambda x: np.array([[50, 0], [0, 40]])
#print(f(np.array([[0.02, 0.025]]).T))
start = np.array([[3., 1.]]).T
precision = 1E-10
max_iter = 1000
# print(steepest_descent(f, df, ddf, start, precision, max_iter))
# print(np.matmul(np.array([[1, 2]]).T, np.array([[1, 2]])))
# print(conjugate_gradient(f, df, ddf, start, precision, max_iter))
print(newton_rhapson_multivariate(f, df, ddf, start, precision, max_iter))

# r = lambda x, xhat: x - xhat
# j = dr = lambda x, xhat: np.eye(x.shape)
