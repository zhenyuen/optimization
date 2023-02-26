import numpy as np

def newton_rhapson_univariate(f, df, ddf, start, precision, max_iter):
    x = start
    for i in range(max_iter):
        d = - df(x) / ddf(x)

        if (abs(d) < precision): break
        
        x += d 
    return x

f = lambda x: 1 / ((x ** 0.148) * (3 * np.pi - 16 * x))
df = lambda x: -0.148 * (x ** -1.148) / (3 * np.pi - 16 * x) + 16 * x ** -0.148 / (3 * np.pi - 16 * x) ** 2
ddf = lambda x: 0.170 * (x ** -2.148) / (3 * np.pi - 16 * x) - 4.736 * x ** -1.148 / (3 * np.pi - 16 * x) ** 2 + 512 * x ** -0.148 * (3 * np.pi - 16 * x) ** -3

start = (f(0.15) + f(0.1)) / 2
precision = 10E-6
max_iter = 1000

print(newton_rhapson_univariate(f, df, ddf, start, precision, max_iter))
print(f(0.07576556030681753), f(0.07593962581838594))
