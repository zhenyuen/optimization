import numpy as np


def golden_section(f, start, end, opt_width, max_iter):
    gr = (np.sqrt(5) - 1) / 2
    x1, x4 = start, end
    d = x4 - x1

    for i in range(max_iter):
        x2, x3 = x4 - gr * d, x1 + gr * d
        f2, f3 = f(x2), f(x3)
        if f3 > f2:
            x4 = x3
        else:
            x1 = x2
        
        d = x4 - x1
        
        if d < opt_width: break
    
    return x3 if f3 < f2 else x2

# Test
f = lambda x: 1 / ((x ** 0.148) * (3 * np.pi - 16 * x))
start = 0.05
end = 0.15
max_iter = 100
opt_width = 0.02
print(golden_section(f, start, end, opt_width, max_iter))







