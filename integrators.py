import numpy as np


def E1(y, x, h, f, *args):
    return y + h * f(x, *args)

def RK2(y, x, h, f, *args):
    k = f(x + 0.5 * h * f(x, *args), *args)
    return y + h * k
    
def RK4(y, x, h, f, *args):
    k1 = f(x, *args)
    k2 = f(x + h * 0.5 * k1)
    k3 = f(x + h * 0.5 * k2)
    k4 = f(x + h * k3)
    return y + (1 / 6) * h * (k1 + 2 * k2 + 2 * k3 + k4)

def RK4_V(x, v, h, f, *args):
    k0 = f(x, *args)
    c0 = v

    v1 = v + 0.5 * h * k0
    x1 = x + 0.5 * h * c0
    k1 = f(x1, *args)
    c1 = v1

    v2 = v + 0.5 * h * k1
    x2 = x + 0.5 * h * c1
    k2 = f(x2, *args)
    c2 = v2

    v3 = v + h * k2
    x3 = x + h * c2
    k3 = f(x3, *args)
    c3 = v3

    v_new = v + (h / 6) * (k0 + 2 * k1 + 2 * k2 + k3)
    x_new = x + (h / 6) * (c0 + 2 * c1 + 2 * c2 + c3)
    

    return v_new, x_new
