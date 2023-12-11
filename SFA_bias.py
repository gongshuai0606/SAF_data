import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp
import pandas as pd
import seaborn as sns
from bisect import bisect_left
import math


def closest_idx(array, value):
    """
    Find index of the element of 'array' which is closest to 'value'.
    """
    if type(array) is np.ndarray:
        pos = bisect_left(array, value)

    else:
        pos = bisect_left(array.numpy(), value)
    if pos == 0:
        return 0
    elif pos == len(array):
        return -1
    else:
        return pos-1


X = np.loadtxt('')
X = (X - np.min(X)) / (np.max(X) - np.min(X))

T = len(X)

bias = X[:, -2]
t = X[:, 0]

temp = 300.
kb = 0.008314
beta = 1./(kb*temp)
logweight = beta*bias

dt = np.round(t[1]-t[0], 3)

d_tprime = np.copy(np.exp(logweight)*dt)

tprime = np.cumsum(d_tprime)


t_lenth = math.floor(tprime[-1])
x_ = []
for i in range(t_lenth):
    idx = closest_idx(tprime, i)
    x_.append(X[idx])

X = np.array(x_)

X = (X - np.mean(X, axis=0))

X = X.T

C = np.dot(X, X.T)

dX = np.diff(X)/dt

dC = np.dot(dX, dX.T)

eigenvalues, eigenvectors = sp.linalg.eig(dC, C)

order = eigenvalues.argsort()

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]


