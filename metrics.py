import numpy as np

# Regression
def r2_score(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    return 1 - (ss_res / ss_tot)

def mse(y, y_pred): ...        # mean squared error
def rmse(y, y_pred): ...       # root mse
def mae(y, y_pred): ...        # mean absolute error

# Classification (add later)
def accuracy(y, y_pred): ...
def precision(y, y_pred): ...
def recall(y, y_pred): ...
def f1(y, y_pred): ...