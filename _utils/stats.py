import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

def nan_corrcoeff(x,y):
    x = np.array(x)
    y = np.array(y)
    
    mask = ~np.isnan(x) & ~np.isnan(y)

    new_x = x[mask]
    new_y = y[mask]

    if len(new_x) == 0 or len(new_y) == 0:
        raise(ValueError("x and y must not be only nans"))
    else:
        return pearsonr(new_x, new_y)

# normalize X's columns to [0,1]
def normalize(X):
    # X_min = X.min(axis=0)
    # X_max = X.max(axis=0)
    # X_range = X_max - X_min
    # print(X_range)
    # if np.sum(X_range) == 0:
    #     print("All values in X are the same")
    #     return 
    # else:
    #     X = (X - X_min) / X_range
    #     return X

    scaler = MinMaxScaler()
    scaler.fit(X)
    Xn = scaler.transform(X)

    return Xn
