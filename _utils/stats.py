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
    scaler = MinMaxScaler()
    scaler.fit(X)
    Xn = scaler.transform(X)

    return Xn
