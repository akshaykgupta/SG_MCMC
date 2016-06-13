import theano
import numpy as np

from utils import Container

def get_data(path, train_size = 456):

    dt = Container()
    DT = []
    with open(path, 'rb') as f:
        for line in f:
            line = [float(val) for val in line.strip().split()]
            DT.append(line)
    DT = np.asarray(DT).astype(theano.config.floatX)

    # shuffle
    shuffle_idx = np.random.permutation(DT.shape[0])
    DT = DT[shuffle_idx]

    DT_x = DT[:, :-1]
    DT_y = DT[:, -1]
    # data standardization
    dt.mu  = np.mean(DT_x, axis=0)
    dt.sig = np.std(DT_x, axis=0)
    DT_x = (DT_x - dt.mu) / dt.sig

    dt.x_train = DT_x[:train_size]
    dt.y_train = DT_y[:train_size]
    dt.x_val = DT_x[train_size:]
    dt.y_val = DT_y[train_size:]

    return dt

