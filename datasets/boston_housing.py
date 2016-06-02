import theano
import numpy as np

from utils import Container

def get_data(path, op):

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

    dt.trn_X = DT_x[:op.train_size]
    dt.trn_Y = DT_y[:op.train_size]
    dt.val_X = DT_x[op.train_size:]
    dt.val_Y = DT_y[op.train_size:]

    return dt

