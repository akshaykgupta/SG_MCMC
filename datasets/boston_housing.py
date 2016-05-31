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
    
    # data standardization
    dt.mu  = np.mean(DT, axis=0)
    dt.sig = np.std(DT, axis=0)
    DT = (DT - dt.mu) / dt.sig 
    dt.no_stdz = lambda val: val * dt.sig[-1] + dt.mu[-1]
    
    # shuffle
    shuffle_idx = np.random.permutation(DT.shape[0])
    DT = DT[shuffle_idx]

    dt.trn_X = DT[:op.train_size,:-1]
    dt.trn_Y = DT[:op.train_size,-1]
    dt.val_X = DT[op.train_size:,:-1]
    dt.val_Y = DT[op.train_size:,-1]

    return dt

