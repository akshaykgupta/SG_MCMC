import theano
import numpy as np

from utils import Container

def get_data(path, train_size = 50000):

    dt = Container()
    DT = []
    
    with open(path, 'rb') as f:
        if (path.endswith('.csv')):
            for line in f:
                line = [float(val) for val in line.strip().split()]
                DT.append(line)

    # shuffle
    shuffle_idx = np.random.permutation(DT.shape[0])
    DT = DT[shuffle_idx]

    DT_x = DT[:, :-1]
    DT_y = DT[:, -1]

    dt.x_train = DT_x[:train_size]
    dt.y_train = DT_y[:train_size]
    dt.x_val = DT_x[train_size:]
    dt.y_val = DT_y[train_size:]

    return dt
