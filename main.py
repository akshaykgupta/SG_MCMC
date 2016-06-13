import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import argparse
import numpy as np
import sys

from utils import Container, tt
from datasets import boston_housing
from new_sgmcmc import SGLD

def shared(xx, borrow=False):
    return theano.shared(np.asarray(xx, dtype=theano.config.floatX), borrow=borrow)

def ishared(xx, borrow=False):
    return theano.shared(np.asarray(xx, dtype=np.int32), borrow=borrow)

def init_normal(shape, scale=1.0):
    if len(shape) == 1:
        scale_factor =  scale / np.sqrt(shape[0])
    elif len(shape) == 2:
        scale_factor =  scale / (np.sqrt(shape[0]) + np.sqrt(shape[1]))
    else:
        scale_factor =  scale / (np.sqrt(shape[1]) + np.sqrt(shape[2]))
    if len(shape) == 2:
        return shared(np.random.randn(*shape) * scale_factor)
    else:
        return shared(np.random.randn(*shape) * scale_factor)

def init_uniform(shape, scale=0.05):
    return shared(np.random.uniform(low=-scale, high=scale, size=shape))

def init_zero(shape):
    return shared(np.zeros(shape))

def init_eye(dim, scale=1.0):
    return shared(np.eye(dim)*scale)

def relu(x):
    return T.nnet.relu(x)

def linear(x):
    return x

def log_prior_normal(params, prec):
    res = 0.
    for p in params:
        res += T.sum(- 0.5 * (T.log(2 * pi / prec) + prec * T.sqr(p)))
    return res
####################################

def mlp_init(lp, inp_edim=None, out_edim=None, pre='mlp'):
    lp[pre+'_W'] = init_normal((inp_edim, out_edim), scale=0.0001)
    lp[pre+'_b'] = init_zero((out_edim,))
    return lp

def mlp_layer(lp, xx, pre="mlp", activ='lambda x: T.nnet.relu(x)'):
    return eval(activ)(T.dot(xx, lp[pre+'_W']) + lp[pre+'_b'])


####################################

def run(dt, params = None):
    print 'Creating model ...'
    # a neural network
    lp = Container() # learning parameters 
    lp = mlp_init(lp, dt.x_train.shape[1], 50, pre='mlp1')
    lp = mlp_init(lp, 50, 1, pre='mlp2')
    
    xx = T.matrix('xx', dtype='float32') # (b,d)
    hh = mlp_layer(lp, xx, pre='mlp1', activ='relu') # (b,h)
    pp = mlp_layer(lp, hh, pre='mlp2', activ='linear') # (b,)
    pp = pp.flatten() # from (b,1) to (b,)

    model = Container()
    model.weights = lp.values() # model params
    model.inputs = xx
    model.outputs = pp

    print 'Creating SGLD instance ...'
    sgld = SGLD()
    print 'Training the model ...'
    sgld.train(model, dt)

if __name__=="__main__":
    
    np.random.seed(1)

    print 'Fetching data ...'
    
    dt = boston_housing.get_data('/Users/Akshay/Documents/MILA/SG_MCMC/data/housing.data', train_size = 456)
    params = {}
    run(dt)