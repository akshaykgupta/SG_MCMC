from __future__ import print_function

import theano
from theano import tensor

import numpy as np

def train(model, data, params):
    '''Train the model.

    Parameters
    ----------
    model: Container
    The model to train. Should have the following attributes:
    model.xx : theano.tensor.matrix  (for input)
    model.yy : theano.tensor.vector  (for output)
    model.params : flat dictionary of model parameters

    data: Container
    The data to train on. Should have the following attributes:
    data.trn_X : np.array
    data.trn_Y : np.array
    data.val_X : np.array
    data.val_Y : np.array

    params: Container
    Parameters for training

    '''

    is_sgd_mode = tensor.fscalar('is_sgd_mode')
    lr = tensor.fscalar('lr')

    if params.algo == 'sgld':
        updates, log_likelihood, pp = sgld(model, params)
    elif params.algo == 'sgfs':
        updates, log_likelihood, pp = sgfs(model.params, params)
    elif params.algo == 'sgrld':
        updates, log_likelihood, pp = sgrld(model.params, params)
    elif params.algo == 'sghmc':
        updates, log_likelihood, pp = sghmc(model.params, params)
    elif params.algo == 'sgnht':
        updates, log_likelihood, pp = sgnht(model.params, params)
    elif params.algo == 'psgld':
        updates, log_likelihood, pp = psgld(model.params, params)
    else:
        raise UnknownArgumentException

    fn = Container()
    fn.train = theano.function(inputs = [xx, yy, lr, is_sgd_mode],
                               outputs = [pp, log_likelihood],
                               updates = updates,
                               on_unused_input = 'warn')
    fn.val = theano.function(inputs = [xx],
                             outputs =[pp],
                             on_unused_input = 'warn')

    avg_pp = np.zeros(data.val_Y.shape)
    sum_pp = np.zeros(data.val_Y.shape)
    sumsq_pp = np.zeros(data.val_Y.shape)
    val_avg_log_likelihood = 0.0

    if not params.is_sgd_mode:
        sampling = False

    for i in range(params.n_iter):

        #prepare next minibatch
        mini_idx = np.floor(np.random.rand(params.batch_sz) * data.train_size).astype('int32')
        mini_X = data.trn_X[mini_idx]
        mini_Y = data.trn_Y[mini_idx]
        train_pp, log_likelihood = fn.train(mini_X, mini_Y, params.lr, params.is_sgd_mode)

        if i == params.burnin and not params.is_sgd_mode:
            #burnin period over, begin sampling
            sampling = True

        if i % params.lr_halflife == 0:
            lr = lr / 2

        if i % thinning == 0:
            val_pp = fn.val(data.val_X, data.val_Y)
            




