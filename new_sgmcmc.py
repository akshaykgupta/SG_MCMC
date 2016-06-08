from __future__ import print_function

import theano
from theano import tensor
from theano.tensor import slinalg
from theano.tensor import nlinalg
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse

import numpy as np

pi = 3.141592653
trng = RandomStreams(1234)

def log_normal(x, prec):
    return tensor.sum(-0.5 * (tensor.log(2*pi / prec) + prec * tensor.sqr(x)))

def log_prior_normal(model_params, prec):
    res = 0.
    for p in model_params:
        res += log_normal(p, prec)
    return res

def logpdf_normal(pp, yy, prec_lik):
    logliks = - 0.5 * (np.log(2 * pi / prec_lik) + prec_lik * (pp - yy)**2)
    return logliks

def compute_rmse(xx, yy):
    return np.sqrt(((xx - yy)**2).mean())

class Trainer(object):
    '''Abstract base class for all SG-MCMC trainers.
    This is NOT a trainer in itself.
    '''

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates = []
        self.params = {}
        self.inputs = tensor.matrix('inputs')
        self.model_outputs = tensor.matrix('model_outputs')
        self.true_outputs = tensor.row('true_outputs')

    def _get_updates(self):
        raise NotImplementedError

    def _get_training_function(self):
        return theano.function(inputs  = [self.model_inputs, self.true_outputs,
                                          self.lr],
                               outputs = [self.model_outputs, self.sumloglik],
                               updates = self.updates)

    def _get_prediction_function(self):
        return theano.function(inputs  = [self.model_inputs],
                               outputs = [self.model_outputs])

    def initialize_params(self, params):
        raise NotImplementedError

    def train(model, data, params):
        raise NotImplementedError

class SGLD(Trainer):
    '''Stochastic Gradient Langevin Dynamics
    Implemented according to the paper:
    Welling, Max, and Yee W. Teh., 2011
    "Bayesian learning via stochastic gradient Langevin dynamics."
    '''

    def __init__(self, initial_lr=1.0e-5, **kwargs):
        super(SGLD, self).__init__(kwargs)
        self.params['lr'] = initial_lr
        self.lr = tensor.scalar('lr')

    def _get_updates(self):
        n = self.params['batch_size']
        N = self.params['train_size']
        prec_lik = self.params['prec_lik']
        prec_prior = self.params['prec_prior']
        gc_norm = self.params['gc_norm']

        error = self.model_outputs - self.true_outputs
        logliks = log_normal(error, prec_lik)
        logprior = log_prior_normal(self.weights, prec_prior)
        sumloglik = logliks.sum()
        logpost = N * sumloglik / n + logprior

        #compute gradients
        grads = tensor.grad(cost = logpost,  wrt = self.weights)

        # gradient clipping 
        if gc_norm is not None:
            adj_norm_gs = grad_clipping(self.weights, grads, gc_norm)  
        else:
            adj_norm_gs = 1.0

        updates = []
        for p, g in zip(self.weights, grads):
            grad = g * adj_norm_gs
            #inject noise
            noise = tensor.sqrt(self.lr) * trng.normal(p.shape, avg = 0.0, std = 1.0)
            updates.append((p, p + 0.5 * self.lr * grad + noise))

        return updates, sumloglik
