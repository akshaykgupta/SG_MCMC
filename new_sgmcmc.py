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

def log_prior_normal(model_params, prec):
    res = 0.
    for p in model_params:
        res += tensor.sum(- 0.5 * (tensor.log(2 * pi / prec) + prec * tensor.sqr(p)))
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

	def get_updates(self, model, params):
		raise NotImplementedError

	def initialize_params(self, params):
		raise NotImplementedError

	def train(model, data, params):
		raise NotImplementedError
		
class SGLD(Trainer):
    '''Stochastic Gradient Riemannian Langevin Dynamics
    Implemented according to the paper:
    Patterson, Sam, and Yee Whye Teh, 2013
    "Stochastic gradient Riemannian Langevin dynamics on the probability simplex."
    '''

    def __init__(self, initial_lr=1.0e-5, **kwargs):
    	super(SGLD, self).__init__(kwargs)
    	self.params['lr'] = initial_lr
    	self.lr = tensor.fscalar('lr')
    	self.
