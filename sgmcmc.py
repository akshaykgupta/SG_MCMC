from __future__ import print_function

import theano
from theano import tensor
from theano.sandbos.rng_mrg import MRG_RandomStreams as RandomStreams

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

def train(model, data, params):
    '''Train the model.

    Parameters
    ----------
    model: Container
    The model to train. Should have the following attributes:
    model.xx : theano.tensor.matrix  (the input variable)
    model.pp : theano.tensor.vector  (the model output variable)
    model.params : list (list of model parameters as theano shared variables)

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
    yy = tensor.vector('yy', dtype='float32')
    params.train_size = data.trn_X.shape[0]

    if params.algo == 'sgld':
        updates, log_likelihood = sgld(model, yy, lr, is_sgd_mode, params)
    elif params.algo == 'sgfs':
        updates, log_likelihood = sgfs(model, yy, params)
    elif params.algo == 'sgrld':
        updates, log_likelihood = sgrld(model, yy, params)
    elif params.algo == 'sghmc':
        updates, log_likelihood = sghmc(model, yy, params)
    elif params.algo == 'sgnht':
        updates, log_likelihood = sgnht(model, yy, params)
    elif params.algo == 'psgld':
        updates, log_likelihood = psgld(model, yy, params)
    else:
        raise UnknownArgumentException

    fn = Container()
    fn.backprop = theano.function(inputs = [model.xx, yy, lr, is_sgd_mode],
                                  outputs = [model.pp, log_likelihood],
                                  updates = updates,
                                  on_unused_input = 'warn')
    fn.forward = theano.function(inputs = [model.xx],
                                 outputs =[model.pp],
                                 on_unused_input = 'warn')

    avg_pp = np.zeros(data.val_Y.shape)
    sum_pp = np.zeros(data.val_Y.shape)
    sumsq_pp = np.zeros(data.val_Y.shape)
    val_avg_log_likelihood = 0.0
    n_samples = 0

    if not params.is_sgd_mode:
        sampling = False

    for i in range(params.n_iter):

        #prepare next minibatch
        mini_idx = np.floor(np.random.rand(params.batch_sz) * params.train_size).astype('int32')
        mini_X = data.trn_X[mini_idx]
        mini_Y = data.trn_Y[mini_idx]

        #parameter update
        (train_pp, log_likelihood) = fn.backprop(mini_X, mini_Y, params.lr, params.is_sgd_mode)

        if i == params.burnin and not params.is_sgd_mode:
            #burnin period over, begin sampling
            sampling = True

        if i % params.lr_halflife == 0:
            params.lr = params.lr / 2

        if i % thinning == 0:
            val_pp = fn.forward(data.val_X)

            if sampling:
                n_samples += 1
                # prediction based on current parameter (sample)
                avg_pp = ((1 - (1./n_samples)) * avg_pp) + ((1./n_samples) * val_pp)
                ppp = avg_pp
                # online sample variance 
                sum_pp += val_pp
                sumsqr_pp += val_pp * val_pp
                var_pp = (sumsqr_pp - (sum_pp * sum_pp)/n_samples) / (n_samples - 1)
            else:
                ppp = val_pp

                trn_pp = fn.forward(dt.trn_X) # train predictions
                var_pp = np.var(trn_pp - data.trn_Y)

            mean_ll = logpdf_normal(ppp, val_yy, 1/var_pp).mean()
            rmse = compute_rmse(ppp, val_yy)

            print '%d/%d, %.2f, %.2f (%.2f)  \r' % \
                    (i, n_samples, sumloglik, meanlogliks, rmse),

    print '%d/%d, %.2f, %.2f (%.2f)' % \
            (i, n_samples, sumloglik, meanlogliks, rmse)

def grad_clipping(model_params, grads, gc_norm=10.):
    norm = ut.norm_gs(model_params, grads)
    sqrtnorm = T.sqrt(norm)
    adj_norm_gs = T.switch(T.ge(sqrtnorm, gc_norm), 
                           gc_norm / sqrtnorm, 1.)
    return adj_norm_gs

def sgld(model, yy, lr, is_sgd_mode, params):
    '''Stochastic Gradient Langevin Dynamics
    Implemented according to the paper:
    Welling, Max, and Yee W. Teh., 2011
    "Bayesian learning via stochastic gradient Langevin dynamics."
    '''

    n = params.batch_sz
    N = params.train_size

    logliks = - 0.5 * (tensor.log(2 * pi / params.prec_lik) + params.prec_lik * (model.pp - yy)**2) 
    logprior = log_prior_normal(model.params, params.prec_prior)
    sumloglik = logliks.sum()
    logpost = N * sumloglik / n + logprior

    #compute gradients
    grads = tensor.grad(cost = logpost,  wrt = model.params)

    # gradient clipping 
    if params.gc_norm is not None:
        adj_norm_gs = grad_clipping(model.params, grads, params.gc_norm)  
    else:
        adj_norm_gs = 1.0

    updates = []
    for p, g in zip(model.params, grads):
        grad = g * adj_norm_gs 
        # noise is discarded in sgd mode which may be used during burnin
        noise = ifelse(tensor.eq(is_sgd_mode, 1.),
                       tensor.alloc(0., *p.shape),
                       tensor.sqrt(lr) * trng.normal(p.shape, avg=0.0, std=1.0))
        updates.append((p, p + 0.5 * lr * grad + noise))
    return updates, sumloglik
