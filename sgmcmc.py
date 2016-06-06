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
        n_params = 1
        for p in model.params:
            n_params *= tensor.prod(p.shape)
        I_t = theano.shared(np.asarray(np.random.randn(n_params),
                                       dtype = theano.config.floatX))
        updates, log_likelihood = sgfs(model, yy, lr, I_t, params)
        params.is_sgd_mode = True
    elif params.algo == 'sgrld':
        # initialise G
        updates, log_likelihood = sgrld(model, yy, lr, is_sgd_mode, params)
    elif params.algo == 'sghmc':
        updates, log_likelihood = sghmc(model, yy, params)
    elif params.algo == 'sgnht':
        updates, log_likelihood = sgnht(model, yy, params)
    elif params.algo == 'psgld':
        V_t = [theano.shared(np.asarray(np.zeros(p.shape), 
                                        dtype = theano.config.floatX))
               for p in model.params]
        updates, log_likelihood = psgld(model, yy, lr, V_t, params)
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
        (train_pp, log_likelihood) = fn.backprop(mini_X, mini_Y, 
                                                 params.lr, params.is_sgd_mode)

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

            print ('%d/%d, %.2f, %.2f (%.2f)  \r' % \
                    (i, n_samples, sumloglik, meanlogliks, rmse), end = "")

    print ('%d/%d, %.2f, %.2f (%.2f)' % \
            (i, n_samples, sumloglik, meanlogliks, rmse))

def grad_clipping(model_params, grads, gc_norm=10.):
    norm = ut.norm_gs(model_params, grads)
    sqrtnorm = tensor.sqrt(norm)
    adj_norm_gs = tensor.switch(tensor.ge(sqrtnorm, gc_norm),
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
                       tensor.sqrt(lr) * trng.normal(p.shape, avg = 0.0, std = 1.0))
        updates.append((p, p + 0.5 * lr * grad + noise))

    return updates, sumloglik

def sgfs(model, yy, lr, I_t, params):
    '''Stochastic Gradient Fisher Scoring
    Implemented according to the paper:
    Ahn, Sungjin et. al., 2012
    "Bayesian posterior sampling via stochastic gradient Fisher scoring."
    '''

    n = params.batch_sz
    N = params.train_size
    gamma = (n + N) / n

    logliks = - 0.5 * (tensor.log(2 * pi / params.prec_lik) + params.prec_lik * (model.pp - yy)**2)
    sumloglik = logliks.sum()

    # compute gradient of likelihood wrt each data point
    # grads, log_liks = tensor.jacobian(expression = logliks, wrt = model.params)
    param_list = []

    grads_logliks = tensor.jacobian(expression = logliks, wrt = model.params)
    avg_grads_logliks = grad_logliks.mean(axis = 0)
    dist_grads_logliks = grad_logliks - avg_grad_logliks

    # compute variance of gradient
    var_grads_logliks = (1. / (n-1)) * T.dot(dist_grad_logliks.T, dist_grad_logliks)

    logprior = log_prior_normal(model.params, params.prec_prior)
    grads_prior = tensor.grad(cost = logprior, wrt = model.params)

    # update Fisher information
    I_t_next = (1 - lr) * I_t + lr * var_grads_logliks

    B_ch = slinalg.Cholesky(params.B)
    noise = T.dot(((2. / tensor.sqrt(lr)) * B_ch), trng.normal(grads_logliks.shape, avg = 0.0, std = 1.0))

    # expensive inversion
    inv_matrix = nlinalg.MatrixInverse(gamma * N * I_t_next + (4. / lr) * params.B)

    updates = []
    updates.append((I_t, I_t_next))

    # update the parameters
    updated_params = 2 * T.dot(inv_matrix, (grads_prior + N * avg_grads_logliks + noise))
    last_row = 0
    for p in model.params:
        sub_index = tensor.prod(p.shape)
        up = updated_params[last_row:sub_index]
        up.reshape(p.shape)
        updates.append((p, up))
        last_row += sub_index

    return updates, sumloglik

def sgrld(model, yy, lr, is_sgd_mode, params):
    n = params.batch_sz
    N = params.train_size

    logliks = - 0.5 * (tensor.log(2 * pi / params.prec_lik) + params.prec_lik * (model.pp - yy)**2)
    logprior = log_prior_normal(model.params, params.prec_prior)
    sumloglik = logliks.sum()
    logpost = N * sumloglik / n + logprior

    grads = tensor.grad(cost = logpost, wrt = model.params)
    updates = []
    for p, g in zip(model.params, grads):
        noise = ifelse(tensor.eq(is_sgd_mode, 1.),
                       tensor.alloc(0., *p.shape),
                       tensor.sqrt(lr * p) * trng.normal(p.shape, avg = 0.0, std = 1.0))
        updates.append((p, p + 0.5 * lr * p * g + noise))

    return updates, sumloglik

def psgld(model, yy, lr, V_t, params):
    n = params.batch_sz
    N = params.train_size

    logliks = - 0.5 * (tensor.log(2 * pi / params.prec_lik) + params.prec_lik * (model.pp - yy)**2)
    sumloglik = logliks.sum()
    meanloglik = sumloglik / n
    grads = tensor.grad(cost = meanloglik, wrt = model.params)
    
    V_t_next = [params.alpha * v + (1 - params.alpha) * g * g for g, v in zip(grads, V_t)]
    G_t = [1. / (params.Lambda + tensor.sqrt(v)) for v in V_t_next]
    
    logprior = log_prior_normal(model.params, params.prec_prior)
    grads_prior = tensor.grad(cost = logprior, wrt = model.params)

    updates = []
    [updates.append(v, v_n) for v, v_n in zip(V_t, V_t_next)]

    for p, g, gp, gt in zip(model.params, grads, grads_prior, G_t):
        noise = tensor.sqrt(lr * G_t) * trng.normal(p.shape, avg = 0.0, std = 1.0)
        updates.append((p, p + 0.5 * lr * ((gt * (gp + N * g))) + noise))

    return updates, sumloglik
