import os
import cPickle as pickle
import numpy as np
import sys
import datetime
import inspect
from collections import OrderedDict

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
sys.setrecursionlimit(100000)


def tt(tensor_var):
    return (tensor_var.tag.test_value, tensor_var.tag.test_value.shape)

#def tts(tensor_var):
    #return tensor_var.tag.test_value.shape


def cur_timetag():
    time_tag = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return time_tag[2:].replace(' ','_').translate(None, '-:')


def one_hot(idxs, n_class):
    onehot = T.zeros((idxs.shape[0], n_class))
    onehot = T.set_subtensor(onehot[T.arange(idxs.shape[0]), idxs], 1)
    return onehot


def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')


class Container():
    def __init__(self):
        #self.__dict__['tparams'] = dict()
        self.__dict__['tparams'] = OrderedDict()
    
    def __setattr__(self,name,array):
        tparams = self.__dict__['tparams']
        #if name not in tparams:
        tparams[name] = array
    
    def __setitem__(self,name,array):
        self.__setattr__(name,array)
    
    def __getitem__(self,name):
        return self.__getattr__(name)
    
    def __getattr__(self,name):
        tparams = self.__dict__['tparams']
        return self.tparams[name]

    #def __getattr__(self):
        #return self.get()

    def remove(self,name):
        del self.__dict__['tparams'][name]

    def get(self):
        return self.__dict__['tparams']

    def items(self):
        return self.__dict__['tparams'].items()

    def iteritems(self):
        return self.__dict__['tparams'].iteritems()

    def keys(self):
        return self.__dict__['tparams'].keys()

    def values(self):
        tparams = self.__dict__['tparams']
        return tparams.values()

    def save(self,filename):
        tparams = self.__dict__['tparams']
        pickle.dump({p:tparams[p] for p in tparams},open(filename,'wb'),2)

    def load(self,filename):
        tparams = self.__dict__['tparams']
        loaded = pickle.load(open(filename,'rb'))
        for k in loaded:
            tparams[k] = loaded[k]

    def setvalues(self, values):
        tparams = self.__dict__['tparams']
        for p, v in zip(tparams, values):
            tparams[p] = v

    def __enter__(self):
        _,_,_,env_locals = inspect.getargvalues(inspect.currentframe().f_back)
        self.__dict__['_env_locals'] = env_locals.keys()

    def __exit__(self,type,value,traceback):
        _,_,_,env_locals = inspect.getargvalues(inspect.currentframe().f_back)
        prev_env_locals = self.__dict__['_env_locals']
        del self.__dict__['_env_locals']
        for k in env_locals.keys():
            if k not in prev_env_locals:
                self.__setattr__(k,env_locals[k])
                env_locals[k] = self.__getattr__(k)
        return True


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.T.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.T.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.T.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano T expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : T
            the concatenated T expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out
