import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import argparse
import numpy as numpy
import sys

from utils import Container, tt

trng = RandomStreams(1234)
pi = 3.141592653

parser = argparse.ArgumentParser(description = 'Run SG-MCMC')
parser.add_argument('--algo', type = str, default = 'sgld', help = 'algorithm to run')
parser.add_argument('--bs', type = int, default = 32, help = 'batch size')
parser.add_argument('--lr', type = float, default = 1e-6, help = 'initial learning rate')
parser.add_argument('--burnin', type = int, default = 10000, help = 'no. of burnin iterations')
parser.add_argument('--iter', type = int, default = 500000, help = 'total no. of iterations')
parser.add_argument('--print-iter', type = int, default = 100, help = 'iterations between prints')
parser.add_argument('--prec-like', type = float, default = 1.25, help = 'noise precision')
parser.add_argument('--prec-prior', type = float, default = 1.0, help = 'prior precision')
parser.add_argument('--lr-halflife', type = int, default = 80000, help = 'iterations after which learning rate is halved')
parser.add_argument('--thinning', type = int, default = 10, help = 'sampling interval')

args = parser.parse_args()

if __name__ == '__main__':
	params = Container()
	params.algo = args.algo
	params.batch_size = args.bs
	params.lr = args.lr
	params.burnin = args.burnin
	params.n_iter = args.iter
	params.print_iter = args.print_iter
	params.prec_like = args.prec_like
	params.prec_prior = args.prec_prior
	params.lr_halflife = args.lr_halflife
	params.thinning = args.thinning
	params.train_size = 456
	params.val_size = 50

	from datasets import boston_housing
	data = boston_housing.get_data(params)