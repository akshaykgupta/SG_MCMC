# SG-MCMC

Implementation of Stochastic Gradient MCMC Algorithms

SG-MCMC are a set of algorithms published over the last few years which leverage the power of bayesian optimization with the speed and memory efficiency of stochastic gradient descent. This library has Theano implementations of these algorithms for general use. All you need is to collect all the parameters (as Theano shared variables) of your model and pass it to one of these algorithms to train.

Algorithms in this repo:
* SGLD (Stochastic Gradient Langevin Dynamics) : Welling, Max, and Yee W. Teh., 2011, "Bayesian learning via stochastic gradient Langevin dynamics."
* SGFS (Stochastic Gradient Fisher Scoring) : Ahn, Sungjin et. al., 2012, "Bayesian posterior sampling via stochastic gradient Fisher scoring."
* SGNHT (Stochastic Gradient Nosé-Hoover Thermostat) : Ding, Nan, et al., 2014, "Bayesian Sampling Using Stochastic Gradient Thermostats."
* mSGNHT (Multivariable Stochastic Gradient Nosé-Hoover Thermostat) : Gan et al., 2015, "Scalable Deep Poisson Factor Analysis for Topic Modeling"
* pSGLD (Preconditioned Stochastic Gradient Langevin Dynamics) : Li, Chunyuan, et al., 2015, "Preconditioned stochastic gradient Langevin dynamics for deep neural networks."
