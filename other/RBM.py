#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time
import math
import argparse
import matplotlib.pyplot as plt
import pickle
import os

if not os.path.exists('../plot'):
    os.makedirs('../plot')
if not os.path.exists('../dump'):
    os.makedirs('../dump')

seed = 10417617 # do not change or remove


# ### helper function

# In[57]:


def binary_data(inp):
    return (inp > 0.5) * 1.


# In[58]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[59]:


def shuffle_corpus(data):
    random_idx = np.random.permutation(len(data))
    return data[random_idx]


# ### model

# In[60]:


class RBM:
    """
    The RBM base class
    """
    def __init__(self, n_visible, n_hidden, k, lr=0.01, minibatch_size=1):
        """
        n_visible, n_hidden: dimension of visible and hidden layer
        k: number of gibbs sampling steps
        k: number of gibbs sampling steps
        lr: learning rate
        vbias, hbias: biases for visible and hidden layer, initialized as zeros
            in shape of (n_visible,) and (n_hidden,)
        W: weights between visible and hidden layer, initialized using Xavier,
            same as Assignment1-Problem1, in shape of (n_hidden, n_visible)
        Do np.random.seed(seed) before you call any np.random.xx()
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.lr = lr
        
        self.vbias = np.zeros(n_visible,)
        self.hbias = np.zeros(n_hidden,)
        self.W = np.random.normal(0, np.sqrt(6.0/(self.n_hidden+self.n_visible)), (n_hidden, n_visible))


    def h_v(self, v):
        """
        Calculates hidden vector distribution P(h=1|v)
        v: visible vector in shape (N, n_visible)
        return P(h=1|v) in shape (N, n_hidden)
        N is the batch size
        """
        return sigmoid(self.hbias + v.dot(self.W.T))


    def sample_h(self, v):
        """
        Sample hidden vector given distribution P(h=1|v)
        v: visible vector in shape (N, n_visible)
        return hidden vector and P(h=1|v) both in shape (N, n_hidden)
        Do np.random.seed(seed) before you call any np.random.xx()
        """
        p_h = self.h_v(v)
        sample = np.random.binomial(1,p_h)
        return sample, p_h

    def v_h(self, h):
        """
        Calculates visible vector distribution P(v=1|h)
        h: hidden vector in shape (N, n_hidden)
        return P(v=1|h) in shape (N, n_visible)
        """
        return sigmoid(self.vbias + h.dot(self.W))

    def sample_v(self, h):
        """
        Sample visible vector given distribution P(h=1|v)
        h: hidden vector in shape (N, n_hidden)
        return visible vector and P(v=1|h) both in shape (N, n_visible)
        Do np.random.seed(seed) before you call any np.random.xx()
        """
        p_v = self.v_h(h)
        sample = np.random.binomial(1,p_v)
        return sample, p_v

    def gibbs_k(self, v, k=0):
        """
        The v (CD-k) procedure
        v: visible vector, in (N, n_visible)
        k: number of gibbs sampling steps
        return (h0, v0, h_sample, v_sample, prob_h, prob_v)
        h0: initial hidden vector sample, in (N, n_hidden)
        v0: the input v, in (N, n_visible)
        h_sample: the hidden vector sample after k steps, in (N, n_hidden)
        v_sample: the visible vector samplg after k steps, in (N, n_visible)
        prob_h: P(h=1|v) after k steps, in (N, n_hidden)
        prob_v: P(v=1|h) after k steps, in (N, n_visible)
        (Refer to Fig.1 in the handout if unsure on step counting)
        """
        if (k==0): k=self.k
        v0 = v
        h0, p_h0 = self.sample_h(v)
        vi = v0
        hi = h0
        for i in range(k):
            vi, p_vi = self.sample_v(hi)
            hi, p_hi = self.sample_h(vi)
        return (h0, v0, hi, vi, p_hi, p_vi)
        
    def update(self, X):
        """
        Updates RBM parameters with data X
        X: in (N, n_visible)
        Compute all gradients first before updating(/making change to) the
        parameters(W and biases).
        """
        N = X.shape[0]
        h0, v0, hi, vi, p_hi, p_vi = self.gibbs_k(X)
        grad_W = (self.h_v(v0).T.dot(v0) - p_hi.T.dot(vi)) / N
        grad_b = (self.h_v(v0) - p_hi).sum(0) / N
        grad_c = (v0 - vi).sum(0) / N
        
        self.W += self.lr * grad_W
        self.vbias += self.lr * grad_c
        self.hbias += self.lr * grad_b
        return grad_W, grad_b, grad_c
        

    def eval(self, X):
        """
        Computes reconstruction error, set k=1 for reconstruction.
        X: in (N, n_visible)
        Return the mean reconstruction error as scalar
        """
        N = X.shape[0]
        pred = self.sample_v(self.sample_h(X)[0])[0]
        return np.sqrt(((pred - X) ** 2).sum(1)).sum() / N


# ### main

# In[61]:


class Args():
    def __init__(self):
        self.max_epoch = 1000
        self.k = 5
        self.lr = 0.1
        self.train = '../data/digitstrain.txt'
        self.valid = '../data/digitsvalid.txt'
        self.test = "../data/digitstest.txt"
        self.n_hidden = 100


# In[62]:


if __name__ == "__main__":
    np.seterr(all='raise')

    args = Args()

    train_data = np.genfromtxt(args.train, delimiter=",")
    train_X = train_data[:, :-1]
    train_Y = train_data[:, -1]
    train_X = binary_data(train_X)
    valid_data = np.genfromtxt(args.valid, delimiter=",")
    valid_X = valid_data[:, :-1]
    valid_X = binary_data(valid_X)
    valid_Y = valid_data[:, -1]
    test_data = np.genfromtxt(args.test, delimiter=",")
    test_X = test_data[:, :-1]
    test_X = binary_data(test_X)
    test_Y = test_data[:, -1]
    
    model = RBM(784, 100, 3, lr=0.01, minibatch_size=1)
    
    for i in range(args.max_epoch):
        model.update(train_X)
        print(model.eval(train_X))

