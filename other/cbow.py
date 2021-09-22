#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pylab as plt

# Turn testing to False while training your model.
# Please, turn it back to True while submitting your code.
_TESTING = True


# In[2]:


seed = 10417617 
_eps = 1e-5 # useful to avoid overflow while using cross entropy.


# In[17]:


f = open('./data.txt')
line_li = []
word_li = []
line = f.readline()
while line:
    line_li.append([line.strip()])
    word_li.extend(line.strip().split(' '))
    line = f.readline()
line = line_li[0]
vocab = sorted(list(set(word_li)))
f.close()


# In[3]:


def weight_init(x_len,y_len):
    b = np.sqrt(6.0/(x_len+y_len))
    if _TESTING:
        np.random.seed(seed) 
    return np.random.normal(-b, b, (x_len, y_len))


# In[19]:


class CBOW:
    """
    This class is an implementation of the continous bag of words model by first principles.
    CBOW model tries to predict the center words from surrounding words.
    """
    def __init__(self, text, window, n, learning_rate=1e-4):
        """
        Recommended hyperparameters.
        Initialize self.n, self.text,self.window,self.lr,self.vocab,self.word2index, self.V and self.U
        n = desired size of word vectors
        window = size of window
        vocab = vocabulary of all words in the dataset -- make sure it is sorted.
        word2index = index for each word in the vocabulary 
        V: input vector matrix of shape [n, len(vocab)] (W)
        U: output vector matrix of shape [len(vocab),n] (W')
        # use weight_init to initialize weights
        """
        self.text = text
        self.vocab = sorted(list(set(text.strip().split(' '))))
        self.n = n
        self.window = window
        self.lr = learning_rate
        self.word2index = dict(zip(self.vocab, [i for i in range(len(self.vocab))]))
        self.V = weight_init(n, len(self.vocab))
        self.U = weight_init(len(self.vocab), n)

    def one_hot_vector(self, index):
        """
        Function for one hot vector encoding.
        :input: a word index
        :return: one hot encoding
        """
        ret = np.zeros((1, len(self.vocab)))
        ret[0, index] = 1
        return ret[0]

    def get_vector_representation(self, one_hot=None):
        """"
        Function for vector representation from one-hot encoding.
        :input: one hot encoding
        :return: word vector
        """
        return (self.V.dot(np.expand_dims(one_hot, 1))).T[0]

    def get_average_context(self, left_context=None, right_context=None):
        """
        Function for average vector generation from surrounding context of current word.
        :input: surrounding context (left/right)
        :return: averaged vector representation
        """
        left_context.extend(right_context)
        all_onehot = [self.one_hot_vector(self.word2index[i]) for i in left_context]
        return self.get_vector_representation(np.stack(all_onehot).mean(0))

    def get_score(self, avg_vector=None):
        """
        Function for product score given an averaged vector in current context of the center word.
        :input: averaged vector
        :return: product score with matrix U.
        """
        return self.U.dot(avg_vector)

    def softmax(self, x):
        """
        Function to return the softmax output of given function.
        """
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x 

    def compute_cross_entropy_error(self, y, y_hat):
        """
        Given one hot encoding and the output of the softmax function, this function computes the
        cross entropy error.
        ---------
        :input y: one_hot encoding of the current center word
        :input y_hat: output of the softmax function
        :return: cross entropy error
        """
        return - np.log(y_hat + _eps) * y

    def compute_EH(self, e):
        """
        Function to compute the value of EH, 
        the sum of the output vectors of all words in the vocabulary,
        weighted by their prediction error.
        Look at https://arxiv.org/pdf/1411.2738.pdf for more information.
        ---------
        :input: e: prediction error. 
        :return: value of EH
        """
        self.eh = (self.U * np.expand_dims(e, 1)).sum(0)
        return self.eh

    def update_U(self, e, avg_vector): # avg_vector: hidden h
        """
        Given the cross entropy error occured in the current sample, this function updates the U matrix.
        :return self.U
        """
        self.U = self.U - self.lr * (np.expand_dims(e, 1).dot(np.expand_dims(avg_vector, 0)))
        return self.U

    def update_V(self, e, left_context=None, right_context=None):
        """
        This function updates the V matrix.
        :return: self.V
        NOTE: Do *not* divide by the context length (as mentioned in
        https://arxiv.org/pdf/1411.2738.pdf) You will need to either Take an appropriate sum. 
        or
        multiply your updates with the context length instead.
        based on your implementation.
        """
        eh = self.compute_EH(e)
        left_context.extend(right_context)
        all_index = [self.word2index[i] for i in left_context]
        self.V[:, all_index] -= self.lr * len(all_index) * np.expand_dims(eh, 1)
        return self.V

    def fit(self):
        """
        Learn the values of V and U vectors with given window size.
        Not Autograded.
        """
        pass


# In[ ]:


if __name__ == "__main__":
    pass

