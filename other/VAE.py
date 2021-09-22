#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from utils import *


# In[34]:


class VAE(object):
    #This is VAE model that you should implement
    def __init__(self, hidden_units=128, z_units=20, input_dim=784, batch_size=64):
        """
        initialize all parameters in the model.
        Encoding part:
        1. W_input_hidden, b_input_hidden: convert input to hidden
        2. W_hidden_mu, b_hidden_mu:
        3. W_hidden_logvar, b_hidden_logvar
        Sampling:
        1. random_sample
        Decoding part:
        1. W_z_hidden, b_z_hidden
        2. W_hidden_out, b_hidden_out
        """
        self.hidden_units = hidden_units
        self.z_units = z_units
        self.input_dim = input_dim
        
        self.W_input_hidden = np.random.randn(input_dim, hidden_units) * 0.01
        self.b_input_hidden = np.zeros((hidden_units))
        self.e_lrelu = LRelu('VAE')
        
        self.W_hidden_mu = np.random.randn(hidden_units, z_units) * 0.01
        self.b_hidden_mu = np.zeros((z_units))
        
        self.W_hidden_logvar = np.random.randn(hidden_units, z_units) * 0.01
        self.b_hidden_logvar = np.zeros((z_units))
        
        self.W_z_hidden = np.random.randn(z_units, hidden_units) * 0.01
        self.b_z_hidden = np.zeros((hidden_units))
        self.d_relu = Relu()
        
        self.W_hidden_out = np.random.randn(hidden_units, input_dim) * 0.01
        self.b_hidden_out = np.zeros((input_dim))
        self.d_sigmoid = Sigmoid()

    def encode(self, x):
        """
        input: x is input image with size (batch, indim)
        return: hidden_mu, hidden_logvar, both sizes should be (batch, z_units)
        """
        self.x = x
        encode_hidden = x.dot(self.W_input_hidden) + self.b_input_hidden
        self.leaky_encode_hidden = self.e_lrelu(encode_hidden)
        
        self.hidden_mu = self.leaky_encode_hidden.dot(self.W_hidden_mu) + self.b_hidden_mu
        self.hidden_logvar = self.leaky_encode_hidden.dot(self.W_hidden_logvar) + self.b_hidden_logvar
        
        return self.hidden_mu, self.hidden_logvar

    def decode(self, z):
        """
        input: z is the result from sampling with size (batch, z_unit)
        return: out, the generated images from decoder with size (batch, indim)
        """
        decode_hidden = z.dot(self.W_z_hidden) + self.b_z_hidden
        self.relu_decode_hidden = self.d_relu(decode_hidden)
        leaky_decode_out = self.relu_decode_hidden.dot(self.W_hidden_out) + self.b_hidden_out
        decode_out = self.d_sigmoid(leaky_decode_out)
        return decode_out

    def forward(self, x, unittest=False):
        """
        combining encode, sampling and decode.
        input: x is input image with size (batch, indim)
        return: out, the generated images from decoder with size (batch, indim)
        """
        if (unittest): np.random.seed(1433125)
        self.hidden_mu, self.hidden_logvar = self.encode(x)
        self.epsilon = np.random.randn(x.shape[0], self.z_units)
        self.z = self.hidden_mu + self.epsilon * np.sqrt(np.exp(self.hidden_logvar))
        out = self.decode(self.z)
        return out
    
    def loss(self, x, out):
        """
        Given the input x (also the ground truth) and out, computing the loss (CrossEntropy + KL).
        input: x is the input of the model with size (batch, indim)
               out is the predicted output of the model with size (batch, indim)
        IMPORTANT: the loss computed should be divided by batch size.
        """
        loss = (BCE_loss(out, x) - 
                0.5 * np.sum((1 + self.hidden_logvar - self.hidden_mu **2 - np.exp(self.hidden_logvar)))) / x.shape[0]
        return loss
        
    def backward(self, x, pred):
        """
        Given the input x (also the ground truth) and out, computing the gradient of parameters.
        input: x is the input of the model with size (batch, indim)
               pred is the predicted output of the model with size (batch, indim)
        return: grad_list = [dW_input_hidden, db_input_hidden, dW_hidden_mu, db_hidden_mu, dW_hidden_logvar, 
        db_hidden_logvar,
                            dW_z_hidden, db_z_hidden, dW_hidden_out, db_hidden_out]
        IMPORTANT: make sure the gradients follows the exact same order in grad_list.
        """
        batch = pred.shape[0]
        
        d_pred = - x / pred + (1 - x) / (1 - pred)
        db_hidden_out = self.d_sigmoid.backward(d_pred)
        dW_hidden_out = self.relu_decode_hidden.T.dot(db_hidden_out)#  / batch
        db_z_hidden = self.d_relu.backward(db_hidden_out.dot(self.W_hidden_out.T))
        dW_z_hidden = self.z.T.dot(db_z_hidden)
        
        dz = db_z_hidden.dot(self.W_z_hidden.T)
        dmu = dz + self.hidden_mu
        dlogvar = dz * (0.5 * np.exp(0.5 * self.hidden_logvar)*self.epsilon) + 0.5 * (np.exp(self.hidden_logvar) - 1)
        
        db_hidden_logvar = dlogvar
        dW_hidden_logvar = self.leaky_encode_hidden.T.dot(dlogvar)
        db_hidden_mu = dmu
        dW_hidden_mu = self.leaky_encode_hidden.T.dot(dmu)
        
        db_input_hidden = self.e_lrelu.backward(db_hidden_mu.dot(self.W_hidden_mu.T) + db_hidden_logvar.dot(self.W_hidden_logvar.T))
        dW_input_hidden =  self.x.T.dot(db_input_hidden) 

        return [dW_input_hidden, 
                db_input_hidden.sum(0), 
                dW_hidden_mu, 
                db_hidden_mu.sum(0), 
                dW_hidden_logvar, 
                db_hidden_logvar.sum(0), 
                dW_z_hidden, 
                db_z_hidden.sum(0), 
                dW_hidden_out, 
                db_hidden_out.sum(0)]

    def set_params(self, parameter_list):
        """
        IMPORTANT: used by autograder and unit-test
        TO set parameters with parameter_list
        input: parameter_list = [W_input_hidden, b_input_hidden, W_hidden_mu, b_hidden_mu, W_hidden_logvar, 
        b_hidden_logvar,
                            W_z_hidden, b_z_hidden, W_hidden_out, b_hidden_out]
        """
        self.W_input_hidden = parameter_list[0]
        self.b_input_hidden = parameter_list[1]
        self.W_hidden_mu = parameter_list[2]
        self.b_hidden_mu = parameter_list[3]
        self.W_hidden_logvar = parameter_list[4]
        self.b_hidden_logvar = parameter_list[5]
        self.W_z_hidden = parameter_list[6]
        self.b_z_hidden = parameter_list[7]
        self.W_hidden_out = parameter_list[8]
        self.b_hidden_out = parameter_list[9]


# In[35]:


if __name__ == "__main__":
    # x_train is of shape (5000 * 784)
    # We've done necessary preprocessing for you so just feed it into your model.
    x_train = np.load('data.npy')



#VAE = VAE()


# In[38]:


#VAE.forward(x_train)


# In[ ]:




