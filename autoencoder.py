# -*- coding: utf-8 -*-
#from time import time
#import numpy as np
from tensorflow.keras.models import Model
#import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense,Input#Layer,InputSpec,
#from tensorflow.keras import callbacks
#from sklearn.cluster import KMeans
#from sklearn import metrics
#import json



def autoencoder(dims, act='relu', init='glorot_uniform'): 
    """
    Fully connected auto-encoder model, symmetric.
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation, not applied to Input, Hidden and Output layers
    return:
        (ae_model, encoder_model), Model of autoencoder and model of encoder
    """
    n_stacks = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_stacks-1):
        h = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

    y = h
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        y = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], kernel_initializer=init, name='decoder_0')(y)

    return Model(inputs=x, outputs=y, name='AE')#, Model(inputs=x, outputs=h, name='encoder')

    
class AE(object):
    def __init__(self,
                 dims,):

        super(AE, self).__init__()
        self.dims = dims
        self.autoencoder = autoencoder(self.dims, init='glorot_uniform')           

    def train(self, x, y=None, epochs=500, batch_size=20):#, optimizer='sgd'
        #if optimizer == 'sgd':
        #    from tensorflow.keras.optimizers import SGD
        #    optimizer = SGD(lr=0.001, momentum=0.9)
        self.autoencoder.compile(optimizer='sgd', loss='mse')
        #csv_logger = callbacks.CSVLogger(self.out + '/ae_history.csv', append=True)
        history = self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs)#, callbacks=[csv_logger])
        self.autoencoder.save_weights('ae_weights.h5')
        return history

    def extract_features(self, x):
        #if ae_weights is not None:
        #    self.autoencoder.load_weights(ae_weights) 
        n_stacks = len(self.dims) - 1
        self.encoder = Model(inputs=self.autoencoder.input, 
                             outputs=self.autoencoder.get_layer('encoder_%d' % (n_stacks - 1)).output, 
                             name='encoder')
        return self.encoder.predict(x)
    
    def extract_out(self, x):
        #if ae_weights is not None:
        #    self.autoencoder.load_weights(ae_weights) 
        return self.autoencoder.predict(x)
    
    def extract_weights(self,):
        return self.autoencoder.weights
    


