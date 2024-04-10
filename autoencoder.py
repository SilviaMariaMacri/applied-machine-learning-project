# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import CSVLogger,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.metrics import RootMeanSquaredError
import os 

def autoencoder(dims, act='relu'): 

    '''
    Fully connected and symmetric autoencoder model
    
    Input:
    ------
    dims: list
          list of number of nodes in each layer of encoder. 
          dims[0] is the input dim, dims[-1] is the hidden layer dimension
    act: str
         activation function applied to internal layers in encoder and decoder
    
    Returns:
    -------
        ae_model: keras.Model
                  model of autoencoder 
    '''
    
    n_layer = len(dims) - 1
    # input
    x = Input(shape=(dims[0],), name='input')
    h = x

    # internal layers in encoder
    for i in range(n_layer-1):
        h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)

    # hidden layer
    h = Dense(dims[-1], name='encoder_%d' % (n_layer - 1))(h)

    y = h
    # internal layers in decoder
    for i in range(n_layer-1, 0, -1):
        y = Dense(dims[i], activation=act, name='decoder_%d' % i)(y)

    # output
    y = Dense(dims[0], name='decoder_0')(y)

    ae_model = Model(inputs=x, outputs=y, name='AE')

    return ae_model

    
class AE(object):

    def __init__(self,dims,):
        super(AE, self).__init__()
        self.dims = dims
        self.autoencoder = autoencoder(self.dims)           

    def train(self, 
              x, 
              optimizer='sgd', loss='mse', metrics=None, # compile parameters
              batch_size=32, epochs=800, callbacks=[EarlyStopping(monitor="loss", patience=15),ReduceLROnPlateau(monitor="loss", patience=10)], # fit parameters
              save_dir=None):
        
        '''
        Trains the autoencoder with the input data
        '''
        
        self.autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if (save_dir is not None) and (callbacks is not None):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            callbacks.append(CSVLogger(save_dir+'/history.csv', separator="\t")) # Save epoch results to a CSV file
        history = self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=2)
        
        if save_dir is not None:
            self.autoencoder.save_weights(save_dir+'/weights.h5')
        
        return history

    def extract_features(self, x):
        
        '''
        Returns latent dimension coordinates given input data, after training or loading saved weights
        '''

        n_layers = len(self.dims) - 1
        self.encoder = Model(inputs=self.autoencoder.input, 
                             outputs=self.autoencoder.get_layer('encoder_%d' % (n_layers - 1)).output, 
                             name='encoder')
        return self.encoder.predict(x)
    
    def extract_out(self, x):

        '''
        Returns output coordinates given input data, after training or loading saved weights
        '''

        return self.autoencoder.predict(x)
    
    def load_weights(self,save_dir):

        '''
        Loads weights from file
        '''

        return self.autoencoder.load_weights(save_dir+'/weights.h5')