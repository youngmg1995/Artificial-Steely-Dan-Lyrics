# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:06:39 2020

@author: Mitchell
"""

# Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, LSTM, BatchNormalization,\
    Dropout, Dense, Activation
from tensorflow.keras.utils import Sequence
import numpy as np


# Define LSTM Model Class
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class my_lstm(Model):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 batch_size,
                 lstm_units,
                 dense_units,
                 max_norm = None,
                 dropout = 0.):
        # Call to Super
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        super(my_lstm, self).__init__()
        
         # Save Parameters
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.lstm_units = lstm_units
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.dropout = dropout
        
         # Build Model
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Input Layer
        x = Input(shape=(None,), batch_size = batch_size, name = 'Input')
        
        # Word Embedding Layer
        y = Embedding(input_dim, embed_dim,
                      batch_input_shape=[batch_size, None])(x)
        
        # First LSTM Layer with BatchNormalization and Dropout
        y = LSTM(lstm_units,
                 return_sequences=True,
                 recurrent_initializer='glorot_uniform',
                 stateful=True)(y)
        
        y = BatchNormalization()(y)
        
        if dropout > 0:
            y = Dropout(dropout)(y)
        
        # Second LSTM Layer with BatchNormalization and Dropout
        y = LSTM(lstm_units,
                 return_sequences=True,
                 recurrent_initializer='glorot_uniform',
                 stateful=True)(y)
        
        y = BatchNormalization()(y)
        
        if dropout > 0:
            y = Dropout(dropout)(y)
        
        # Dense Layer with Batch Normalization and Dropout
        if max_norm:
            kernel_constraint = tf.keras.constraints.MaxNorm(max_value=max_norm)
        else:
            kernel_constraint = None
        
        y = Dense(dense_units, kernel_constraint = kernel_constraint)(y)
        
        y = BatchNormalization()(y)
        
        y = Activation('relu')(y)
        
        if dropout > 0:
            y = Dropout(dropout)(y)
        
        # Output Layer
        y = Dense(input_dim, activation = 'softmax')(y)
        
        self.model = Model(inputs = x, outputs = y)
        
    def call(self, x):
        return self.model(x)
    
    def lstm_loss(self, x, y):
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = loss_function(y, tf.math.log(x))
        
        return loss
    

# Define DataSequence Generator Class
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class DataSequence(Sequence):
    def __init__(self, dataset, batch_size, seq_length, steps_per_epoch):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, epoch_step):
        # Length of dataset
        M = self.dataset.shape[0]
        
        # randomly choose the starting indices for the examples in the training batch
        idx = np.random.choice(M-self.seq_length, self.batch_size)
        
        # Using indices, slice dataset into batches of length seq_length      
        input_batch = [self.dataset[i:i+self.seq_length] for i in idx]
        output_batch = [self.dataset[i+1:i+self.seq_length+1] for i in idx]
          
        # x_batch, y_batch provide the true inputs and targets for network training
        x_batch = np.reshape(input_batch, [self.batch_size, self.seq_length])
        y_batch = np.reshape(output_batch, [self.batch_size, self.seq_length])

        return x_batch, y_batch
        
        