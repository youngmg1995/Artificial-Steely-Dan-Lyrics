# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:47:08 2020

@author: Mitchell
"""

# Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import load_data
from lstm import my_lstm, DataSequence
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


### Load Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# filenames
filename_1 = '../lyrics_data/all_song_lyrics_2.json'
filename_2 = '../lyrics_data/SD_song_lyrics_2.json'

# Load dataset with all songs
training_dataset, validation_dataset, int2text, text2int =\
    load_data(filename_1, load_mappings = True)

# Load dataset with just Steely Dan songs
SD_training_dataset, SD_validation_dataset = load_data(filename_2)


### Build Models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Model Parameters
input_dim = len(text2int)
embed_dim = 256
batch_size = 2
lstm_units = 512
dense_units = 512
max_norm = None,
dropout = 0.5

model = my_lstm(input_dim,
                embed_dim,
                batch_size,
                lstm_units,
                dense_units,
                max_norm = None,
                dropout = 0.)
model.build(tf.TensorShape([batch_size, None]))
model.summary()


### Train Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Training Parameters
seq_length = 100
steps_per_epoch = 100
validation_steps = 10
epochs = 100

# Cost Function
cost_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)#model.lstm_loss

# Learning_rate schedule
lr_0 = .001
decay_rate = 1.0
lr_decay = lambda t: lr_0 * decay_rate**t
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_decay)

# Optimizer
optimizer = tf.keras.optimizers.Adam()

# Define callbacks
callbacks = [lr_schedule]

# Keras Sequences for Datasets (need to use since one-hot datasets too
# large for storing in memory)
training_seq = DataSequence(training_dataset, batch_size, seq_length,
                            steps_per_epoch)
validation_seq = DataSequence(validation_dataset, batch_size, seq_length,
                              validation_steps)

# Compile Model
model.compile(optimizer = optimizer,
              loss = cost_function)

# Train model
tic = time.perf_counter()
history = model.fit(training_seq,
                    epochs = epochs,
                    callbacks = callbacks,
                    validation_data = validation_seq,
                    use_multiprocessing = True)
toc = time.perf_counter()
print(f"Trained Model in {(toc - tic)/60:0.1f} minutes")