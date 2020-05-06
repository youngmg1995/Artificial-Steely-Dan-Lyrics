# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:00:45 2020

@author: Mitchell
"""
import numpy as np

class bigram(object):
    def __init__(self, dataset):
        self.bigram = build_bigram(dataset)
    
    def get_prob(self, word_1, word_2):
        try:
            p = self.bigram[word_1][word_2]
        except KeyError:
            p = 0.
        return p
    
    def get_distr(self, word_1):
        return self.bigram[word_1]
    
    def get_sample(self, word_1, n = 1):
        distr = self.bigram[word_1]
        x = list(distr.keys())
        p = list(distr.values())
        return np.random.choice(x, n, p = p)
    
    def get_accuracy(self, validation_dataset):
        # Grab length of validation_dataset
        N = len(validation_dataset)
        
        # Keep track of correct outputs
        # Note: output considered correct if word with highest probability
        # of being returned is same as true next word
        correct_preds = 0
        for i in range(N-1):
            word_1 = validation_dataset[i]
            word_2 = validation_dataset[i+1]
            if word_1 in self.bigram:
                pred_distr = self.bigram[word_1]
                max_pred = max(pred_distr, key = pred_distr.get)
                if max_pred == word_2:
                    correct_preds += 1
        
        # calculate accuracy
        accuracy = correct_preds / (N-1)
        
        return accuracy
    
    def get_loss(self, validation_dataset):
        # Grab length of validation_dataset
        N = len(validation_dataset)
        
        # Keep track of total loss
        loss = 0.
        
        # Calculate loss between each word in sequence and model predictions
        for i in range(N-1):
            word_1 = validation_dataset[i]
            word_2 = validation_dataset[i+1]
            if word_1 in self.bigram:
                pred_distr = self.bigram[word_1]
                loss += self.mean_squared(word_2, pred_distr)
            else:
                loss += 2.
        
        mean_loss = loss / (N-1)
        
        return mean_loss
        
    
    def mean_squared(self, y, pred_distr):
        loss = 0.
        if y not in pred_distr:
            loss += 1.
        for y_hat in pred_distr:
            p = pred_distr[y_hat]
            if y_hat == y:
                loss += (1. - p)**2
            else:
                loss += p**2
        
        return loss
    
    def gen_lyrics(self, seed, gen_method = 0, num_songs = 1, seq_length = 300,
                 keep_seed = True, song_end = 91160):
        # 0 generates num_songs number of songs while 1 generates single
        # song sequence with length seq_length
        if gen_method == 0:
            # Initialize list for generated lyrics
            lyrics = []
            # Set current word in lyrics
            current_word = seed[-1]
            # Add seed if specified
            if keep_seed:
                lyrics += seed.tolist()
            # Define counter for keeping track of # of songs generated
            # Considered end of song if song end indicator generated
            songs_generated = 0
            # Iteratively generate songs
            while songs_generated < num_songs:
                current_word = self.get_sample(current_word)[0]
                lyrics.append(current_word)
                if current_word == song_end:
                    songs_generated += 1
                    current_word = seed[-1]
                    if keep_seed and songs_generated < num_songs:
                        lyrics += seed.tolist()
            # Convert lyrics to np.array
            lyrics = np.array(lyrics)
                
        elif gen_method == 1:
            # Initialize list for generated lyrics
            lyrics = []
            # Set current word in lyrics
            current_word = seed[-1]
            # Iteratively generate new lyrics
            for i in range(seq_length):
                current_word = self.get_sample(current_word)[0]
                lyrics.append(current_word)
            # Convert lyrics list to np.array
            lyrics = np.array(lyrics)
            # Add seed if specified
            if keep_seed:
                lyrics = np.concatenate((seed, lyrics))
        
        return lyrics
    
    
def build_bigram(dataset):
    # Get length of dataset
    N = len(dataset)
    
    # Initiate dictionary for bigram
    bigram = {}
    
    # Iterate over dataset and build bigram based on frequencies of word
    # pairings
    for i in range(N-1):
        # Grab words in current pair
        word_1 , word_2 = dataset[i] , dataset[i+1]
        
        # Update bigram based on pairing instance
        # First create dictionary in bigram for word_1 if not yet encountered
        if word_1 not in bigram:
            bigram[word_1] = {}
        # Next update P(word_2|word_1) in bigram
        try:
            bigram[word_1][word_2] += 1
        except KeyError:
            bigram[word_1][word_2] = 1
    
    # Convert frequencies in bigram to probabilities
    for word_1 in bigram:
        frequencies = bigram[word_1]
        total_count = sum(frequencies.values())
        for word_2 in frequencies:
            bigram[word_1][word_2] /= total_count
    
    return bigram