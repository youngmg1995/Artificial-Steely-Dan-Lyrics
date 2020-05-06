# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:00:45 2020

@author: Mitchell
"""

import numpy as np
import json

def load_data(filename, load_mappings = False):
    print('Reloading Pre-Transformed Training and Validation Data from Filename "{}"'\
          .format(filename))
    
    # load data from file
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Build datasets from file contents
    training_dataset = np.array(data["training_dataset"])
    validation_dataset = np.array(data["validation_dataset"])
    
    # If load_mappings = True then build vectorizing and inverse mappings
    if load_mappings:
        int2text = np.array(data["unique_values"])
        text2int = {word:i for i, word in enumerate(int2text)}
        output = (training_dataset, validation_dataset, int2text, text2int)
    else:
        output = (training_dataset, validation_dataset)        
    
    # Print that we are done
    print('Loading Data Complete')
    
    return output

def lyrics_encoder(lyrics, text2int):
    # Initialize list for storing vectorized text
    int_lyrics = []
    
    # Iterate over lyrics and use text2int mapping to convert to ints
    for word in lyrics.split(' '):
        int_lyrics.append(text2int[word])
    
    # Convert list to numpy array
    int_lyrics = np.array(int_lyrics)
    
    return int_lyrics

def lyrics_decoder(int_lyrics, int2text, song_end = 91160):
    # Initialize list for storing each individual song
    songs = []
    
    # Find song ends
    song_ends = np.where(int_lyrics == song_end)[0]
    
    # split lyrics up into individual songs
    song_start = 0
    for i in range(len(song_ends)):
        song_end = song_ends[i]
        songs.append(int_lyrics[song_start:song_end+1])
        song_start = song_end+1
    if song_start != len(int_lyrics):
        songs.append(int_lyrics[song_start:])
    
    # Convert each song from sequence of ints to string of text
    for i in range(len(songs)):
        song = songs[i]
        song = int2text[song]
        song = ' '.join(song)
        songs[i] = song
        
    return songs

def reformat_lyrics(songs,
                    artist_and_title = False,
                    artist_indicator = 'xxx010',
                    title_indicator  = 'xxx100',
                    start_indicators = ['xxx000','xxx010','xxx100','xxx110'],
                    end_indicators   = ['xxx001','xxx011','xxx101','xxx111'],
                    punctuation = '",:;.!?'):
    # Iterate over songs to remove indicators and separate lines
    for i in range(len(songs)):
        song = ' '+songs[i]+' '
        # indicate artist
        song = song.replace(artist_indicator+' ', 'Artist: ')
        # indicate title
        song = song.replace(title_indicator+' ', 'Title: ')
        # remove line start indicators
        for indicator in start_indicators:
            song = song.replace(indicator+' ', '')
        # remove line end indicators
        for indicator in end_indicators:
            song = song.replace(indicator+' ', '\n')
        # fix punctuation
        for char in punctuation:
            song = song.replace(' '+char, char)
        # strip whitespace from start and end
        song = song.strip()
        # add back to list
        songs[i] = song
        
    return songs