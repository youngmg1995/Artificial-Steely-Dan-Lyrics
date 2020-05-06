# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:19:12 2020

@author: Mitchell
"""
# Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import load_data, lyrics_encoder, lyrics_decoder, reformat_lyrics
from bigram import bigram
import numpy as np
import matplotlib.pyplot as plt


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
# build general model using all songs
print('Building bigram model using all song lyrics.')
model = bigram(training_dataset)

# build Steely Dan model using just Steely Dan songs
print('Building bigram model using Steely Dan song lyrics.')
SD_model = bigram(SD_training_dataset)


### Test Models Against Validation Datasets
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# set whether or not to run validation test (since it takes forever)
run_validation = False

if run_validation:
    # Loss and Accuracy for general model against all songs and SD songs
    model_loss = model.get_loss(validation_dataset)
    model_accuracy = model.get_accuracy(validation_dataset)
    model_SD_loss = model.get_loss(SD_validation_dataset)
    model_SD_accuracy = model.get_accuracy(SD_validation_dataset)
    
    # Loss and Accuracy for Steely Dan model against all songs and SD songs
    SD_model_loss = SD_model.get_loss(validation_dataset)
    SD_model_accuracy = SD_model.get_accuracy(validation_dataset)
    SD_model_SD_loss = SD_model.get_loss(SD_validation_dataset)
    SD_model_SD_accuracy = SD_model.get_accuracy(SD_validation_dataset)    
    
    ### Plot losses and accuracies
    # set bar widths and axis ticks
    ind = np.arange(2)  # the x locations for the groups
    width = 0.33  # the width of the bars
    
    # Losses
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, [model_loss, SD_model_loss],
                    width, color = 'b', label = 'All Songs')
    rects2 = ax.bar(ind + width/2, [model_SD_loss, SD_model_SD_loss],
                    width, color = 'r', label = 'Steely Dan Songs')
    ax.set_ylabel('Loss')
    ax.set_title('Mean Squared Loss On Validation Data')
    ax.set_xticks(ind)
    ax.set_xticklabels(('General\nBigram', 'Steely Dan\nBigram'))
    ax.legend()    
    fig.tight_layout() 
    plt.show()
    
    # Accuracies
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width/2, [model_accuracy, SD_model_accuracy],
                    width, color = 'b', label = 'All Songs')
    rects2 = ax.bar(ind + width/2, [model_SD_accuracy, SD_model_SD_accuracy],
                    width, color = 'r', label = 'Steely Dan Songs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracies On Validation Data')
    ax.set_xticks(ind)
    ax.set_xticklabels(('General\nBigram', 'Steely Dan\nBigram'))
    ax.legend()    
    fig.tight_layout() 
    plt.show()
    

### Create Seeds for Song Generation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# write song seeds, being sure to include necessary song, title, artist, line
# start and end indicators
seed_text_1 = "xxx000 xxx010 steely dan xxx011 xxx100 brother michael xxx101"
seed_text_2 = "xxx000 xxx010 steely dan xxx011 xxx100 jacob's street xxx101"\
    +" xxx110 friday night strolling in uptown xxx111"\
    +" xxx110 how the boys had fun and the girls got down xxx111"

# Converting textual lyrics to integer sequences
seed_1 = lyrics_encoder(seed_text_1, text2int)
seed_2 = lyrics_encoder(seed_text_2, text2int)


### Generate New Songs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate new songs either by specifying the length of the output sequence
# or the number of songs to generate

# define generation parameters
num_songs = 2
seq_length = 600    # About length of 2 average songs

# Generate lyrics using seeds and general model
model_int_lyrics_1 = model.gen_lyrics(seed_1.copy(), gen_method = 0,
                                      num_songs = num_songs,
                                      keep_seed = True, song_end = 91160)
model_int_lyrics_2 = model.gen_lyrics(seed_2.copy(), gen_method = 1,
                                      seq_length = seq_length,
                                      keep_seed = True)

# Generate lyrics using seeds and SD model
SD_model_int_lyrics_1 = SD_model.gen_lyrics(seed_1.copy(), gen_method = 0,
                                            num_songs = num_songs,
                                            keep_seed = True, song_end = 91160)
SD_model_int_lyrics_2 = SD_model.gen_lyrics(seed_2.copy(), gen_method = 1,
                                            seq_length = seq_length,
                                            keep_seed = True)


### Convert to Text
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Convert lyrics to text
model_ugly_lyrics_1 = lyrics_decoder(model_int_lyrics_1.copy(), int2text)
model_ugly_lyrics_2 = lyrics_decoder(model_int_lyrics_2.copy(), int2text)
SD_model_ugly_lyrics_1 = lyrics_decoder(SD_model_int_lyrics_1.copy(), int2text)
SD_model_ugly_lyrics_2 = lyrics_decoder(SD_model_int_lyrics_2.copy(), int2text)

# Reformat lyrics by removing special line indicators and separating lines 
model_lyrics_1 = reformat_lyrics(model_ugly_lyrics_1.copy())
model_lyrics_2 = reformat_lyrics(model_ugly_lyrics_2.copy())
SD_model_lyrics_1 = reformat_lyrics(SD_model_ugly_lyrics_1.copy())
SD_model_lyrics_2 = reformat_lyrics(SD_model_ugly_lyrics_2.copy())


### Save Songs
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# decide whether or not to save songs
save_lyrics = True

# filenames
filenames = ['model_lyrics_1.txt',
             'model_lyrics_2.txt',
             'SD_model_lyrics_1.txt',
             'SD_model_lyrics_2.txt']

# lyrics
lyrics = [model_lyrics_1,
          model_lyrics_2,
          SD_model_lyrics_1,
          SD_model_lyrics_2]

# save songs
if save_lyrics:
    for i in range(len(lyrics)):
        text = '\n\n'.join(lyrics[i])
        filename = filenames[i]
        with open(filename, 'wt') as f:
            n = f.write(text)
            

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------END FILE------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~