# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:15:27 2020

@author: Mitchell
"""

### Imports
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import glob, pickle, json
import numpy as np
import os, random
import pandas as pd


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### Names of People
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load Filenames
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
names_folder = 'names/'
name_files = glob.glob(names_folder+'*')


# Iteratively Grab Names From US Census Files
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initate set for storing names
unique_names = set()

# Iterative over each files
for filename in name_files:
    current_data = pd.read_csv(filename, header = None, encoding = 'UTF-8')
    current_names = set(current_data[0])
    unique_names = unique_names.union(current_names)


# Grab Names from Other Files and Combine
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Filenames
other_file_1 = 'all_first_names.txt'
other_file_2 = 'all_surnames.txt'

# Read in names from files
with open(other_file_1, 'r', encoding = 'UTF-8') as f:
    text = f.read()
    other_names_1 = text.split('\n')
    
with open(other_file_2, 'r', encoding = 'UTF-8') as f:
    text = f.read()
    other_names_2 = text.split('\n')
    
# Combine all names
unique_names = unique_names.union(set(other_names_1), set(other_names_2))
    
print("Number of unique names: {}".format(len(unique_names)))


# Save combined unique names to new file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

new_filename = 'combined_names.txt'
with open(new_filename, 'w', encoding = 'UTF-8') as f:
    f.write(('\n'.join(unique_names))[1:])
    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### Names of Places
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
filename = 'worldcities.csv'
place_data = pd.read_csv(filename, encoding = 'UTF-8')


# Grab unique city, country, and territory names and combine
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cities_1 = set(place_data['city'])
cities_2 = set(place_data['city_ascii'])
countries = set(place_data['country'])
territories = set(place_data['admin_name'])

# remove nulls from datanames and combine
all_places = []
datasets = [cities_1, cities_2, countries, territories]
for data in datasets:
    for place in data:
        if type(place) == str:
            all_places.append(place)

all_places = set(all_places)


# Save places ti new file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
new_filename = 'combined_places.txt'
with open(new_filename, 'w', encoding = 'UTF-8') as f:
    f.write('\n'.join(all_places))
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#----------------------------------END FILE------------------------------------
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~