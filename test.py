#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 23:10:56 2018

@author: vedantc6
"""
#%%
import os
import pandas as pd
import ast
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

METADATA_DIR = "/home/vedantc6/Desktop/Projects/CS543-MusicGenreRecognition/Data/fma_metadata/"
AUDIO_DIR = "/home/vedantc6/Desktop/Projects/CS543-MusicGenreRecognition/Data/fma_small/"
DATA_DIR = "/home/vedantc6/Desktop/Projects/CS543-MusicGenreRecognition/Data/"
#%%
def cleanTracksData(filename):
    tracks = pd.read_csv(filename, index_col = 0, header=[0, 1])
    
    COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres')]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ('small', 'medium', 'large')
    tracks['set', 'subset'] = tracks['set', 'subset'].astype('category', categories=SUBSETS, ordered=True)
    
    COLUMNS = [('track', 'license'), ('artist', 'bio'), ('album', 'type'), ('album', 'information')]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype('category')
    
    return tracks
#%%
genres = pd.read_csv(METADATA_DIR + "genres.csv", index_col = 0) 
tracks = cleanTracksData(METADATA_DIR + "tracks2.csv")
#%%
# Input - tracks and genres dataset
# Return - list of genres to be predicted (present in small subset of data)
def validGenres(tracks, genres):
    subset = tracks['set', 'subset'] <= "small"
    d = genres.reset_index().set_index('title')
    d = d.loc[tracks.loc[subset, ('track', 'genre_top')].unique()]
    
    return list(d.index)
#%%
validGenres = validGenres(tracks, genres)
#%%
# USING LIBROSA EXAMPLE
y, sr = librosa.load(AUDIO_DIR + "001/001039.mp3", duration=0.1)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

#%%
# Input - Audio directory and tracks dataset
# Return - a list [track_number, track_path, track_genre]
def getTrackIDs(aud_dir, tracks):
    track_ids = []
    
    small = tracks[tracks['set', 'subset'] <= 'small']
    
    trackGenres = pd.DataFrame(small['track', 'genre_top'])
    trackGenres.columns = ['Genre']
    trackGenres.reset_index(level=0, inplace=True)
    
    for root, dirnames, files in os.walk(aud_dir):
        if dirnames == []:
            for file in files:
                f = int(file[:-4])
                g = list(trackGenres[trackGenres['track_id'] == f]['Genre'])
                track_ids.append((f,root,g[0]))
             
    return track_ids

trackIDs = getTrackIDs(AUDIO_DIR, tracks)

#%%
# Input - Main directory, tracks dataset
# Return - Pickled data
def createDataStructure(data_dir, track):
    categ_dir = data_dir + "Categorized/"
    if not os.path.exists(categ_dir):
        os.makedirs(categ_dir)
      
    for genre in validGenres: 
        g = categ_dir + genre
        if not os.path.exists(g):
            os.makedirs(g)
    
    for t in track:
        

createDataStructure(DATA_DIR, trackIDs)