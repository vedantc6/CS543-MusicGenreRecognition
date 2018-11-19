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
from pickle import dump
import numpy as np
from common import METADATA_DIR, AUDIO_DIR, DATA_DIR, TRACK_COUNT, load_track
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
genres_data = pd.read_csv(METADATA_DIR + "genres.csv", index_col = 0) 
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
# Valid genres present in the subset "small". Total 8 in number
# One Hot Encoding done and stored in a dictionary
GENRES = validGenres(tracks, genres_data)
GENRES = sorted(GENRES)
genresDict = {}

labelEncoded = LabelEncoder().fit_transform(GENRES)
labelEncoded = labelEncoded.reshape(len(labelEncoded), 1)
oneHotEncoder = OneHotEncoder(sparse=False)
oneHotEncoded = oneHotEncoder.fit_transform(labelEncoded)

for i, genre in enumerate(GENRES):
    genresDict[genre] = np.array(oneHotEncoded[i])

del i, genre
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
# Input - Audio file
# Return - Shape of a melspectrogram
# To try - Find out different values of shapes, take median as the default shape 
def getDefaultShape():
    tempFeatures, _ = load_track(AUDIO_DIR + "009/009152.mp3")
    return tempFeatures.shape

# Input - Main directory, tracks dataset
# Return - Pickled data
def createDataStructure(data_dir, trackList):
    defaultShape = getDefaultShape()
    
#    np.zeros makes 8000 lists which have rows and columns shaped according to defaultShape
    X = np.zeros((TRACK_COUNT,) + defaultShape, dtype=np.float32)
    y = np.zeros((TRACK_COUNT, len(GENRES)), dtype=np.float32)
    track_paths = {}
#    print(X.shape, y.shape)
    
    for i, track in enumerate(trackList[:5]):
        path = track[1] + "/" + str(track[0]) + ".mp3"
        X[i], _ = load_track(path, defaultShape)
        y[i] = genresDict[track[2]]
        track_paths[track[0]] = path
        
    return (X, y, track_paths)
        

(X, y, track_paths) = createDataStructure(DATA_DIR, trackIDs)
data = {'X': X, 'y': y, 'track_paths': track_paths}
with open(DATA_DIR + "data.pkl", 'wb') as f:
    dump(data, f)