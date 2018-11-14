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

DATA_DIR = "/home/vedantc6/Desktop/Projects/CS543-MusicGenreRecognition/Data/fma_metadata/"
AUDIO_DIR = "/home/vedantc6/Desktop/Projects/CS543-MusicGenreRecognition/Data/fma_small/"
#%%
def cleanTracksData(filename):
    tracks = pd.read_csv(DATA_DIR + filename, index_col = 0, header=[0, 1])
    
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
genres = pd.read_csv(DATA_DIR + "genres.csv", index_col = 0) 
tracks = cleanTracksData("tracks2.csv")
#%%
def getTrackIDS(AUDIO_DIR):
    track_ids = []
    for root, dirnames, files in os.walk(AUDIO_DIR):
        if dirnames == []:
            track_ids.append(int(file[:-4]) for file in files)
    
    return track_ids
#%%
def validGenres(tracks, genres):
    subset = tracks['set', 'subset'] <= "small"
    d = genres.reset_index().set_index('title')
    d = d.loc[tracks.loc[subset, ('track', 'genre_top')].unique()]
    return list(d.index)
#%%
GENRES = validGenres(tracks, genres)