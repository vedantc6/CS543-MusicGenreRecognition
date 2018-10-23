#%%
import os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import functools

DATA_DIR = os.getcwd() + "/Data/fma_metadata/"

#%%
genres_df = pd.read_csv(DATA_DIR + "genres.csv", encoding="latin-1", index_col=0) 
print(genres_df.head())
#%%
tracks_df = pd.read_csv(DATA_DIR + "tracks.csv", encoding="Latin-1")

# Cleaning data
idx = tracks_df.columns.get_loc("track_title")
tracks_df = tracks_df.iloc[:, :53]
print(tracks_df.head())
#%%
echonest_df = pd.read_csv(DATA_DIR + "echonest.csv", encoding="Latin-1", index_col=0, header=[0,1,2])
print(echonest_df.head())
#%%
artist_df = pd.read_csv(DATA_DIR + "raw_artists.csv", encoding="Latin-1")
print(artist_df.head())
#%%
album_df = pd.read_csv(DATA_DIR + "raw_albums.csv", encoding="Latin-1")
print(album_df.head())
#%%
features_df = pd.read_csv(DATA_DIR + "features.csv", encoding="Latin-1", index_col=0, header=[0,1,2])
print(features_df.head())
#%%
col_tracks = list(tracks_df.columns.values)
#%%
# Missing values in datasets
def num_missing(x):
    i = sum(x.isnull())
    return (i/len(x))*100

print("Missing values per column: ")
print("Tracks dataset: ")
missing_track_info, rm_tracks = tracks_df.apply(num_missing, axis = 0) 
print(missing_track_info)
print("\nGenres dataset: " )
missing_genre_info = genres_df.apply(num_missing, axis = 0)
print(missing_genre_info)
print("\nEchonest dataset: " )
missing_echo_info = echonest_df.apply(num_missing, axis = 0)
print(missing_echo_info)
print("\nFeatures dataset: " )
missing_features_info = features_df.apply(num_missing, axis = 0)
print(missing_features_info)

# Delete columns which have more than 30% data missing


#%%
# Number of unique number of tracks, artists, albums, genres
print('{} tracks, {} artists, {} albums, {} genres'.format(len(tracks_df), 
      len(artist_df['artist_id'].unique()), len(album_df), sum(genres_df['#tracks'] > 0)))
#%%

