#%%
import os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

sns.set()

DATA_DIR = os.getcwd() + "/Desktop/Projects/CS543-MusicGenreRecognition/Data/fma_metadata/"

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
# Number of unique number of tracks, artists, albums, genres
print('{} tracks, {} artists, {} albums, {} genres'.format(len(tracks_df), 
      len(artist_df['artist_id'].unique()), len(album_df), sum(genres_df['#tracks'] > 0)))
#%%
#track_cols = tracks_df.columns
#percent_missing_tracks = tracks_df.isnull().sum()*100/len(tracks_df)
#missing_value_tracks_df = pd.DataFrame({'ColName': track_cols,
#                                        'percent_missing': percent_missing_tracks})

def missing_values_plotter(x):
    track_cols = x.columns
    percent_missing_tracks = x.isnull().sum()*100/len(x)
    missing_value_df = pd.DataFrame({'ColName': track_cols,
                                            'percent_missing': percent_missing_tracks})
    g = sns.barplot(missing_value_df[missing_value_df['percent_missing'] > 0]['ColName'], 
                missing_value_df[missing_value_df['percent_missing'] > 0]['percent_missing'])

    for item in g.get_xticklabels():
        item.set_rotation(90)
    

missing_values_plotter(tracks_df)
      
#%%