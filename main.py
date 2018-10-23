#%%
import os
import pandas as pd
import matplotlib.pyplot as plt 
import ast

DATA_DIR = os.getcwd() + "/Data/fma_metadata/"

#%%
genres_df = pd.read_csv(DATA_DIR + "genres.csv", encoding="latin-1", index_col=0) 
print(genres_df.head())
#%%
tracks_df = pd.read_csv(DATA_DIR + "tracks.csv", encoding="Latin-1", header=[0,1])

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
features_df = pd.read_csv(DATA_DIR + "features.csv", encoding="Latin-1", index_col=0)
print(features_df.head())
#%%
# Key values about the dataset
unique_titles_count = tracks_df["track_id"].count()
unique_artist_count = artist_df["artist_id"].count()
unique_album_count = album_df["album_id"].count()
unique_genre_count = genres_df["genre_id"].count()
#%%
col_tracks = list(tracks_df.columns.values)
#%%
# Missing values in datasets
def num_missing(x):
    return sum(x.isnull())

print("Missing values per column: ")
#print(tracks_df.apply(num_missing, axis = 0))
print(genres_df.apply(num_missing, axis = 0))
#%%
plt.plot(genres_df['genre_id'], genres_df['#tracks'])
plt.show()