#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 23:12:12 2018

@author: vedantc6
"""
import numpy as np
import librosa as lbr

METADATA_DIR = "/home/vedantc6/Desktop/Projects/CS543-MusicGenreRecognition/Data/fma_metadata/"
AUDIO_DIR = "/home/vedantc6/Desktop/Projects/CS543-MusicGenreRecognition/Data/fma_small/"
DATA_DIR = "/home/vedantc6/Desktop/Projects/CS543-MusicGenreRecognition/Data/"
MAIN_DIR = "/home/vedantc6/Desktop/Projects/CS543-MusicGenreRecognition/"

GENRES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 
          'International', 'Pop', 'Rock']

# Some hard-coded variables, which can be adjusted as per requirement
TRACK_COUNT = 8000
WINDOW_SIZE = 2048
WINDOW_STRIDE = WINDOW_SIZE // 2
N_MELS = 128
MEL_KWARGS = {
    'n_fft': WINDOW_SIZE,
    'hop_length': WINDOW_STRIDE,
    'n_mels': N_MELS
}

# Input - Filename and forceShape (to bring uniformity to the feature shape)
# To try - Spectrograms of a song divided by some duration, sliding window types
def load_track(filename, forceShape=None):
    sample_input, sample_rate = lbr.load(filename, mono=True)
    features = lbr.feature.melspectrogram(sample_input, **MEL_KWARGS).T
    
    if forceShape is not None:
        if features.shape[0] < forceShape[0]:
            delta_shape = (forceShape[0] - features.shape[0], forceShape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0)
        elif features.shape[0] > forceShape[0]:
            features = features[: forceShape[0], :]
    
    features[features == 0] = 1e-6
    
    return (np.log(features), float(sample_input.shape[0]) / sample_rate)

## USING LIBROSA EXAMPLE
#y, sr = librosa.load(AUDIO_DIR + "001/001039.mp3", duration=0.1)
#
#S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#
#plt.figure(figsize=(10, 4))
#librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time')
#plt.colorbar(format='%+2.0f dB')
#plt.title('Mel spectrogram')
#plt.tight_layout()