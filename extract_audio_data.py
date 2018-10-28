#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 00:31:01 2018

@author: vedantc6
"""
import os

ROOT = os.getcwd()

AUDIO_DIR = ROOT + "/Data/fma_small"

def get_track_ids(AUDIO_DIR):
    track_ids = []
    for root, dirnames, files in os.walk(AUDIO_DIR):
        if dirnames == []:
            track_ids.append(int(file[:-4]) for file in files)
    
    return track_ids

