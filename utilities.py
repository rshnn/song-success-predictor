"""
Utilities for handling song metadata in .db 
"""
 
import os
import sys
import time
import glob
import datetime
import sqlite3
import json
import numpy as np
import pandas as pd
import random

# config.py contains configuration constants 
import config


class Utilities(object):
    """ Holds utility functions for queries and creating data structures """

    def __init__(self, features, use_json=False):
        self.features = features
        self.use_json = use_json
        if not self.use_json:
            self.db = sqlite3.connect(config.DB_FILENAME)


    def __teardown__(self):
        """cleanup class instances"""
        if not self.use_json:
            self.db.close()



    def get_datasets(self):
        """ 
        Queries for sets to be used as training and testing sets 
        RETURN: training_data, test_data as lists 
        """
        query = ' '.join(['SELECT * FROM songs', config.HOTFILTER])
        response = self.db.execute(query)
        songs = response.fetchall()

        partition_point = int(len(songs)*config.TRAIN_FRAC)
        random.shuffle(songs)
        train_data = songs[:partition_point]
        test_data = songs[partition_point+1:]
        return  train_data, test_data



    def get_hotttnesss_list(self):
        """
        Queries to get all hotttnesss scores
        RETURN: list of floats 
        """
        if not self.use_json:
            query = ' '.join(['SELECT song_hotttnesss FROM songs', config.HOTFILTER])
            response = self.db.execute(query)
            hotttnesss = response.fetchall()
            hotttnesss = [value[0] for value in hotttnesss]
        return hotttnesss


    def get_average_hotttnesss(self):
        """
        Queries to get all hotttnesss scores and gets average
        RETURN: float 
        """
        query = ' '.join(['SELECT song_hotttnesss FROM songs', config.HOTFILTER])
        response = self.db.execute(query)
        hotttnesss = response.fetchall()
        hotttnesss = [value[0] for value in hotttnesss]
        return sum(hotttnesss)/float(len(hotttnesss))



    def create_dataframes(self, training_list, testing_list):
        """
        transforms the list outputs of 'get_datasets()'
        RETURN two dataframes: training_DF, testing_DF 
        """

        column_names = "track_id \
                        title               \
                        song_id             \
                        release             \
                        artist_id           \
                        artist_mbid         \
                        artist_name         \
                        duration            \
                        artist_familiarity  \
                        artist_hotttnesss   \
                        year                \
                        track_7digitalid    \
                        shs_perf            \
                        shs_work            \
                        song_hotttnesss     \
                        danceability        \
                        energy              \
                        key                 \
                        tempo               \
                        loudness            \
                        time_signature".split()

        training_DF = pd.DataFrame(training_list, columns=column_names)
        testing_DF = pd.DataFrame(testing_list, columns=column_names)
        return training_DF, testing_DF