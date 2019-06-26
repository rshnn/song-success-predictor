"""
Utilities for handling song metadata in .db 
"""

import sqlite3
import numpy as np
import pandas as pd
import random
from sklearn import preprocessing

# config.py contains configuration constants 
import config


class Utilities(object):
    """ 
        Holds utility functions for queries and creating data structures
        Keeps pointer to sqlite db
    """

    def __init__(self, features, use_json=False):
        self.features = features
        self.use_json = use_json
        if not self.use_json:
            self.db = sqlite3.connect(config.DB_FILENAME)


    def __teardown__(self):
        """cleanup class instances"""
        if not self.use_json:
            self.db.close()



    def get_master_dataframe(self):
        """ 
        Queries to genreate master dataset.  Returns as Pandas DF  
        RETURN: master_DF 
        """
        query = ' '.join(['SELECT * FROM songs', config.HOTFILTER])
        response = self.db.execute(query)
        songs = response.fetchall()

        random.shuffle(songs)

        column_names = "track_id            \
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
                        time_signature      \
                        segments_avg        \
                        tatums_avg          \
                        beats_avg           \
                        bars_avg            \
                        sections_avg".split()

        master_DF = pd.DataFrame(songs, columns=column_names)

        return master_DF




    def split_master_df(self, master_DF):
        """
        Splits master dataframe into training, validation, and test sets
        Division paramters come from config.py  
        RETURN:  dataframes for training, cross-validation, and test sets 
        """
        np.random.seed(100) 
        m = len(master_DF) 

        split1 = config.TRAIN_FRAC
        split2 = split1 + config.CV_FRAC

        train, cv, test = np.split(master_DF, 
                                   [int(split1 * m), int(split2 * m)])

        return train, cv, test 





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
        return sum(hotttnesss) / float(len(hotttnesss))



    def generate_energy_measure(self, master_DF):
        """
        adds energy measure values for all rows in input dataframes
        RETURN master_DF
        """
        DF = master_DF 

        loudness = DF['loudness']
        tempo = DF['tempo']
        time_sig = DF['time_signature']
        sections_avg = DF['sections_avg']
        beats_avg = DF['beats_avg']
        tatums_avg = DF['tatums_avg']
        art_fam = DF['artist_familiarity']
        art_hot = DF['artist_hotttnesss']

        DF['energy1'] = art_fam * (50 + loudness) / 100
        DF['energy2'] = (50 + loudness) * 3 * art_fam / (500 * beats_avg**0.5)
        DF['energy3'] = (50 + loudness) * art_fam / (100 * tatums_avg**0.25)
        DF['energy4'] = (50 + loudness) * 2 * art_fam / (100000 * tatums_avg * beats_avg)**0.5

        # OLD HEURISTICS 
        # DF['energy1'] = (50+loudness)**2*(12-time_sig)/1000
        # DF['energy2'] = (50+loudness)**2*(12-time_sig)/(5000*beats_avg)
        # DF['energy3'] = (50+loudness)**2*(12-time_sig)/(10000*tatums_avg)
        # DF['energy4'] = (50+loudness)**2*(12-time_sig)/(10000*tatums_avg*beats_avg)

        return DF



    def generate_dance_measure(self, master_DF):
        """
        adds energy measure values for all rows in input dataframes
        RETURN master_DF
        """

        DF = master_DF

        loudness = DF['loudness']
        tempo = DF['tempo']
        time_sig = DF['time_signature']
        key = DF['key']
        sections_avg = DF['sections_avg']
        beats_avg = DF['beats_avg']
        tatums_avg = DF['tatums_avg']
        art_fam = DF['artist_familiarity']
        art_hot = DF['artist_hotttnesss']

        DF['dance1'] = (12 - time_sig) * (tempo)**0.5 * (50 + loudness) * art_hot / 1000
        DF['dance2'] = (12 - time_sig)**0.5 * (tempo)**0.5 * (50 + loudness) * 2 * art_hot / 1000
        DF['dance3'] = (12 - time_sig)**0.5 * (tempo) * (50 + loudness) ** 2 * art_fam / 1000000
        DF['dance4'] = (12 - time_sig)**0.5 * (tempo) * (50 + loudness) * art_fam / 10000

        # OLD HUERISTICS 
        # DF['dance1'] = (12-time_sig)**2*(tempo)*(50+loudness)/10000
        # DF['dance2'] = (12-time_sig)*(tempo)**2*(50+loudness)/(2500000)
        # DF['dance3'] = (12-time_sig)*(tempo)*(50+loudness)**2/(1000000)
        # DF['dance4'] = (12-time_sig)**2*(tempo)*(50+loudness)/(100000)


        return DF



    def normalize_numeric_columns(self, master_DF):
        """
        Normalizes the numeric features of the input dataframe.
        Non-numeric features are also copied to the returned dataframe 
        RETURN: normalized_DF 
        """

        X = master_DF._get_numeric_data()

        cols = set(master_DF.columns) - set(X.columns)
        X_vals = X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X_vals)
        normalized_DF = pd.DataFrame(X_scaled, columns=X.columns)

        for col in cols: 
            normalized_DF[col] = master_DF[col]

        return normalized_DF
