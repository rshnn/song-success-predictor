"""
Utilities for handling song metadata in .db 
"""

import sqlite3
import numpy as np
import pandas as pd
import random

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn import metrics

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

        partition_point_1 = int(len(songs) * config.TRAIN_FRAC)
        partition_point_2 = partition_point_1 + int(len(songs) * config.CV_FRAC)

        random.shuffle(songs)

        train_data = songs[:partition_point_1]
        test_data = songs[partition_point_1 + 1:partition_point_2]
        cv_data = songs[partition_point_2 + 1:] 

        return train_data, cv_data, test_data


    def create_dataframes(self, training_list, crossvalid_list, testing_list):
        """
        transforms the list outputs of 'get_datasets()'
        RETURN dataframes: training_DF, crossvalid_list, testing_DF 
        """

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

        training_DF = pd.DataFrame(training_list, columns=column_names)
        cv_DF = pd.DataFrame(crossvalid_list, columns=column_names)
        testing_DF = pd.DataFrame(testing_list, columns=column_names)

        return training_DF, cv_DF, testing_DF


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



    def generate_energy_measure(self, training_DF, cv_DF, testing_DF):
        """
        adds energy measure values for all rows in input dataframes
        RETURN training_df, cv_df, testing_df
        """

        for DF in [training_DF, cv_DF, testing_DF]:

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


        return training_DF, cv_DF, testing_DF



    def generate_dance_measure(self, training_DF, cv_DF, testing_DF):
        """
        adds energy measure values for all rows in input dataframes
        RETURN training_DF, cv_DF, testing_DF
        """

        for DF in [training_DF, cv_DF, testing_DF]:

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


        return training_DF, cv_DF, testing_DF



    def normalize_numeric_columns(training_DF, cv_DF, testing_DF):

        numeric_cols = training_DF._get_numeric_data().columns 


        return training_DF, cv_DF, testing_DF








    def RunAndTestLinearRegModel(self, string_of_features, training_DF, testing_DF):
        """
        Put in raw string of features.  Will run sklearn.linear_model.LinearRegression() to predict hotttnesss.
        Errors are calculated and returned as tuples.
        Returns: 
            model class
            training_error (mean_absolute_error, mean_squared_error, mean_error)
            testing_error (mean_absolute_error, mean_squared_error, mean_error)
            std dev of hotttnesss
        """
        X_cols = string_of_features.split()
        X = training_DF[X_cols]
        y_true_training = training_DF['song_hotttnesss']
        y_true_testing = testing_DF['song_hotttnesss']

        linreg = LinearRegression()
        linreg.fit(X, y_true_training)

        # Predicting the training_set
        y_pred_training = linreg.predict(X)

        # Predicting the testing_set
        y_pred_testing = linreg.predict(testing_DF[X_cols])


        # Calculating errors
        mean_abs = metrics.mean_absolute_error(y_true_training, y_pred_training) 
        mean_sq = metrics.mean_squared_error(y_true_training, y_pred_training)
        mean_err = np.sqrt(metrics.mean_squared_error(y_true_training, y_pred_training))
        training_error = (mean_abs, mean_sq, mean_err)

        mean_abs = metrics.mean_absolute_error(y_true_testing, y_pred_testing) 
        mean_sq = metrics.mean_squared_error(y_true_testing, y_pred_testing)
        mean_err = np.sqrt(metrics.mean_squared_error(y_true_testing, y_pred_testing))
        testing_error = (mean_abs, mean_sq, mean_err)

        # Getting std deviation of hotttnesss    
        total_DF = training_DF.append(testing_DF)
        hot_std = total_DF['song_hotttnesss'].std()

        return linreg, training_error, testing_error, hot_std




    def RunAndTestSVRModel(self, string_of_features, training_DF, testing_DF):
        """
        Put in raw string of features.  Will run sklearn.linear_model.LinearRegression() to predict hotttnesss.
        Errors are calculated and returned as tuples.
        Returns: 
            model class
            training_error (mean_absolute_error, mean_squared_error, mean_error)
            testing_error (mean_absolute_error, mean_squared_error, mean_error)
            std dev of hotttnesss
        """
        X_cols = string_of_features.split()
        X = training_DF[X_cols]
        y_true_training = training_DF['song_hotttnesss']
        y_true_testing = testing_DF['song_hotttnesss']

        # linreg = SVR(kernel='poly', C=1e3, degree=2)
        # linreg.fit(X, y_true_training)

        # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        # svr_rbf.fit(X, y_true_training)
        svr_lin = SVR(kernel='linear', C=0.5)
        svr_lin.fit(X, y_true_training)
        # svr_poly = SVR(kernel='poly', C=1e3, degree=2)
        # svr_poly.fit(X, y_true_training)
        # y_rbf = svr_rbf.fit(X, y).predict(X)
        # y_lin = svr_lin.fit(X, y).predict(X)
        # y_poly = svr_poly.fit(X, y).predict(X)


        # Predicting the training_set
        # y_pred_training_rbf = svr_rbf.predict(X)
        y_pred_training_lin = svr_lin.predict(X)
        # y_pred_training_poly = svr_poly.predict(X)

        # Predicting the testing_set
        # y_pred_testing_rbf = svr_rbf.predict(testing_DF[X_cols])
        y_pred_testing_lin = svr_lin.predict(testing_DF[X_cols])
        # y_pred_testing_poly = svr_poly.predict(testing_DF[X_cols])


        # Calculating errors
        # mean_abs_rbf = metrics.mean_squared_error(y_true_training, y_pred_training_rbf) 
        mean_abs_lin = metrics.mean_squared_error(y_true_training, y_pred_training_lin) 
        # mean_abs_poly = metrics.mean_squared_error(y_true_training, y_pred_training_poly) 
        training_error = (0, mean_abs_lin, 0)
        # training_error = (mean_abs_rbf, mean_abs_lin, mean_abs_poly)
        

        # mean_abs_rbf = metrics.mean_squared_error(y_true_testing, y_pred_testing_rbf) 
        mean_abs_lin = metrics.mean_squared_error(y_true_testing, y_pred_testing_lin) 
        # mean_abs_poly = metrics.mean_squared_error(y_true_testing, y_pred_testing_poly)
        testing_error = (0, mean_abs_lin, 0)
        # testing_error = (mean_abs_rbf, mean_abs_lin, mean_abs_poly)

        # Getting std deviation of hotttnesss    
        total_DF = training_DF.append(testing_DF)
        hot_std = total_DF['song_hotttnesss'].std()

        return training_error, testing_error, hot_std



    def RunAndTestKNNModel(self, string_of_features, training_DF, testing_DF, num_neighbors=3):
        """
        Put in raw string of features.  Will run sklearn.neighbors.KNeighborsClassifier() to predict hotttnesss.
        Errors are calculated and returned as tuples.
        Returns:
            model class
            training_error (mean_absolute_error, mean_squared_error, mean_error)
            testing_error (mean_absolute_error, mean_squared_error, mean_error)
            std dev of hotttnesss
        """

        X_cols = string_of_features.split()
        X = training_DF[X_cols]
        y_true_training = training_DF['song_hotttnesss']
        y_true_testing = testing_DF['song_hotttnesss']

        out_y_true_training = []
        for ele in y_true_training:
            integer = int(ele * 10)
            out_y_true_training.append(integer)

        out_y_true_testing = []
        for ele in y_true_testing:
            integer = int(ele * 10)
            out_y_true_testing.append(integer)

        KNN = KNeighborsClassifier(n_neighbors=num_neighbors)
#         print X
#         print y_true_training[0]
        KNN.fit(X, out_y_true_training)

        # Predicting the training_set
        y_pred_training = KNN.predict(X)

        Y = testing_DF[X_cols]
        # Predicting the testing_set
        y_pred_testing = KNN.predict(Y)


        print "Score for Training: \t" + str(KNN.score(X, out_y_true_training))
        print "Score for Testing: \t" + str(KNN.score(Y, out_y_true_testing))

        y_pred_training = [round(i/10.0, 1) for i in y_pred_training]

        y_pred_testing = [round(i/10.0, 1)  for i in y_pred_testing]

        out_y_true_training = [i/10.0 for i in out_y_true_training]

        out_y_true_testing = [i/10.0 for i in out_y_true_testing]
        # print y_pred_testing
        # print out_y_true_testing


#         print accuracy_score()


        # Calculating training errors
        mean_abs = metrics.mean_absolute_error(y_true_training, y_pred_training)
        mean_sq =  metrics.mean_squared_error(y_true_training, y_pred_training)
        mean_err = np.sqrt(metrics.mean_squared_error(y_true_training, y_pred_training))
        training_error = (mean_abs, mean_sq, mean_err)

        # Calculating errors for discrete values
        discrete_mean_abs = metrics.mean_absolute_error(out_y_true_training, y_pred_training)
        discrete_mean_sq =  metrics.mean_squared_error(out_y_true_training, y_pred_training)
        discrete_mean_err = np.sqrt(metrics.mean_squared_error(out_y_true_training, y_pred_training))
        discrete_training_error = (discrete_mean_abs, discrete_mean_sq, discrete_mean_err)


        # Calculating testing errors
        mean_abs = metrics.mean_absolute_error(y_true_testing, y_pred_testing)
        mean_sq =  metrics.mean_squared_error(y_true_testing, y_pred_testing)
        mean_err = np.sqrt(metrics.mean_squared_error(y_true_testing, y_pred_testing))
        testing_error = (mean_abs, mean_sq, mean_err)

        # Calculating errors for discrete values
        discrete_mean_abs = metrics.mean_absolute_error(out_y_true_testing, y_pred_testing)
        discrete_mean_sq =  metrics.mean_squared_error(out_y_true_testing, y_pred_testing)
        discrete_mean_err = np.sqrt(metrics.mean_squared_error(out_y_true_testing, y_pred_testing))
        discrete_testing_error = (discrete_mean_abs, discrete_mean_sq, discrete_mean_err)

        # Getting std deviation of hotttnesss
        total_DF = training_DF.append(testing_DF)
        hot_std = total_DF['song_hotttnesss'].std()

        return KNN, training_error, testing_error, hot_std, discrete_training_error, discrete_testing_error
