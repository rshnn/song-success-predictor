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
